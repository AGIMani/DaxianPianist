from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import inspect
import os
import pickle
import shutil
import subprocess
import tyro
from dataclasses import dataclass, asdict
import swanlab
import time
import random
import numpy as np
import jax
from tqdm import tqdm

import sac
import specs
import replay

from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers


@dataclass(frozen=True)
class Args:
    root_dir: str = "/tmp/robopianist"
    seed: int = 42
    max_steps: int = 1_000_000
    warmstart_steps: int = 5_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    eval_episodes: int = 1
    batch_size: int = 256
    discount: float = 0.99
    tqdm_bar: bool = False
    replay_capacity: int = 1_000_000
    project: str = "robopianist"
    workspace: str = ""
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "online"
    swanlab_api_key: str = ""
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    robot: str = "daxian"
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    # Comparison run: reduced space but four-finger PIP/DIP are policy-controlled.
    unlock_four_finger_pip_dip: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = False
    frame_stack: int = 1
    clip: bool = True
    record_dir: Optional[Path] = None
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"
    action_reward_observation: bool = False
    # If set, videos / checkpoints / metric pkls are written here instead of
    # ``root_dir/<run_name>``. ``run_daxian.sh`` points this at ``eval_daxian``.
    artifact_dir: str = ""
    agent_config: sac.SACConfig = sac.SACConfig()


def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def _unwrap_attr(env, name: str):
    cur = env
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, name):
            try:
                return getattr(cur, name)
            except AttributeError:
                pass
        cur = getattr(cur, "environment", None) or getattr(cur, "_environment", None)
    return None


def _get_reward_terms(env) -> Dict[str, float]:
    task = _unwrap_attr(env, "task")
    reward_fn = getattr(task, "reward_fn", None) if task is not None else None
    if reward_fn is None:
        return {}
    return {k: float(v) for k, v in reward_fn.reward_terms.items()}


def _accumulate_reward_terms(acc: Dict[str, float], env) -> None:
    for name, value in _get_reward_terms(env).items():
        acc[name] = acc.get(name, 0.0) + value


def _reward_term_logs(acc: Dict[str, float]) -> Dict[str, float]:
    """Episode sums plus magnitude share of each reward term."""
    if not acc:
        return {}
    abs_total = sum(abs(v) for v in acc.values()) or 1.0
    logs = {f"reward/{name}": value for name, value in acc.items()}
    logs.update(
        {f"reward_frac/{name}": abs(value) / abs_total for name, value in acc.items()}
    )
    logs["reward/total"] = float(sum(acc.values()))
    signed_total = sum(acc.values())
    if abs(signed_total) > 1e-12:
        logs.update(
            {
                f"reward_contrib/{name}": value / signed_total
                for name, value in acc.items()
            }
        )
    return logs


def _scalar_dict(d: dict) -> Dict[str, float]:
    """Convert jax/numpy scalars so SwanLab always gets plain floats."""
    out: Dict[str, float] = {}
    for key, value in d.items():
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (ValueError, TypeError):
                value = float(np.asarray(value).mean())
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _tree_to_numpy(tree):
    return jax.tree_util.tree_map(
        lambda x: np.asarray(x) if hasattr(x, "__array__") else x, tree
    )


def _safe_call(fn, default=None):
    try:
        return fn()
    except (AttributeError, ValueError):
        return default


def _action_debug_logs(actions: np.ndarray) -> Dict[str, float]:
    if actions.size == 0:
        return {}
    abs_actions = np.abs(actions)
    return {
        "action_mean": float(actions.mean()),
        "action_std": float(actions.std()),
        "action_abs_mean": float(abs_actions.mean()),
        "action_abs_max": float(abs_actions.max()),
        "action_sat_frac": float(np.mean(abs_actions > 0.99)),
    }


def _save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _save_checkpoint(path: Path, agent: sac.SAC, step: int, args: Args) -> None:
    _save_pickle(
        path,
        {
            "step": int(step),
            "actor_params": _tree_to_numpy(agent.actor.params),
            "actor_opt_state": _tree_to_numpy(agent.actor.opt_state),
            "critic_params": _tree_to_numpy(agent.critic.params),
            "critic_opt_state": _tree_to_numpy(agent.critic.opt_state),
            "target_critic_params": _tree_to_numpy(agent.target_critic.params),
            "temp_params": _tree_to_numpy(agent.temp.params),
            "temp_opt_state": _tree_to_numpy(agent.temp.opt_state),
            "rng": np.asarray(agent.rng),
            "discount": float(agent.discount),
            "tau": float(agent.tau),
            "target_entropy": float(agent.target_entropy),
            "args": _jsonable(asdict(args)),
        },
    )


def _keep_eval_video(src: Path, dest_dir: Path, step: int) -> Optional[Path]:
    """Copy the eval mp4 (and sidecar mp3) to a step-named file; do not delete."""
    if not src.exists():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"eval_step_{step:08d}.mp4"
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    audio = src.with_suffix(".mp3")
    if audio.exists():
        shutil.copy2(audio, dest.with_suffix(".mp3"))
    return dest


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _swanlab_config(args: Args) -> dict:
    cfg = asdict(args)
    cfg.pop("swanlab_api_key", None)
    return _jsonable(cfg)


def _mp4_to_gif(mp4_path: Path, fps: int = 4) -> Optional[Path]:
    """Convert eval MP4 to GIF. SwanLab currently only accepts GIF videos."""
    gif_path = mp4_path.with_suffix(".gif")
    ret = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(mp4_path),
            "-vf",
            f"fps={fps},scale=320:-1:flags=lanczos",
            str(gif_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ret.returncode == 0 and gif_path.exists():
        return gif_path
    return None


def get_env(args: Args, record_dir: Optional[Path] = None):
    task_kwargs = dict(
        n_steps_lookahead=args.n_steps_lookahead,
        trim_silence=args.trim_silence,
        gravity_compensation=args.gravity_compensation,
        reduced_action_space=args.reduced_action_space,
        unlock_four_finger_pip_dip=args.unlock_four_finger_pip_dip,
        control_timestep=args.control_timestep,
        wrong_press_termination=args.wrong_press_termination,
        disable_fingering_reward=args.disable_fingering_reward,
        disable_forearm_reward=args.disable_forearm_reward,
        disable_colorization=args.disable_colorization,
        disable_hand_collisions=args.disable_hand_collisions,
        primitive_fingertip_collisions=args.primitive_fingertip_collisions,
        change_color_on_activation=True,
    )
    load_kwargs = dict(
        environment_name=args.environment_name,
        seed=args.seed,
        stretch=args.stretch_factor,
        shift=args.shift_factor,
        task_kwargs=task_kwargs,
    )
    # This repo's suite.load accepts `robot`; the original RoboPianist package does not
    # (it is Shadow-hand-only). Only pass it when the installed package supports it.
    if "robot" in inspect.signature(suite.load).parameters:
        load_kwargs["robot"] = args.robot
        task_kwargs["robot"] = args.robot
    elif args.robot not in ("shadow", "shadow_hand"):
        raise ValueError(
            f"Installed robopianist does not support robot={args.robot!r}. "
            "Use this repo's package (see run.sh PYTHONPATH) or --robot shadow."
        )
    env = suite.load(**load_kwargs)
    if record_dir is not None:
        env = robopianist_wrappers.PianoSoundVideoWrapper(
            environment=env,
            record_dir=record_dir,
            record_every=args.record_every,
            camera_id=args.camera_id,
            height=args.record_resolution[0],
            width=args.record_resolution[1],
        )
        stats_deque = args.record_every
    else:
        stats_deque = 1
    env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=stats_deque)
    env = robopianist_wrappers.MidiEvaluationWrapper(
        environment=env, deque_size=stats_deque
    )
    if args.action_reward_observation:
        env = wrappers.ObservationActionRewardWrapper(env)
    env = wrappers.ConcatObservationWrapper(env)
    if args.frame_stack > 1:
        env = wrappers.FrameStackingWrapper(
            env, num_frames=args.frame_stack, flatten=True
        )
    # Daxian: shift MCP / thumb / forearm_roll bounds so CanonicalSpec 0 is rest
    # (straight MCP, THUMB_REST_CTRL, no extra roll), not the ctrlrange midpoint.
    if str(args.robot).lower().startswith("daxian"):
        env = robopianist_wrappers.PipRestAtZeroWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=args.clip)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    return env


def main(args: Args) -> None:
    if args.name:
        run_name = args.name
    else:
        run_name = f"SAC-{args.robot}-{args.environment_name}-{args.seed}-{time.time()}"

    experiment_dir = Path(args.root_dir) / run_name
    artifact_root = Path(args.artifact_dir) if args.artifact_dir else experiment_dir
    if args.artifact_dir:
        experiment_dir = artifact_root
    video_dir = artifact_root / "videos"
    ckpt_dir = artifact_root / "checkpoints"
    metrics_dir = artifact_root / "metrics"
    for path in (experiment_dir, artifact_root, video_dir, ckpt_dir, metrics_dir):
        path.mkdir(parents=True, exist_ok=True)

    _save_pickle(artifact_root / "args.pkl", _jsonable(asdict(args)))

    # Seed RNGs.
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_key = args.swanlab_api_key or os.environ.get("SWANLAB_API_KEY", "")
    if api_key:
        swanlab.login(api_key=api_key, save=True)

    swanlab.init(
        project=args.project,
        workspace=args.workspace or None,
        experiment_name=run_name,
        description=args.notes or None,
        tags=(args.tags.split(",") if args.tags else None),
        config=_swanlab_config(args),
        mode=args.mode,
        logdir=str(artifact_root / "swanlog"),
    )

    env = get_env(args)
    eval_env = get_env(args, record_dir=video_dir)

    spec = specs.EnvironmentSpec.make(env)

    agent = sac.SAC.initialize(
        spec=spec,
        config=args.agent_config,
        seed=args.seed,
        discount=args.discount,
    )

    replay_buffer = replay.Buffer(
        state_dim=spec.observation_dim,
        action_dim=spec.action_dim,
        max_size=args.replay_capacity,
        batch_size=args.batch_size,
    )

    timestep = env.reset()
    replay_buffer.insert(timestep, None)
    train_reward_acc: Dict[str, float] = {}

    start_time = time.time()
    for i in tqdm(range(1, args.max_steps + 1), disable=not args.tqdm_bar):
        # Act.
        if i < args.warmstart_steps:
            action = spec.sample_action(random_state=env.random_state)
        else:
            agent, action = agent.sample_actions(timestep.observation)

        # Observe.
        timestep = env.step(action)
        replay_buffer.insert(timestep, action)
        _accumulate_reward_terms(train_reward_acc, env)

        # Reset episode.
        if timestep.last():
            train_logs = env.get_statistics() | _reward_term_logs(train_reward_acc)
            music = _safe_call(env.get_musical_metrics, {})
            if music:
                train_logs.update(music)
            swanlab.log(prefix_dict("train", _scalar_dict(train_logs)), step=i)
            train_reward_acc.clear()
            timestep = env.reset()
            replay_buffer.insert(timestep, None)

        # Train.
        if i >= args.warmstart_steps:
            if replay_buffer.is_ready():
                transitions = replay_buffer.sample()
                agent, metrics = agent.update(transitions)
                if i % args.log_interval == 0:
                    swanlab.log(prefix_dict("train", _scalar_dict(metrics)), step=i)

        # Eval.
        if i % args.eval_interval == 0:
            eval_reward_acc: Dict[str, float] = {}
            eval_actions = []
            for _ in range(args.eval_episodes):
                timestep = eval_env.reset()
                while not timestep.last():
                    action = agent.eval_actions(timestep.observation)
                    eval_actions.append(np.asarray(action))
                    timestep = eval_env.step(action)
                    _accumulate_reward_terms(eval_reward_acc, eval_env)
            n_eval = max(args.eval_episodes, 1)
            eval_reward_mean = {k: v / n_eval for k, v in eval_reward_acc.items()}
            eval_logs = (
                eval_env.get_statistics()
                | _reward_term_logs(eval_reward_mean)
                | (_safe_call(eval_env.get_musical_metrics, {}) or {})
                | _action_debug_logs(
                    np.stack(eval_actions) if eval_actions else np.empty(0)
                )
            )
            eval_logs = _scalar_dict(eval_logs)
            swanlab.log(prefix_dict("eval", eval_logs), step=i)

            _save_pickle(
                metrics_dir / f"eval_step_{i:08d}.pkl",
                {"step": int(i), **eval_logs},
            )
            _save_checkpoint(ckpt_dir / f"step_{i:08d}.pkl", agent, i, args)
            _save_checkpoint(ckpt_dir / "latest.pkl", agent, i, args)

            video_path = _safe_call(lambda: Path(eval_env.latest_filename))
            kept = (
                _keep_eval_video(video_path, video_dir, i)
                if video_path is not None
                else None
            )
            gif_src = kept or video_path
            gif_path = _mp4_to_gif(gif_src, fps=4) if gif_src is not None else None
            if gif_path is not None:
                swanlab.log(
                    {
                        "video": swanlab.Video(
                            str(gif_path), caption=f"eval step {i}"
                        )
                    },
                    step=i,
                )

        if i % args.log_interval == 0:
            swanlab.log({"train/fps": int(i / (time.time() - start_time))}, step=i)

    _save_checkpoint(ckpt_dir / "latest.pkl", agent, args.max_steps, args)
    swanlab.finish()


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
