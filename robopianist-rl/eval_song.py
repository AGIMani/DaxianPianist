#!/usr/bin/env python3
"""Run a trained eval_daxian checkpoint on a (possibly different) song."""

from __future__ import annotations

import pickle
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import tyro

import sac
import specs
import train


def _args_from_pkl(path: Path) -> train.Args:
    with path.open("rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        return raw
    valid = {f.name for f in fields(train.Args)}
    raw = {k: v for k, v in raw.items() if k in valid}
    cfg = raw.get("agent_config")
    if isinstance(cfg, dict):
        raw["agent_config"] = sac.SACConfig(**cfg)
    if raw.get("record_resolution") is not None:
        raw["record_resolution"] = tuple(raw["record_resolution"])
    if raw.get("record_dir"):
        raw["record_dir"] = Path(raw["record_dir"])
    return train.Args(**raw)


def _load_agent(ckpt_path: Path, spec: specs.EnvironmentSpec, args: train.Args) -> sac.SAC:
    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)
    agent = sac.SAC.initialize(
        spec=spec,
        config=args.agent_config,
        seed=args.seed,
        discount=args.discount,
    )
    return agent.replace(
        actor=agent.actor.replace(params=ckpt["actor_params"]),
        rng=ckpt.get("rng", agent.rng),
    )


def main(
    ckpt: Path = Path("/home/houjue/pianist_daxian_v2/eval_daxian/checkpoints/latest.pkl"),
    args_pkl: Path = Path("/home/houjue/pianist_daxian_v2/eval_daxian/args.pkl"),
    environment_name: str = "RoboPianist-debug-NocturneRousseau-v0",
    out_dir: Path = Path("/home/houjue/pianist_daxian_v2/eval_daxian/videos"),
) -> None:
    args = _args_from_pkl(args_pkl)
    args = replace(args, environment_name=environment_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = train.get_env(args, record_dir=out_dir)
    spec = specs.EnvironmentSpec.make(env)
    agent = _load_agent(ckpt, spec, args)

    timestep = env.reset()
    while not timestep.last():
        timestep = env.step(agent.eval_actions(timestep.observation))

    stats = env.get_statistics()
    music = env.get_musical_metrics()
    print("song", environment_name)
    print("ckpt", ckpt)
    print("return", float(stats["return"]), "length", float(stats["length"]))
    for key in ("precision", "recall", "f1"):
        print(key, float(music[key]))
    src = Path(env.latest_filename)
    dest = out_dir / f"eval_{environment_name.split('-')[-2]}.mp4"
    if src.exists() and src.resolve() != dest.resolve():
        dest.write_bytes(src.read_bytes())
        mp3 = src.with_suffix(".mp3")
        if mp3.exists():
            dest.with_suffix(".mp3").write_bytes(mp3.read_bytes())
    print("video", dest if dest.exists() else src)


if __name__ == "__main__":
    tyro.cli(main)
