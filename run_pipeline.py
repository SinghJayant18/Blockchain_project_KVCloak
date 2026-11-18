#!/usr/bin/env python3
"""
KV-CLOAK Automated End-to-End Pipeline Runner
Paper: "Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference" (NDSS 2026)
GitHub: https://github.com/SiO-2/kvcloak

Usage:
  python run_pipeline.py --help
  python run_pipeline.py --step all
  python run_pipeline.py --step data
  python run_pipeline.py --step bench --device cpu --num-trials 5
  python run_pipeline.py --step demo
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
PYTHON_EXE = sys.executable


def run_command(cmd: list, desc: str):
    print(f"\n>>> Running {desc}...")
    print(f"Command: {' '.join(cmd)}")
    env = os.environ.copy()
    env["KVCLOAK_ALLOW_ATTACKS"] = "1"
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if result.returncode != 0:
        print(f"[-] Step '{desc}' exited with code {result.returncode}")
        return False
    print(f"[+] Step '{desc}' completed successfully.")
    return True


def step_data():
    """Verify or generate sample datasets."""
    print(">>> Verifying sample datasets...")
    for ds_name in ["lmsys-chat-1m_1k.jsonl", "alpaca_1k.jsonl", "gsm8k_1k.jsonl"]:
        p = PROJECT_ROOT / "dataset" / ds_name
        if p.exists():
            lines = sum(1 for _ in open(p, "r", encoding="utf-8"))
            print(f"  ✓ {ds_name} present ({lines} records)")
        else:
            print(f"  ! {ds_name} not found. Running demo generator...")
            run_command([PYTHON_EXE, "demo.py"], "Initialize datasets")
            break


def step_demo():
    """Run full simulated paper verification demo."""
    run_command([PYTHON_EXE, "demo.py"], "End-to-End KV-Cloak Demonstration")


def step_bench(device: str = "cpu", trials: int = 5, model: str = "gpt2"):
    """Run micro-benchmarks comparing protection overheads."""
    cmd = [
        PYTHON_EXE,
        "defense/eval/micro_benchmark.py",
        "--device", device,
        "--num-trials", str(trials),
        "--batch-size", "2",
        "--min-seq-len", "64",
        "--max-seq-len", "128",
        "--seq-step", "64",
        "--model-name", model,
    ]
    run_command(cmd, "Micro-benchmarks (AES vs KV-Cloak Fused / No-Fuse)")


def step_tests():
    """Run full unit test suite."""
    cmd = [PYTHON_EXE, "-m", "pytest", "tests/", "-v"]
    run_command(cmd, "Pytest unit test suite")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Pipeline Runner for KV-Cloak (NDSS 2026)"
    )
    parser.add_argument(
        "--step",
        choices=["all", "data", "demo", "bench", "tests"],
        default="demo",
        help="Pipeline step to execute.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use (cpu, mps, cuda:0).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials for benchmarking.",
    )
    parser.add_argument(
        "--model",
        default="gpt2",
        help="Model name for benchmark.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" KV-CLOAK: LLM KV-CACHE PRIVACY PROTECTION PIPELINE")
    print("=" * 70)

    if args.step == "all":
        step_data()
        step_tests()
        step_bench(device=args.device, trials=args.trials, model=args.model)
        step_demo()
    elif args.step == "data":
        step_data()
    elif args.step == "demo":
        step_demo()
    elif args.step == "bench":
        step_bench(device=args.device, trials=args.trials, model=args.model)
    elif args.step == "tests":
        step_tests()


if __name__ == "__main__":
    main()
