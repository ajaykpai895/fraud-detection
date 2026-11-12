"""End-to-end pipeline runner for Credit Card Fraud Detection.

Runs the complete sequence:
  1. Exploratory Data Analysis (scripts/run_eda.py)
  2. Baseline & Tuned Model Training (scripts/run_modeling.py)
  3. Cost & Sensitivity Analysis (scripts/run_cost.py)
  4. Rigorous Evaluation & Rolling Stability (scripts/run_eval_rigor.py)
  5. Feature Ablation & Alternative Models (scripts/run_phase4.py)
  6. Calibration & Capacity Constraints (scripts/run_phase5.py)
  7. SHAP Explainability & Error Analysis (scripts/run_phase6.py)
  8. Portfolio Figure Generation (scripts/generate_figures.py)
  9. Production Replay Simulation (fraud.replay)
 10. Unit Test Suite (pytest)
"""

import logging
import subprocess
import sys
import time

from fraud.config import DATA_PATH, ROOT

log = logging.getLogger(__name__)


def run_step(name: str, cmd: list[str]) -> float:
    print(f"\n{'=' * 60}")
    print(f">> RUNNING: {name}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n", flush=True)

    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"\nFAILED: {name} exited with code {res.returncode}", file=sys.stderr)
        sys.exit(res.returncode)

    print(f"\nCOMPLETED: {name} in {elapsed:.1f}s\n", flush=True)
    return elapsed


def main() -> None:
    print(f"Project root: {ROOT}")

    if not DATA_PATH.exists():
        print(f"\nERROR: Dataset not found at {DATA_PATH}", file=sys.stderr)
        print("Please download it via Kaggle API:", file=sys.stderr)
        print(
            "  kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip", file=sys.stderr
        )
        sys.exit(1)

    print(f"Dataset verified: {DATA_PATH} ({DATA_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

    timings = {}
    py = sys.executable

    steps = [
        ("01_eda", "Exploratory Data Analysis", [py, str(ROOT / "scripts" / "run_eda.py")]),
        (
            "02_modeling",
            "Model Training & Validation",
            [py, "-u", str(ROOT / "scripts" / "run_modeling.py")],
        ),
        ("03_cost", "Cost-Based Evaluation", [py, str(ROOT / "scripts" / "run_cost.py")]),
        (
            "04_eval_rigor",
            "Rigorous Bootstrap & Rolling Evaluation",
            [py, "-u", str(ROOT / "scripts" / "run_eval_rigor.py")],
        ),
        (
            "05_phase4",
            "Feature Ablation & Extended Models",
            [py, "-u", str(ROOT / "scripts" / "run_phase4.py")],
        ),
        (
            "06_phase5",
            "Calibration & Capacity Constraints",
            [py, "-u", str(ROOT / "scripts" / "run_phase5.py")],
        ),
        (
            "07_phase6",
            "SHAP Explainability & Error Analysis",
            [py, "-u", str(ROOT / "scripts" / "run_phase6.py")],
        ),
        (
            "08_figures",
            "Portfolio Figure Generation",
            [py, str(ROOT / "scripts" / "generate_figures.py")],
        ),
        ("09_replay", "Test Period Replay Simulation", [py, "-m", "fraud.replay"]),
        ("10_tests", "Unit Test Suite", [py, "-m", "pytest", "-q"]),
    ]

    for step_id, step_name, cmd in steps:
        timings[step_id] = run_step(step_name, cmd)

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE — All 10 steps succeeded cleanly")
    print(f"{'=' * 60}")
    for step, duration in timings.items():
        print(f"  {step:<20}: {duration:>6.1f}s")
    print(f"  {'Total':<20}: {sum(timings.values()):>6.1f}s\n")


if __name__ == "__main__":
    main()
