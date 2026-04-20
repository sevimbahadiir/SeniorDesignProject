"""
run_all.py
----------
Pipeline orchestration script for the Robot Shop RL data collection runs.

Coordinates three concurrent processes for each run:
    1. random_scaler.py   -- scales pod counts using structured exploration
    2. robot_shop.py      -- generates realistic HTTP traffic via Locust
    3. collect_metrics.py -- queries Prometheus after the run ends

After each run the services are reset to 2 replicas and a configurable
cool-down period is observed before the next run starts.

Run plan
--------
Runs 1-5 cover the standard load range used in prior experiments.
Runs 6-7 (250 and 300 users) are capacity boundary tests placed last
so that any system instability does not affect the core dataset.

    Run 1:   40 users  |  12 hours  |  ~4,320 rows
    Run 2:   80 users  |  12 hours  |  ~4,320 rows
    Run 3:  120 users  |  12 hours  |  ~4,320 rows
    Run 4:  150 users  |  12 hours  |  ~4,320 rows
    Run 5:  200 users  |  12 hours  |  ~4,320 rows
    Run 6:  250 users  |  12 hours  |  ~4,320 rows  (capacity boundary)
    Run 7:  300 users  |  12 hours  |  ~4,320 rows  (capacity boundary)

    Total: ~84 hours | ~30,240 rows (before quality filtering)

Each run saves its own CSV file. Runs 6 and 7 may be discarded during
the data processing stage if the zero-latency fraction exceeds 30%.

Usage
-----
    # Run the full plan (all 7 runs)
    python run_all.py

    # Run specific runs only
    python run_all.py --runs 1 2 3

    # Preview commands without executing
    python run_all.py --dry-run

    # Change the cool-down between runs (seconds, default: 120)
    python run_all.py --cooldown 120

Requirements
------------
    - kubectl configured and pointing at the robot-shop cluster
    - Port-forwarding active (start_portforward.ps1 must be running)
    - random_scaler.py, robot_shop.py, collect_metrics.py in the same directory
    - locust installed: pip install locust
"""

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Run:
    id:           int
    users:        int
    ramp_up:      int    # seconds
    duration_min: int
    output_file:  str


RUN_PLAN: list[Run] = [
    Run(1,   40,  5,  720, "dataset_run1_40user.csv"),
    Run(2,   80, 10,  720, "dataset_run2_80user.csv"),
    Run(3,  120, 15,  720, "dataset_run3_120user.csv"),
    Run(4,  150, 20,  720, "dataset_run4_150user.csv"),
    Run(5,  200, 25,  720, "dataset_run5_200user.csv"),
    Run(6,  250, 30,  720, "dataset_run6_250user.csv"),   # capacity boundary
    Run(7,  300, 35,  720, "dataset_run7_300user.csv"),   # capacity boundary
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NAMESPACE        = "robot-shop"
ROBOT_SHOP_HOST  = "http://localhost:8080"
LOCUST_FILE      = "robot_shop.py"
SCALER_SCRIPT    = "random_scaler.py"
COLLECT_SCRIPT   = "collect_metrics.py"

# Seconds to wait after Locust finishes before querying Prometheus.
# Allows the final scrape interval and rate window to complete.
POST_RUN_SETTLE  = 60

SERVICES = [
    "cart", "catalogue", "payment", "shipping",
    "user", "ratings", "dispatch",
]


# ---------------------------------------------------------------------------
# Kubernetes helpers
# ---------------------------------------------------------------------------

def reset_pods(replicas: int = 2) -> None:
    """Scale all managed services to replicas."""
    logger.info("Resetting all services to %d replicas.", replicas)
    for svc in SERVICES:
        subprocess.run(
            [
                "kubectl", "scale", "deployment", svc,
                f"--replicas={replicas}",
                "-n", NAMESPACE,
            ],
            capture_output=True,
        )


def wait_for_pods(timeout_seconds: int = 180) -> bool:
    """
    Block until all pods in the namespace report 2/2 containers running,
    or until timeout_seconds elapses.

    Returns True if all pods are ready, False on timeout.
    """
    logger.info("Waiting for pods to reach 2/2 Running state.")
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", NAMESPACE, "--no-headers"],
            capture_output=True,
            text=True,
        )
        lines     = result.stdout.strip().splitlines()
        not_ready = [l for l in lines if "2/2" not in l and l.strip()]

        if not not_ready:
            logger.info("All pods are ready.")
            return True

        logger.info(
            "%d pod(s) not yet ready. Retrying in 10 s.", len(not_ready)
        )
        time.sleep(10)

    logger.warning(
        "Pod readiness timeout after %d s. Proceeding anyway.", timeout_seconds
    )
    return False


# ---------------------------------------------------------------------------
# Single run execution
# ---------------------------------------------------------------------------

def execute_run(run: Run, dry_run: bool = False) -> bool:
    """
    Execute one data collection run.

    Starts random_scaler.py and Locust as background processes, waits for
    both to finish, then invokes collect_metrics.py.

    Parameters
    ----------
    run:     Run configuration dataclass.
    dry_run: If True, log commands without executing them.

    Returns
    -------
    True if the run completed without error, False otherwise.
    """
    logger.info(
        "--- Run %d | %d users | ramp-up %d s | duration %d min | output: %s ---",
        run.id, run.users, run.ramp_up, run.duration_min, run.output_file,
    )

    scaler_cmd = [
        sys.executable, SCALER_SCRIPT,
        "--duration", str(run.duration_min),
        "--interval", "90",
    ]
    locust_cmd = [
        "locust", "-f", LOCUST_FILE,
        f"--host={ROBOT_SHOP_HOST}",
        "--headless",
        "-u", str(run.users),
        "-r", str(run.ramp_up),
        "--run-time", f"{run.duration_min}m",
        "--stop-timeout", "30",
    ]
    collect_cmd = [
        sys.executable, COLLECT_SCRIPT,
        "--window", str(run.duration_min),
        "--step",   "10s",
        "--out",    run.output_file,
    ]

    if dry_run:
        logger.info("[DRY-RUN] scaler  : %s", " ".join(scaler_cmd))
        logger.info("[DRY-RUN] locust  : %s", " ".join(locust_cmd))
        logger.info("[DRY-RUN] collect : %s", " ".join(collect_cmd))
        return True

    # Prepare: reset pods and wait for stability
    reset_pods(2)
    wait_for_pods()

    # Launch scaler and Locust concurrently
    logger.info("Launching random_scaler.py.")
    scaler_proc = subprocess.Popen(scaler_cmd)

    logger.info("Launching Locust.")
    locust_proc = subprocess.Popen(locust_cmd)

    # Wait for Locust to finish (bounded by --run-time)
    locust_proc.wait()
    logger.info("Locust finished.")

    # Terminate scaler if still running
    if scaler_proc.poll() is None:
        logger.info("Terminating random_scaler.py.")
        scaler_proc.terminate()
        scaler_proc.wait()
    logger.info("Scaler finished.")

    # Allow Prometheus rate window to complete
    logger.info(
        "Waiting %d s for Prometheus to finalise metrics.", POST_RUN_SETTLE
    )
    time.sleep(POST_RUN_SETTLE)

    # Collect metrics
    logger.info("Running collect_metrics.py.")
    result = subprocess.run(collect_cmd)

    if result.returncode != 0:
        logger.error("Metric collection failed for Run %d.", run.id)
        return False

    logger.info("Run %d complete. Dataset: %s", run.id, run.output_file)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate all Robot Shop RL data collection runs."
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        metavar="ID",
        default=None,
        help=(
            "Run IDs to execute (e.g. --runs 1 2 3). "
            "Executes all runs if omitted."
        ),
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=120,
        metavar="SECONDS",
        help="Cool-down period between runs in seconds (default: 120).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_ids = args.runs or [r.id for r in RUN_PLAN]
    plan         = [r for r in RUN_PLAN if r.id in selected_ids]

    if not plan:
        valid = [r.id for r in RUN_PLAN]
        logger.error("No valid run IDs. Available: %s", valid)
        sys.exit(1)

    total_min = sum(r.duration_min for r in plan) + (
        (len(plan) - 1) * args.cooldown // 60
    )
    eta = datetime.now() + timedelta(minutes=total_min)

    print()
    print("=" * 70)
    print("  Robot Shop RL -- Data Collection Pipeline")
    print("=" * 70)
    print(
        f"  {'Run':<5}  {'Users':>6}  {'Duration':>10}  "
        f"{'Note':<20}  Output"
    )
    print("  " + "-" * 65)
    for r in plan:
        note = "capacity boundary" if r.id in (6, 7) else ""
        print(
            f"  {r.id:<5}  {r.users:>6}  "
            f"{r.duration_min:>8} min  {note:<20}  {r.output_file}"
        )
    print("  " + "-" * 65)
    print(f"  Estimated total : ~{total_min // 60} h {total_min % 60} min")
    print(f"  Estimated end   : {eta.strftime('%d %b %H:%M')}")
    print()
    print(
        "  Note: Runs 6 and 7 are capacity boundary tests. They will be\n"
        "  discarded during data processing if zero-latency > 30%."
    )
    print("=" * 70)
    print()

    if not args.dry_run:
        try:
            input("Press ENTER to start, or Ctrl+C to abort.\n")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    wall_start = time.time()
    failures: list[int] = []

    for i, run in enumerate(plan):
        success = execute_run(run, dry_run=args.dry_run)
        if not success:
            failures.append(run.id)

        if i < len(plan) - 1 and not args.dry_run:
            logger.info(
                "Cool-down: %d s before next run.", args.cooldown
            )
            reset_pods(2)
            time.sleep(args.cooldown)

    elapsed = int(time.time() - wall_start)
    print()
    print("=" * 70)
    print("  Pipeline complete.")
    print(f"  Elapsed : {elapsed // 3600} h {(elapsed % 3600) // 60} min")
    if failures:
        print(f"  Failed runs : {failures}")
    else:
        print("  All runs completed successfully.")
    print("=" * 70)
    print()

    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
