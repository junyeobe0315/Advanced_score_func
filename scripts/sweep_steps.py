from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    """Parse arguments for repeated evaluation with multiple NFE grids."""
    parser = argparse.ArgumentParser(description="Run evaluation with custom NFE sweep")
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--nfe_list", default="10,20,50,100,200", type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    return parser.parse_args()


def main() -> None:
    """Invoke module evaluator once using provided NFE sweep arguments."""
    args = parse_args()

    cmd = [sys.executable, "-m", "src.main_eval", "--run_dir", args.run_dir, "--nfe_list", args.nfe_list]
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
