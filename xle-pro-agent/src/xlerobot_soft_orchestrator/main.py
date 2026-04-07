"""CLI entry point for quick local checks."""

from __future__ import annotations

import argparse

from .agent import run_directive


def main() -> None:
    parser = argparse.ArgumentParser(description="Run software-only orchestration.")
    parser.add_argument("directive", help="User directive to process")
    args = parser.parse_args()
    result = run_directive(args.directive)
    print(result.final_response)
    print("\nTrace:")
    for event in result.trace:
        print(f"- [{event['stage']}] {event['detail']}")


if __name__ == "__main__":
    main()

