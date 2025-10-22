"""Entry point for the Hallucination Data Synthesizer pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.utils.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the synthesis pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default_config.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the pipeline using the provided configuration."""
    configure_logging()
    args = parse_args()
    logging.info("Pipeline bootstrap placeholder. Implement orchestration in src/pipeline.")
    logging.debug("Using configuration file: %s", args.config)


if __name__ == "__main__":
    main()
