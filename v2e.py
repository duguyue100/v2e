#!/usr/bin/env python

import argparse

from v2ecore.config import V2EConfig
from v2ecore.pipeline import V2EPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v2e: generate synthetic DVS events from video"
    )
    # minimal stub parser
    args, other_args = parser.parse_known_args()

    config = V2EConfig.from_args(args)
    pipeline = V2EPipeline(config)

    if config.input.synthetic_input:
        # TODO: load synthetic
        # synth = load_synthetic_input(config.input.synthetic_input, other_args)
        # stats = pipeline.process_synthetic(synth)
        pass
    else:
        if config.input.input_path:
            pipeline.process_video(config.input.input_path)

    pipeline.cleanup()


if __name__ == "__main__":
    main()
