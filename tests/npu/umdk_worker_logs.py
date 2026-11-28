# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Build a vLLM logging config that preserves the first child failure on the board."""

import json
import logging
import os
import sys
from pathlib import Path


def worker_handler():
    directory = Path(os.environ["AFD_WORKER_LOG_DIR"])
    role = os.environ.get("AFD_E2E_PROCESS_ROLE", "launcher")
    prefix = directory / f"{role}-{os.getpid()}"
    # vLLM initializes its logger in each spawned worker. Capture native writes
    # to fd 2 there as well as Python tracebacks emitted through logging.
    with prefix.with_suffix(".stderr").open("ab") as stream:
        os.dup2(stream.fileno(), 2)
    return logging.FileHandler(prefix.with_suffix(".log"), encoding="utf-8")


if __name__ == "__main__":
    level = os.environ.get("VLLM_LOGGING_LEVEL", "INFO")
    streams = ["console", "worker"]
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detail": {
                "format": "%(asctime)s %(levelname)s %(name)s [%(process)d] %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "detail",
            },
            "worker": {
                "()": "tests.npu.umdk_worker_logs.worker_handler",
                "formatter": "detail",
            },
        },
        "loggers": {"vllm": {"handlers": streams, "level": level, "propagate": False}},
        "root": {"handlers": streams, "level": level},
    }
    Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
