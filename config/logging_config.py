"""Loguru-based structured logging configuration.

All logs are written to server.log as JSON lines for full traceability.
Stdlib logging is intercepted and funneled to loguru.
Context vars (request_id, node_id, chat_id) from contextualize() are
included at top level for easy grep/filter.
"""

import asyncio
import json
import logging
import os

from loguru import logger

_configured = False

# Context keys we promote to top-level JSON for traceability
_CONTEXT_KEYS = ("request_id", "node_id", "chat_id")


def _serialize_with_context(record) -> str:
    """Format record as JSON with context vars at top level.
    Returns a format template; we inject _json into record for output.
    """
    extra = record.get("extra", {})
    out = {
        "time": str(record["time"]),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    for key in _CONTEXT_KEYS:
        if key in extra and extra[key] is not None:
            out[key] = extra[key]
    record["_json"] = json.dumps(out, default=str)
    return "{_json}\n"


class InterceptHandler(logging.Handler):
    """Redirect stdlib logging to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


async def _truncate_log_periodically(log_file: str, interval_seconds: int = 86400) -> None:
    """Periodically truncate the given log file."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            # truncate -s 0 equivalent in Python
            with open(log_file, "w", encoding="utf-8") as f:
                f.truncate()
        except Exception as e:
            logger.error(f"Failed to truncate log file {log_file}: {e}")


def configure_logging(log_file: str, *, force: bool = False) -> None:
    """Configure loguru with JSON output to log_file and intercept stdlib logging.

    Idempotent: skips if already configured (e.g. hot reload).
    Use force=True to reconfigure (e.g. in tests with a different log path).
    """
    global _configured
    if _configured and not force:
        return
    _configured = True

    # Remove default loguru handler (writes to stderr)
    logger.remove()

    # Truncate log file on fresh start for clean debugging
    open(log_file, "w", encoding="utf-8").close()

    # Add file sink: JSON lines, DEBUG level, context vars at top level
    logger.add(
        log_file,
        level="DEBUG",
        format=_serialize_with_context,
        encoding="utf-8",
        mode="a",
    )

    # Add error file sink: JSON lines, ERROR level, NEVER TRUNCATED
    error_log_file = os.path.join(os.path.dirname(log_file), "error.log")
    logger.add(
        error_log_file,
        level="ERROR",
        format=_serialize_with_context,
        encoding="utf-8",
        mode="a",
    )

    # Start background task to truncate the main log file every 24 hours
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_truncate_log_periodically(log_file))
    except RuntimeError:
        # If no event loop is running (e.g. during certain tests), we just skip setting up the truncator
        pass

    # Intercept stdlib logging: route all root logger output to loguru
    intercept = InterceptHandler()
    logging.root.handlers = [intercept]
    logging.root.setLevel(logging.DEBUG)
