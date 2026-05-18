from typing import TypedDict


class LogContext(TypedDict, total=False):
    """Fields every future structured log line will carry.

    Populated by request middleware / span context in a later chore. Defined
    here now so call sites (and reviewers) can see the target shape.
    """

    request_id: str
    trace_id: str


def configure_logging() -> None:
    """Placeholder for future logging setup."""
