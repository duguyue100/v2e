from typing import Any
from typing import Protocol


class EventWriter(Protocol):
    """Protocol for event output writers."""

    def write(self, events: Any) -> None: ...
    def close(self) -> None: ...


class CompositeEventWriter:
    """Fans out events to multiple format writers."""

    def __init__(self, writers: list[EventWriter]) -> None:
        self.writers = writers

    def write(self, events: Any) -> None:
        for writer in self.writers:
            writer.write(events)

    def close(self) -> None:
        for writer in self.writers:
            writer.close()
