"""Ctrl+C handling: the first press asks for a clean stop, a second stops at once."""
import signal


class StopRequest:
    def __init__(self, first_press_message: str):
        self.requested = False
        self.message = first_press_message
        signal.signal(signal.SIGINT, self._handle)
        if hasattr(signal, "SIGBREAK"):  # Ctrl+Break on Windows.
            signal.signal(signal.SIGBREAK, self._handle)

    def _handle(self, *_) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print(f"\n{self.message} Press Ctrl+C again to stop at once.", flush=True)
