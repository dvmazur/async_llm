"""Owned, buffered disk output; no GPU operations in telemetry."""
import json
import os
from pathlib import Path
import queue
import threading
import time


def stamp():
    return {"monotonic": time.monotonic(), "unix_time": time.time()}


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(temp, path)


class JsonlWriter:
    """Small CPU records go to one ordered writer thread, errors are propagated.

    Bounded queue fails loudly if storage cannot keep up; never blocks inference
    waiting for disk or silently drops records. Images use Recorder's async queue.
    """
    def __init__(self, path, capacity=65536):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", buffering=65536)
        self.queue = queue.Queue(capacity)
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        try:
            while True:
                row = self.queue.get()
                if row is None:
                    break
                self.file.write(row)
                if self.queue.empty():
                    self.file.flush()
        except BaseException as exc:
            self.error = exc
        finally:
            self.file.close()

    def emit(self, **row):
        if self.closed:
            raise RuntimeError("writer is closed")
        if self.error:
            raise RuntimeError(f"writer failed: {self.path}") from self.error
        # Serialize now: callers may reuse/mutate their dicts after this call.
        self.queue.put_nowait(json.dumps({**stamp(), **row}, allow_nan=False) + "\n")

    def close(self):
        if not self.closed:
            self.closed = True
            while self.thread.is_alive():
                try:
                    self.queue.put(None, timeout=.1)
                    break
                except queue.Full:
                    continue
            self.thread.join()
        if self.error:
            raise RuntimeError(f"writer failed: {self.path}") from self.error


def read_jsonl(path):
    with Path(path).open() as f:
        for line in f:
            yield json.loads(line)
