"""Low-frequency nvidia-smi samples, not a CUDA profiler or GPU idle estimate."""
import asyncio
import csv
import io
import os
import subprocess
import threading

from .logs import JsonlWriter


class GPUMonitor:
    def __init__(self, directory):
        self.writer = JsonlWriter(directory/'gpu-samples.jsonl')
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True, name='gpu-monitor')
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--id='+os.environ['CUDA_VISIBLE_DEVICES'],
                     '--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                    text=True, capture_output=True, check=True, timeout=3)
                uuid, name, util, used, total = next(csv.reader(io.StringIO(result.stdout)))
                def number(value):
                    try:
                        return float(value)
                    except ValueError:
                        return None
                self.writer.emit(uuid=uuid.strip(), name=name.strip(), utilization_percent=number(util),
                                 used_mib=number(used), total_mib=number(total))
            except Exception as exc:
                try:
                    self.writer.emit(error=repr(exc))
                except Exception as writer_error:
                    self.error = writer_error
                    break
            self.stop.wait(1.)

    async def close(self):
        self.stop.set()
        await asyncio.to_thread(self.thread.join)
        await asyncio.to_thread(self.writer.close)
        if self.error:
            raise self.error
