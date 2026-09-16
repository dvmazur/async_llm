"""Episode-owned recorder. Media is optional; numbers never depend on PNGs."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .logs import JsonlWriter, atomic, stamp


class Recorder:
    def __init__(self, directory, *, dump_images=False, gif_on=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dump_images, self.gif_on = dump_images, gif_on
        self.events = JsonlWriter(self.directory / "events.jsonl")
        self.steps = JsonlWriter(self.directory / "steps.jsonl")
        self.queue = None
        self.task = None
        self.executor = None
        self.gif = None
        self.closed = False
        if dump_images:
            (self.directory / "frames").mkdir()

    def log(self, kind, data):
        """Synchronous small-event adapter for the existing role policy."""
        self.events.emit(kind=kind, **data)

    async def event(self, kind, **data):
        self.log(kind, data)

    def _save_frame(self, step, image):
        from PIL import Image, GifImagePlugin
        frame = Image.fromarray(image)
        if self.dump_images:
            frame.save(self.directory / "frames" / f"{step:06}.png")
        if self.gif_on:
            frame = frame.convert("P", palette=Image.Palette.ADAPTIVE)
            if self.gif is None:
                self.gif = (self.directory / "episode.gif").open("xb")
                blocks, _ = GifImagePlugin.getheader(frame, info={"loop": 0})
                for block in blocks:
                    self.gif.write(block)
            for block in GifImagePlugin.getdata(frame, duration=100, include_color_table=True):
                self.gif.write(block)

    async def _media_worker(self):
        loop = asyncio.get_running_loop()
        while True:
            item = await self.queue.get()
            if item is None:
                break
            await loop.run_in_executor(self.executor, self._save_frame, *item)

    async def _enqueue(self, item):
        put = asyncio.create_task(self.queue.put(item))
        done, _ = await asyncio.wait([put, self.task], return_when=asyncio.FIRST_COMPLETED)
        if self.task in done:
            if item is None and put.done() and not put.cancelled():
                await self.task
                return
            put.cancel()
            await asyncio.gather(put, return_exceptions=True)
            await self.task  # propagate disk/encoder failure, never deadlock a full queue
            raise RuntimeError("media writer stopped unexpectedly")
        await put

    async def _frame(self, step, image):
        if not (self.dump_images or self.gif_on):
            return
        if self.task is None:
            self.queue = asyncio.Queue(maxsize=8)
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recorder-media")
            self.task = asyncio.create_task(self._media_worker())
        await self._enqueue((step, image.copy()))

    async def observation(self, *, step, image, info=None, height=None):
        self.steps.emit(kind="observation", step=step, height=height, info=info or {})
        await self._frame(step, image)

    async def step(self, *, step, image, action, reward, done, info=None, height=None, **timings):
        self.steps.emit(kind="action", step=step, action=action, reward=reward,
                        done=done, height=height, info=info or {}, **timings)
        await self._frame(step, image)

    async def finish(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.task:
                if self.task.done():
                    await self.task
                else:
                    await self._enqueue(None)
                    await self.task
        finally:
            if self.gif:
                self.gif.write(b";")
                self.gif.close()
            if self.executor:
                self.executor.shutdown(wait=True)
            try:
                await asyncio.to_thread(self.events.close)
            finally:
                await asyncio.to_thread(self.steps.close)
