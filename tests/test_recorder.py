import asyncio
import json

import numpy as np
from PIL import Image
import pytest

from experiment_runner import Recorder
from experiment_runner.logs import read_jsonl, JsonlWriter


@pytest.mark.parametrize('dump,gif', [(False, False), (True, False), (False, True), (True, True)])
def test_media_switches_and_logs(tmp_path, dump, gif):
    async def run():
        recorder = Recorder(tmp_path, dump_images=dump, gif_on=gif)
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        await recorder.observation(step=0, image=image, height=10.)
        for i in range(1, 12):
            image[:] = i * 20
            await recorder.step(step=i, image=image, action='wait', reward=float(i),
                done=False, height=10.-i)
        image[:] = 255  # mutation after enqueue must not change the recorded frames
        await recorder.finish()
        await recorder.finish()
    asyncio.run(run())
    rows = list(read_jsonl(tmp_path/'steps.jsonl'))
    assert [r['height'] for r in rows] == [10.-i for i in range(12)]
    assert len(list(tmp_path.glob('frames/*.png'))) == (12 if dump else 0)
    assert (tmp_path/'episode.gif').exists() == gif
    if dump:
        assert np.asarray(Image.open(tmp_path/'frames/000001.png'))[0, 0, 0] == 20
    if gif:
        with Image.open(tmp_path/'episode.gif') as image:
            assert image.n_frames == 12
            image.seek(1)
            assert image.convert('RGB').getpixel((0, 0)) == (20, 20, 20)
    if not dump and not gif:
        assert {p.suffix for p in tmp_path.iterdir()} == {'.jsonl'}


def test_writer_snapshot_and_closed(tmp_path):
    writer = JsonlWriter(tmp_path/'log.jsonl')
    values = [1]
    writer.emit(values=values)
    values.append(2)
    writer.close()
    assert list(read_jsonl(tmp_path/'log.jsonl'))[0]['values'] == [1]
    with pytest.raises(RuntimeError):
        writer.emit(other=1)


def test_media_failure_does_not_deadlock(tmp_path):
    async def run():
        recorder = Recorder(tmp_path, gif_on=True)
        def fail(*args):
            raise OSError('disk failed')
        recorder._save_frame = fail
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        with pytest.raises(OSError):
            for i in range(30):
                await recorder.observation(step=i, image=image)
            await recorder.finish()
        with pytest.raises(OSError):
            await recorder.finish()
    asyncio.run(asyncio.wait_for(run(), 3))
