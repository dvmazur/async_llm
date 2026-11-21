"""Exercise the wire protocol without loading model code or accessing GPUs."""

import argparse
import asyncio
import base64
import json
import math
from functools import partial


async def reply(reader, writer, *, imu):
    try:
        payload = json.loads(await asyncio.wait_for(reader.readline(), 12))
        if payload.get("health"):
            result = {"ready": True, "backend": "rover_transport_fixture"}
        else:
            assert payload["protocol"] == "direct_navigation_v1"
            sensors = payload["sensors"]
            assert sensors["imu_available"] is imu
            assert sensors["yaw_rate_source"] == ("imu" if imu else "odometry")
            if imu:
                gyro = sensors["angular_velocity_rad_s"]
                assert len(gyro) == 3 and all(math.isfinite(rate) for rate in gyro)
                assert sensors["yaw_rate_rad_s"] == gyro[2]
            else:
                assert sensors["acceleration_m_s2"] is None
            assert payload["frame_id"] == "map"
            for image in ("camera", "map_image"):
                assert base64.b64decode(payload[image], validate=True).startswith(b"\x89PNG\r\n\x1a\n")
            print(
                f"snapshot={payload['snapshot_id']} map={payload['map_version']} "
                f"imu={imu} RGB+map=PNG lidar={sensors['lidar_min_m']}",
                flush=True,
            )
            result = {
                "protocol": payload["protocol"],
                "snapshot_id": payload["snapshot_id"],
                "map_version": payload["map_version"],
                "goal_id": "F",
                "action": "forward",
                "task_complete": False,
                "latency_s": 0.0,
            }
        writer.write((json.dumps(result) + "\n").encode())
        await writer.drain()
    except Exception as error:
        print("fixture error:", repr(error), flush=True)
        writer.write((json.dumps({"error": str(error) or repr(error)}) + "\n").encode())
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def main(imu):
    port = 18867 if imu else 18866
    server = await asyncio.start_server(partial(reply, imu=imu), "127.0.0.1", port, limit=5_000_000)
    print("FIXTURE READY", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imu", action="store_true")
    asyncio.run(main(parser.parse_args().imu))
