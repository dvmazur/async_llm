"""Read-only live rover camera and map, served on loopback through SSH."""

import argparse
import io
import json
import math
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from nav_msgs.msg import OccupancyGrid
from PIL import ImageDraw
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Float32, String
from tf2_ros import Buffer, TransformListener

from .agent import grid_message, msg_time, yaw
from .candidates import Candidate
from .live import Camera
from .render import pixel, render_map
from .map_memory import accumulated_view


class ViewCache:
    def __init__(self, rover_id=80):
        self.lock = threading.Lock()
        self.state = {'rover_id': rover_id, 'ready': False}
        self.images = {}

    def set(self, state, images):
        with self.lock:
            self.state = state
            self.images = images

    def get(self, resource):
        with self.lock:
            if resource == '/api/state':
                state = dict(self.state)
                state['server_time'] = time.time()
                return 'application/json', json.dumps(state, ensure_ascii=False, allow_nan=False).encode()
            return self.images.get(resource)


def serve(cache, port, host):
    page = Path(__file__).with_name('live_view.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            resource = urlsplit(self.path).path
            result = ('text/html; charset=utf-8', page) if resource == '/' else cache.get(resource)
            if result is None:
                self.send_error(404)
                return
            kind, payload = result
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


class LiveView(Node):
    def __init__(self, cache, rover_id=80):
        super().__init__('rover_live_view')
        self.cache = cache
        self.rover_id = rover_id
        self.camera = Camera()
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.maps = {}
        self.telemetry = None
        self.last_run = None
        self.battery = None
        self.trail = []
        self.sequence = 0
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        for topic in ('/map', '/candidate_nav/planning_map'):
            self.create_subscription(OccupancyGrid, topic,
                                     lambda msg, topic=topic: self.maps.__setitem__(topic, msg), latched)
        self.create_subscription(String, '/rover_live/telemetry', self.receive_telemetry, 1)
        self.create_subscription(Float32, '/hardware/battery', self.receive_battery, 1)
        self.create_timer(0.5, self.update)

    def receive_telemetry(self, message):
        try:
            telemetry = json.loads(message.data)
            if telemetry.get('rover_id', 80) != self.rover_id:
                return
            if telemetry.get('run') != self.last_run:
                self.trail = []
                self.last_run = telemetry.get('run')
            self.telemetry = telemetry
        except ValueError:
            return

    def receive_battery(self, message):
        self.battery = (message.data, time.time())

    def pose(self):
        try:
            transform = self.buffer.lookup_transform('map', 'base_link', Time())
            if not -0.15 < time.time() - msg_time(transform) < 1:
                return None
            t = transform.transform
            return (t.translation.x, t.translation.y, yaw(t.rotation))
        except Exception:
            return None

    def update(self):
        now = time.time()
        telemetry = self.telemetry or {}
        age = now - telemetry.get('stamp', 0)
        mission_live = -0.15 <= age < 3 and not telemetry.get('finished', True)
        planning = self.maps.get('/candidate_nav/planning_map')
        use_planning = planning is not None and -0.15 <= now - msg_time(planning) < 3
        message = planning if use_planning else self.maps.get('/map')
        pose = self.pose()
        images = {}
        if message is not None and pose is not None:
            grid = grid_message(message, telemetry.get('map_version', 0) if use_planning else 0)
            if not self.trail or math.dist(self.trail[-1], pose[:2]) > 0.03:
                self.trail.append(pose[:2])
                self.trail = self.trail[-3000:]
            candidates = [Candidate(**c) for c in telemetry.get('candidates', [])] if mission_live else []
            active = Candidate(**telemetry['active_goal']) if mission_live and telemetry.get('active_goal') else None
            if active is not None:
                candidates = [c for c in candidates if c.id != active.id] + [active]
            image, projection = render_map(grid, pose, candidates, self.trail, active, size=768)
            route = telemetry.get('route', []) if mission_live else []
            if len(route) > 1:
                draw = ImageDraw.Draw(image)
                draw.line([pixel(point, projection) for point in route], fill=(240, 132, 59), width=3)
            stream = io.BytesIO()
            image.save(stream, format='PNG')
            images['/map.png'] = ('image/png', stream.getvalue())
        accumulated_meta = {'available': False}
        slam_message = self.maps.get('/map')
        if slam_message is not None and message is not None and pose is not None:
            slam = grid_message(slam_message, 0)
            accumulated, accumulated_meta = accumulated_view(slam, pose, self.trail, grid, now, size=768)
            if accumulated is not None:
                stream = io.BytesIO()
                accumulated.save(stream, format='PNG')
                images['/accumulated_map.png'] = ('image/png', stream.getvalue())
        frame = self.camera.latest
        camera_age = now - frame[1] if frame is not None else None
        if frame is not None:
            stream = io.BytesIO()
            frame[0].save(stream, format='JPEG', quality=82)
            images['/camera.jpg'] = ('image/jpeg', stream.getvalue())
        self.sequence += 1
        state = {
            'rover_id': self.rover_id, 'ready': message is not None and pose is not None,
            'stamp': now, 'sequence': self.sequence,
            'mission_live': mission_live, 'mission': telemetry,
            'telemetry_age_s': round(age, 2) if self.telemetry is not None else None,
            'map_source': 'Карта планирования' if use_planning else 'Штатная карта SLAM',
            'map_age_s': round(now - msg_time(message), 2) if message is not None else None,
            'map_available': '/map.png' in images, 'pose': pose,
            'accumulated_map_available': '/accumulated_map.png' in images,
            'accumulated_map': accumulated_meta,
            'camera_available': frame is not None,
            'camera_age_s': round(camera_age, 2) if camera_age is not None else None,
            'battery_v': self.battery[0] if self.battery is not None and now-self.battery[1] < 5 else None,
        }
        self.cache.set(state, images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18870)
    parser.add_argument('--host', default=socket.gethostbyname(socket.gethostname()))
    parser.add_argument('--rover-id', type=int, choices=[80, 107], default=80)
    args = parser.parse_args()
    cache = ViewCache(args.rover_id)
    server = serve(cache, args.port, args.host)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    rclpy.init()
    node = LiveView(cache, args.rover_id)
    print(f'LIVE VIEW READY {args.host}:{args.port}', flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.camera.closed.set()
        server.shutdown()
        server.server_close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
