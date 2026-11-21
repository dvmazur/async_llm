"""Independent Mac-to-rover motion lease over a per-run SSH Unix socket."""

import argparse
import json
from pathlib import Path
import secrets
import socket
import socketserver
import threading
import time


class ControlLink:
    def __init__(self, path, token, timeout=2.0):
        self.path = str(path)
        self.token = token
        self.timeout = timeout
        self.last_ok = None
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    @property
    def age(self):
        return time.monotonic() - self.last_ok if self.last_ok is not None else None

    @property
    def healthy(self):
        age = self.age
        return age is not None and age < self.timeout

    def monitor(self):
        while not self.closed.is_set():
            try:
                nonce = secrets.token_hex(12)
                with socket.socket(socket.AF_UNIX) as client:
                    client.settimeout(0.5)
                    client.connect(self.path)
                    client.sendall((json.dumps({'nonce': nonce, 'token': self.token}) + '\n').encode())
                    with client.makefile('rb') as reader:
                        reply = json.loads(reader.readline(4096))
                if isinstance(reply, dict) and reply.get('nonce') == nonce and reply.get('token') == self.token:
                    self.last_ok = time.monotonic()
            except (OSError, ValueError):
                pass
            self.closed.wait(0.25)

    def close(self):
        self.closed.set()
        self.thread.join(timeout=0.8)


def serve(ready_file, token):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(1)
            try:
                request = json.loads(self.rfile.readline(4096))
                if not isinstance(request, dict) or request.get('token') != token:
                    return
                nonce = request.get('nonce')
                if not isinstance(nonce, str) or not 1 <= len(nonce) <= 128:
                    return
                self.wfile.write((json.dumps({'nonce': nonce, 'token': token}) + '\n').encode())
            except (OSError, ValueError):
                return

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True

    with Server(('127.0.0.1', 0), Handler) as server:
        ready_file.write_text(str(server.server_address[1]))
        server.serve_forever(poll_interval=0.2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready-file', type=Path, required=True)
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    serve(args.ready_file, args.token)
