"""Control-link expiry and physical-only completion rules."""

import json
from pathlib import Path
import socketserver
import tempfile
import threading
import time
import unittest

from robot_sim.candidate_nav.direct_control import validate_response
from robot_sim.candidate_nav.live_link import ControlLink


class LinkTests(unittest.TestCase):
    def test_expiry_requires_a_fresh_reply(self):
        allowed = threading.Event()
        allowed.set()
        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                packet = json.loads(self.rfile.readline())
                if allowed.is_set():
                    self.wfile.write((json.dumps(packet)+'\n').encode())
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path = str(Path(folder)/'lease.sock')
            with socketserver.UnixStreamServer(path, Handler) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                link = ControlLink(path, 'run-123', timeout=.3)
                try:
                    deadline = time.monotonic()+2
                    while not link.healthy and time.monotonic()<deadline:
                        time.sleep(.01)
                    self.assertTrue(link.healthy)
                    allowed.clear()
                    deadline = time.monotonic()+1
                    while link.healthy and time.monotonic()<deadline:
                        time.sleep(.01)
                    self.assertFalse(link.healthy)
                finally:
                    link.close()
                    server.shutdown()
                    thread.join()

    def test_replayed_nonce_is_not_accepted(self):
        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                packet = json.loads(self.rfile.readline())
                packet['nonce'] = 'stale'
                self.wfile.write((json.dumps(packet)+'\n').encode())
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path = str(Path(folder)/'lease.sock')
            with socketserver.UnixStreamServer(path, Handler) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                link = ControlLink(path,'run-123')
                try:
                    time.sleep(.4)
                    self.assertFalse(link.healthy)
                    self.assertIsNone(link.last_ok)
                finally:
                    link.close()
                    server.shutdown()
                    thread.join()

    def test_task_completion_requires_explicit_physical_mode(self):
        reply={'protocol':'direct_navigation_v1','action':'stop','goal_id':'Z',
               'task_complete':True,'completion_mode':'task_completion',
               'completion_probabilities':{'yes':.95}}
        observation={'physical':True,'completion_mode':'task_completion','objects':[]}
        validate_response(reply,observation)
        observation['physical']=False
        with self.assertRaises(ValueError):validate_response(reply,observation)


if __name__=='__main__':unittest.main()
