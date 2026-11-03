"""Fake v2 worker: publish a launch heartbeat and never finish (never touches the GPU)."""
import json
import os
import sys
import time

request = json.load(open(sys.argv[1]))
fd = os.open(request["heartbeat"], os.O_WRONLY | os.O_CREAT, 0o600)
os.pwrite(fd, f"B{time.time():020.6f}\n".encode(), 0)
time.sleep(120)
