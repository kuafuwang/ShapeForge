"""Fake v2 worker: publish a compile heartbeat and never finish (never touches the GPU).

It proves the first real compile is inside the externally supervised worker: the parent's
``supervise`` must kill this process group from the compile deadline, not wait for a parent-side
compile that would be outside the watchdog.
"""
import json
import os
import sys
import time

request = json.load(open(sys.argv[1]))
fd = os.open(request["heartbeat"], os.O_WRONLY | os.O_CREAT, 0o600)
os.pwrite(fd, f"C{time.time():020.6f}\n".encode(), 0)
time.sleep(120)
