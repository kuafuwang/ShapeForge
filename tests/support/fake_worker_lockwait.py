import json, os, sys, time
req = json.load(open(sys.argv[1]))
fd = os.open(req["heartbeat"], os.O_WRONLY | os.O_CREAT, 0o600)
os.pwrite(fd, f"W{time.time():020.6f}\n".encode(), 0)  # waiting for the GPU lock
time.sleep(3)
os.pwrite(fd, f"E{time.time():020.6f}\n".encode(), 0)  # got it; then never finishes
time.sleep(120)
