import json, os, sys, time
req = json.load(open(sys.argv[1]))
fd = os.open(req["heartbeat"], os.O_WRONLY | os.O_CREAT, 0o600)
os.pwrite(fd, f"B{time.time():020.6f}\n".encode(), 0)  # "launch started" ... and never finishes
time.sleep(120)
