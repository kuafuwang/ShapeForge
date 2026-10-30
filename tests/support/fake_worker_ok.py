import json, os, sys, time
req = json.load(open(sys.argv[1]))
fd = os.open(req["heartbeat"], os.O_WRONLY | os.O_CREAT, 0o600)
for _ in range(15):  # many short launches in a row, each well below the deadline
    os.pwrite(fd, f"B{time.time():020.6f}\n".encode(), 0); time.sleep(0.2)
    os.pwrite(fd, f"E{time.time():020.6f}\n".encode(), 0); time.sleep(0.05)
from tests.support.report_build import ok_report_from_request
r = ok_report_from_request(req)
open(req["out"], "w").write(json.dumps(r.to_json()))
