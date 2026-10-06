import json, sys
req = json.load(open(sys.argv[1]))
from tcenv.gpulock import gpu_lock
from tcenv.evaluate import EvalReport
with gpu_lock("fake worker benchmark", max_wait_s=20):  # exclusive, like the real benchmark stage
    pass
r = EvalReport(task=req["task"], dtype=req["dtype"], size=req["size"], mode=req["mode"], seed=req["seed"], status="ok", stage="done")
open(req["out"], "w").write(json.dumps(r.to_json()))
