import json, sys
req = json.load(open(sys.argv[1]))
from tcenv.gpulock import gpu_lock
from tests.support.report_build import ok_report_from_request
with gpu_lock("fake worker benchmark", max_wait_s=20):  # exclusive, like the real benchmark stage
    pass
# The report is a whole schema-4 document (status ``ok`` requires its build record); the same helper
# the other fake workers use, so this stub never becomes a looser path than production.
r = ok_report_from_request(req)
open(req["out"], "w").write(json.dumps(r.to_json()))
