"""Worker stand-in: immediately write a valid successful report."""

import json
import sys
from pathlib import Path

from tests.support.report_build import ok_report_from_request

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
report = ok_report_from_request(request)
Path(request["out"]).write_text(json.dumps(report.to_json()), encoding="utf-8")
