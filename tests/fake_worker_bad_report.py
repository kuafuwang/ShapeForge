"""Worker stand-in: the process exits after writing a truncated report."""

import json
import sys

req = json.load(open(sys.argv[1]))
open(req["out"], "w").write("{not json")
