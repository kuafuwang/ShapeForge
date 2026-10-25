# Non-Python gRPC conformance client

This Go module speaks the published `tcenv.remote.v1` API without generated Go
stubs or Python.  It loads `proto/tcenv/remote/v1/evaluator.desc` into
`protodesc`/`dynamicpb`, computes the canonical `sha256-rfc8785-v1` request
digest, and verifies the attempt receipt, blob descriptors and
`hmac-sha256-v1` result signature itself.

It is test/support tooling only; it is not a product client.  Dependencies are
not vendored.  With network access:

```bash
cd tests/support/remotegrpc
go mod download
go run . --desc ../../../proto/tcenv/remote/v1/evaluator.desc --help
```

The acceptance test `tests/shared/test_remote_go_conformance.py` sets
`GOMODCACHE`/`GOCACHE`/`GOPATH` under the test temp directory when the default
Go caches are not writable, and skips when Go is unavailable.  Required flags
are shown by `go run . --help`; the flow is
GetCapabilities → Submit → Watch/GetJob → Retry (interrupted or completed
rejection) → Cancel.  It prints a JSON report with `client: "go"`.
