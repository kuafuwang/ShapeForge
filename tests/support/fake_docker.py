"""Deterministic fake ``docker`` CLI for container-supervisor tests.

This is a test double, never a production backend.  It implements just enough
of the Docker CLI surface used by
:class:`tcenv.remote.container_supervisor.DockerAttemptSupervisor` and stores
its state under the directory that contains the generated executable.

``docker start --attach --interactive`` really executes the recorded command
with inherited stdio, so the private worker protocol is exercised end to end.
``docker kill`` writes a kill marker; the attach process then SIGKILLs the
worker's process group, emulating the container/cgroup boundary.  Control
switches in ``control.json`` inject lifecycle faults (image mismatch, create or
rm failure, a survivor after kill, a stuck attach).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _dump(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def _locked(state: Path) -> Iterator[None]:
    state.mkdir(parents=True, exist_ok=True)
    with (state / "lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _parse_memory(value: str) -> int:
    units = {"b": 1, "k": 1000, "m": 1000**2, "g": 1000**3}
    unit = value[-1].lower()
    if unit in units:
        return int(float(value[:-1]) * units[unit])
    return int(float(value))


def _parse_flags(argv: list[str], with_value: set[str], boolean: set[str]) -> tuple[dict[str, Any], list[str]]:
    parsed: dict[str, Any] = {}
    positionals: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in with_value:
            parsed[token] = argv[index + 1]
            index += 2
            continue
        if token in boolean:
            parsed[token] = True
            index += 1
            continue
        positionals.append(token)
        index += 1
    return parsed, positionals


class FakeDocker:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.state = base / "state"
        self.containers = self.state / "containers"
        self.control_path = base / "control.json"

    # -- test-side helpers -------------------------------------------------
    def control(self) -> dict[str, Any]:
        return _load(self.control_path) or {}

    def set_control(self, **values: Any) -> None:
        current = self.control()
        current.update(values)
        _dump(self.control_path, current)

    def calls(self) -> list[list[str]]:
        path = self.state / "calls.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line)["argv"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def records(self) -> list[dict[str, Any]]:
        return [_load(path) or {} for path in sorted(self.containers.glob("*.json"))]

    def record(self, container_id: str) -> dict[str, Any] | None:
        return _load(self.containers / f"{container_id}.json")

    def log_argv(self, argv: list[str]) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        with (self.state / "calls.jsonl").open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps({"argv": argv, "at": time.time()}) + "\n")

    # -- CLI ---------------------------------------------------------------
    def main(self, raw: list[str]) -> int:
        argv = list(raw)
        if argv[:1] == ["--host"]:
            argv = argv[2:]
        if not argv:
            print("usage: docker COMMAND", file=sys.stderr)
            return 64
        self.log_argv(["docker", *raw])
        command, rest = argv[0], argv[1:]
        handler = getattr(self, f"_cmd_{command}", None)
        if handler is None:
            print(f"fake docker does not implement {command}", file=sys.stderr)
            return 64
        return int(handler(rest))

    def _cmd_info(self, argv: list[str]) -> int:
        flags, positionals = _parse_flags(argv, {"--format"}, set())
        del positionals
        control = self.control()
        if control.get("slow_info_s"):
            # A daemon query that is slow enough to spend a candidate deadline.
            time.sleep(float(control["slow_info_s"]))
        if control.get("fail_info"):
            print("Error response from daemon: injected info failure", file=sys.stderr)
            return 1
        if flags.get("--format") == "{{.ID}}":
            print(self.control().get("daemon_id", "11111111-2222-3333-4444-555555555555"))
            return 0
        if flags.get("--format") and "flood" in str(flags.get("--format")):
            sys.stdout.write("x" * int(self.control().get("flood_bytes", 2_000_000)))
            return 0
        print("{}")
        return 0

    def _cmd_image(self, argv: list[str]) -> int:
        if argv[:1] != ["inspect"]:
            print("unsupported image command", file=sys.stderr)
            return 64
        control = self.control()
        if control.get("fail_image"):
            print("Error response from daemon: image not found", file=sys.stderr)
            return 1
        print(control.get("image_id", f"sha256:{'a' * 64}"))
        return 0

    def _cmd_create(self, argv: list[str]) -> int:
        flags, positionals = _parse_flags(argv, {
            "--name", "--label", "--network", "--cap-drop", "--security-opt",
            "--pids-limit", "--memory", "--cpus", "--user", "--workdir",
            "--tmpfs", "--env-file", "--runtime",
        }, {"--interactive", "--init", "--read-only"})
        control = self.control()
        if control.get("fail_create"):
            print("Error response from daemon: injected create failure", file=sys.stderr)
            return 125
        if control.get("slow_create_s"):
            # A create that outlives the candidate deadline: the container really
            # exists afterwards, so the engine must clean it up without starting.
            time.sleep(float(control["slow_create_s"]))
        if len(positionals) < 2:
            print("fake docker create requires an image and command", file=sys.stderr)
            return 64
        env_file = Path(flags["--env-file"])
        if not env_file.is_file():
            print("no such env file", file=sys.stderr)
            return 64
        env_lines = [line for line in env_file.read_text(encoding="utf-8").splitlines() if line]
        labels: dict[str, str] = {}
        for token in self._multi(argv, "--label"):
            key, _, value = token.partition("=")
            labels[key] = value
        image_ref = positionals[0]
        command = positionals[1:]
        tmpfs = self._multi(argv, "--tmpfs")
        container_id = uuid.uuid4().hex + uuid.uuid4().hex[:32]
        name = flags["--name"]
        cpus = flags.get("--cpus")
        record = {
            "Id": container_id,
            "Name": f"/{name}",
            "Image": control.get("image_id", f"sha256:{'a' * 64}"),
            "Config": {
                "Image": image_ref,
                "Labels": labels,
                "User": flags.get("--user", ""),
                "Env": env_lines,
                "Cmd": command,
            },
            "HostConfig": {
                "NetworkMode": flags.get("--network", "default"),
                "ReadonlyRootfs": bool(flags.get("--read-only")),
                "CapDrop": self._multi(argv, "--cap-drop"),
                "SecurityOpt": self._multi(argv, "--security-opt"),
                "Privileged": bool(control.get("privileged")),
                "Binds": list(control.get("binds") or []),
                "PidMode": flags.get("--pid", ""),
                "IpcMode": "private",
                "UsernsMode": "",
                "PidsLimit": int(flags.get("--pids-limit", -1)),
                "Memory": _parse_memory(flags.get("--memory", "0")),
                "NanoCpus": int(float(cpus) * 1e9) if cpus else 0,
                "Runtime": flags.get("--runtime", "runc"),
            },
            "Mounts": [
                {"Type": "tmpfs", "Destination": spec.split(":")[0]}
                for spec in tmpfs
            ],
            "State": {"Status": "created", "ExitCode": 0, "Pid": 0},
            "fake": {"command": command, "env": env_lines, "flags": flags, "tmpfs": tmpfs},
        }
        with _locked(self.state):
            attempt_label = labels.get("tcenv.attempt")
            for item in self.containers.glob("*.json"):
                existing = (_load(item) or {}).get("Config", {}).get("Labels", {})
                if attempt_label and existing.get("tcenv.attempt") == attempt_label:
                    print(
                        f"Error response from daemon: Conflict. The container name {name} is already in use",
                        file=sys.stderr,
                    )
                    return 125
            _dump(self.containers / f"{container_id}.json", record)
        if control.get("fail_create_after_record"):
            # The daemon really created the container while the CLI still failed
            # (a timeout or a broken client): the caller never learns the id.
            print("Error response from daemon: create completed but the client failed", file=sys.stderr)
            return 125
        print(container_id)
        return 0

    @staticmethod
    def _multi(argv: list[str], flag: str) -> list[str]:
        values = []
        for index, token in enumerate(argv):
            if token == flag and index + 1 < len(argv):
                values.append(argv[index + 1])
        return values

    @staticmethod
    def _load_record(path: Path) -> dict[str, Any] | None:
        return _load(path)

    def _find(self, ref: str) -> tuple[dict[str, Any] | None, Path | None]:
        for path in self.containers.glob("*.json"):
            record = _load(path) or {}
            if record.get("Id") == ref or record.get("Name") == f"/{ref}" or str(record.get("Id", "")).startswith(ref):
                return record, path
        return None, None

    def _cmd_inspect(self, argv: list[str]) -> int:
        _, positionals = _parse_flags(argv, {"--type", "--format"}, set())
        if not positionals:
            print("fake docker inspect requires a reference", file=sys.stderr)
            return 64
        control = self.control()
        if control.get("slow_inspect_s"):
            # An identity/absence probe that outlives the candidate deadline: the
            # container may not be started (or written to) after it returns.
            time.sleep(float(control["slow_inspect_s"]))
        documents = []
        for ref in positionals:
            record, _path = self._find(ref)
            if record is None:
                print(f"Error: No such object: {ref}", file=sys.stderr)
                return 1
            documents.append(record)
        print(json.dumps(documents, sort_keys=True))
        return 0

    def _cmd_start(self, argv: list[str]) -> int:
        flags, positionals = _parse_flags(argv, {"--detach-keys"}, {"--attach", "--interactive"})
        del flags
        if not positionals:
            print("fake docker start requires a container", file=sys.stderr)
            return 64
        record, path = self._find(positionals[0])
        if record is None or path is None:
            print(f"Error: No such container: {positionals[0]}", file=sys.stderr)
            return 1
        control = self.control()
        if control.get("fail_start"):
            with _locked(self.state):
                record["State"] = {"Status": "exited", "ExitCode": 125, "Pid": 0}
                _dump(path, record)
            print("Error response from daemon: injected start failure", file=sys.stderr)
            return 125
        command = record["fake"]["command"]
        # Docker gives the container only the image/config environment plus the
        # explicit env file; it never forwards the client's environment.  The
        # fake must do the same or unit tests would not prove isolation.
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin"}
        for line in record["fake"]["env"]:
            key, _, value = line.partition("=")
            env[key] = value
        child = subprocess.Popen(
            command, stdin=sys.stdin.buffer, stdout=sys.stdout.buffer, stderr=sys.stderr.buffer,
            env=env, start_new_session=True,
        )
        with _locked(self.state):
            record["State"] = {"Status": "running", "ExitCode": 0, "Pid": child.pid}
            _dump(path, record)
        kill_marker = self.state / "kill" / record["Id"]
        exit_code = None
        while exit_code is None:
            if kill_marker.exists():
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            exit_code = child.poll()
            if exit_code is None:
                time.sleep(0.01)
        with _locked(self.state):
            if path.exists() and not control.get("detach_leaves_running"):
                record["State"] = {"Status": "exited", "ExitCode": exit_code, "Pid": 0}
                _dump(path, record)
        if control.get("hang_attach"):
            while True:
                time.sleep(1)
        return exit_code

    def _cmd_kill(self, argv: list[str]) -> int:
        flags, positionals = _parse_flags(argv, {"--signal"}, set())
        del flags
        if not positionals:
            return 64
        record, path = self._find(positionals[0])
        if record is None or path is None:
            print(f"Error: No such container: {positionals[0]}", file=sys.stderr)
            return 1
        if record["State"]["Status"] != "running":
            print(f"Error response from daemon: Container {positionals[0]} is not running", file=sys.stderr)
            return 1
        if self.control().get("ignore_kill"):
            return 0
        kill_dir = self.state / "kill"
        kill_dir.mkdir(parents=True, exist_ok=True)
        (kill_dir / record["Id"]).write_text("kill", encoding="utf-8")
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            current = _load(path) or {}
            if current.get("State", {}).get("Status") != "running":
                return 0
            time.sleep(0.01)
        # The kernel/container died even if no attach client is watching.
        with _locked(self.state):
            current = _load(path)
            if current is not None and current.get("State", {}).get("Status") == "running":
                current["State"] = {"Status": "exited", "ExitCode": 137, "Pid": 0}
                _dump(path, current)
        return 0

    def _cmd_rm(self, argv: list[str]) -> int:
        flags, positionals = _parse_flags(argv, set(), {"--force", "-f"})
        del flags
        if not positionals:
            return 64
        record, path = self._find(positionals[0])
        if record is None or path is None:
            print(f"Error: No such container: {positionals[0]}", file=sys.stderr)
            return 1
        if self.control().get("fail_rm"):
            print("Error response from daemon: injected rm failure", file=sys.stderr)
            return 1
        if self.control().get("rm_leaves_running"):
            return 0
        kill_dir = self.state / "kill"
        kill_dir.mkdir(parents=True, exist_ok=True)
        (kill_dir / record["Id"]).write_text("kill", encoding="utf-8")
        with _locked(self.state):
            try:
                path.unlink()
            except OSError:
                pass
        return 0

    def _cmd_ps(self, argv: list[str]) -> int:
        _, positionals = _parse_flags(argv, set(), {"--all", "-a", "--quiet", "-q"})
        del positionals
        labels = self._multi(argv, "--filter")
        matches = []
        for path in self.containers.glob("*.json"):
            record = _load(path) or {}
            record_labels = (record.get("Config") or {}).get("Labels") or {}
            for label in labels:
                key, _, value = label.partition("=")
                if key != "label":
                    continue
                wanted_key, _, wanted_value = value.partition("=")
                if record_labels.get(wanted_key) != wanted_value:
                    break
            else:
                matches.append(record["Id"])
        for container_id in matches:
            print(container_id)
        return 0


def main(base: str | None = None, argv: list[str] | None = None) -> int:
    root = Path(base or os.path.dirname(os.path.abspath(__file__)))
    return FakeDocker(root).main(list(sys.argv[1:] if argv is None else argv))


_LAUNCHER = '''#!{python}
import os
import sys

sys.path.insert(0, {root!r})
from tests.support.fake_docker import main

raise SystemExit(main(base=os.path.dirname(os.path.abspath(__file__))))
'''


def install_fake_docker(directory: Path, *, image_digest: str = "a" * 64, control: Mapping[str, Any] | None = None) -> tuple[Path, FakeDocker]:
    """Create an executable fake ``docker`` and return ``(path, controller)``."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = str(Path(__file__).resolve().parents[2])
    executable = directory / "docker"
    executable.write_text(_LAUNCHER.format(python=sys.executable, root=root), encoding="utf-8")
    executable.chmod(0o755)
    fake = FakeDocker(directory)
    fake.state.mkdir(parents=True, exist_ok=True)
    (fake.state / "containers").mkdir(parents=True, exist_ok=True)
    # Each installed fake is a distinct "Docker Engine": two fake daemons must
    # not share an identity, or a recovery test could reconcile across them.
    default_daemon = "fake-" + hashlib.sha256(str(directory.resolve()).encode("utf-8")).hexdigest()[:32]
    initial = {"image_id": f"sha256:{image_digest}", "daemon_id": default_daemon}
    initial.update(control or {})
    _dump(fake.control_path, initial)
    return executable, fake
