#!/usr/bin/env python3
"""Negative fixture: connect to an outer-owned loopback listener.

The listener is not the offline proxy.  Runs only after the hmz Profile is
applied.  Exit 13 means the connection was denied (required); exit 14 means the
Profile allowed non-proxy loopback traffic (a real failure).
"""

from __future__ import annotations

import argparse
import socket


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--outside-port", type=int, required=True)
    parser.add_argument("--artifacts", type=str)
    args = parser.parse_args()
    try:
        connection = socket.create_connection(("127.0.0.1", args.outside_port), timeout=2.0)
    except PermissionError:
        return 13
    except OSError:
        return 13
    connection.close()
    return 14


if __name__ == "__main__":
    raise SystemExit(main())
