#!/usr/bin/env python3
"""Docker HEALTHCHECK: layanan OCR hidup dan Paddle siap inferensi."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8001/systems/ocr/health"
TIMEOUT = 8


def main() -> int:
    try:
        with urllib.request.urlopen(URL, timeout=TIMEOUT) as resp:
            body = resp.read()
    except urllib.error.URLError as exc:
        print(f"healthcheck: request failed: {exc}", file=sys.stderr)
        return 1

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        print(f"healthcheck: invalid json: {exc}", file=sys.stderr)
        return 1

    if data.get("status") != "ok":
        print(f"healthcheck: unexpected status: {data.get('status')!r}", file=sys.stderr)
        return 1
    if not data.get("inference_ready"):
        print("healthcheck: inference_ready=false", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
