"""OCR via server remote (mis. https://checkinpro-ocr.web.id) — validasi tetap lokal."""

from __future__ import annotations

import json
import logging
import os
import ssl
import threading
import time
from http.client import HTTPConnection, HTTPSConnection
from typing import Any
from urllib.parse import urlencode, urlparse

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120.0
_DEFAULT_VL_TIMEOUT = 180.0

_conn_lock = threading.Lock()
_https_conn: HTTPConnection | HTTPSConnection | None = None
_https_key: tuple[str, str, int] | None = None


def ocr_remote_base_url() -> str | None:
    raw = (os.environ.get("OCR_REMOTE_BASE_URL") or "").strip().rstrip("/")
    return raw or None


def ocr_remote_enabled() -> bool:
    flag = (os.environ.get("OCR_REMOTE_ENABLED") or "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    return bool(ocr_remote_base_url())


def default_ocr_mode() -> str:
    """Engine OCR default: PP-OCRv6 (`fast`). Override: `OCR_DEFAULT_MODE=fast|mistral|vl`."""
    raw = (os.environ.get("OCR_DEFAULT_MODE") or "fast").strip().lower()
    if raw in {"fast", "mistral", "vl"}:
        return raw
    return "fast"


def _timeout_s(*, for_vl: bool = False) -> float:
    raw = (os.environ.get("OCR_REMOTE_TIMEOUT_S") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return _DEFAULT_VL_TIMEOUT if for_vl else _DEFAULT_TIMEOUT


def _browser_ua() -> str:
    return (os.environ.get("OCR_REMOTE_USER_AGENT") or "").strip() or (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )


def _sniff_upload(image_bytes: bytes, filename: str) -> tuple[str, str]:
    name = (filename or "").strip() or "upload.jpg"
    if len(image_bytes) >= 8 and image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        if not name.lower().endswith(".png"):
            name = "upload.png"
        return name, "image/png"
    if len(image_bytes) >= 2 and image_bytes[:2] == b"\xff\xd8":
        if not name.lower().endswith((".jpg", ".jpeg")):
            name = "upload.jpg"
        return name, "image/jpeg"
    if (
        len(image_bytes) >= 12
        and image_bytes[:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):
        if not name.lower().endswith(".webp"):
            name = "upload.webp"
        return name, "image/webp"
    return name, "application/octet-stream"


def _close_https_conn() -> None:
    global _https_conn, _https_key
    if _https_conn is not None:
        try:
            _https_conn.close()
        except Exception:
            pass
    _https_conn = None
    _https_key = None


def _https_post(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
    """POST keep-alive (satu TLS handshake, dipakai ulang antar file benchmark)."""
    global _https_conn, _https_key
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    host = parsed.hostname or ""
    port = parsed.port or (443 if scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    key = (scheme, host, port)

    def _new_conn() -> HTTPConnection | HTTPSConnection:
        if scheme == "https":
            return HTTPSConnection(
                host,
                port,
                timeout=timeout,
                context=ssl.create_default_context(),
            )
        return HTTPConnection(host, port, timeout=timeout)

    def _request(conn: HTTPConnection | HTTPSConnection) -> tuple[int, bytes]:
        conn.timeout = timeout
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, data

    with _conn_lock:
        if _https_conn is None or _https_key != key:
            _close_https_conn()
            _https_conn = _new_conn()
            _https_key = key
        try:
            status, data = _request(_https_conn)
        except Exception:
            _close_https_conn()
            _https_conn = _new_conn()
            _https_key = key
            status, data = _request(_https_conn)
        if status >= 500:
            logger.warning("ocr_remote HTTP %s, retry sekali %s", status, url)
            _close_https_conn()
            time.sleep(0.45)
            _https_conn = _new_conn()
            _https_key = key
            status, data = _request(_https_conn)
        if status >= 400:
            err = data.decode("utf-8", errors="replace")[:800]
            raise RuntimeError(f"OCR remote {url} gagal HTTP {status}: {err}")
        return data


def _post_image(
    path: str,
    image_bytes: bytes,
    *,
    query: dict[str, str] | None = None,
    filename: str = "upload.jpg",
    content_type: str | None = None,
    for_vl: bool = False,
) -> dict[str, Any]:
    base = ocr_remote_base_url()
    if not base:
        raise RuntimeError("OCR_REMOTE_BASE_URL tidak diset.")

    fname, mime = _sniff_upload(image_bytes, filename)
    if content_type:
        mime = content_type

    url = base + path
    if query:
        qs = urlencode({k: v for k, v in query.items() if v is not None and v != ""})
        if qs:
            url = f"{url}?{qs}"

    boundary = "----OCRRemoteBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8") + image_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    timeout = _timeout_s(for_vl=for_vl)
    ua = _browser_ua()
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Accept": "application/json",
        "User-Agent": ua,
        "Origin": base,
        "Referer": base + "/pipeline",
        "Connection": "keep-alive",
        "Content-Length": str(len(body)),
    }
    logger.info("ocr_remote POST %s bytes=%d timeout=%.0fs", url, len(image_bytes), timeout)
    raw = _https_post(url, body, headers, timeout)

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError("OCR remote mengembalikan JSON tidak valid.") from e
    if not isinstance(parsed, dict):
        raise RuntimeError("OCR remote mengembalikan payload yang tidak dikenali.")
    parsed.setdefault("ocr_remote", True)
    parsed.setdefault("ocr_remote_url", url)
    return parsed


def run_remote_ocr_fast(
    image_bytes: bytes,
    *,
    pp_ocr_tier: str | None = None,
    filename: str = "upload.jpg",
) -> dict[str, Any]:
    tier = (pp_ocr_tier or "medium").strip() or "medium"
    return _post_image(
        "/systems/ocr/api/v1/ocr-fast",
        image_bytes,
        query={"pp_ocr_tier": tier},
        filename=filename,
    )


def run_remote_ocr_vl(
    image_bytes: bytes,
    *,
    include_full_json: bool = False,
    filename: str = "upload.jpg",
) -> dict[str, Any]:
    return _post_image(
        "/systems/ocr/api/v1/ocr",
        image_bytes,
        query={"full_json": "true" if include_full_json else "false"},
        filename=filename,
        for_vl=True,
    )


def run_remote_ocr_vl_ktp(
    image_bytes: bytes,
    *,
    filename: str = "upload.jpg",
) -> dict[str, Any]:
    return _post_image(
        "/systems/ocr/api/v1/ocr-vl-ktp",
        image_bytes,
        filename=filename,
        for_vl=True,
    )


def run_remote_ocr_mistral(
    image_bytes: bytes,
    *,
    filename: str = "upload.jpg",
) -> dict[str, Any]:
    return _post_image(
        "/systems/ocr/api/v1/ocr-mistral",
        image_bytes,
        filename=filename,
    )
