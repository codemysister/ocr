"""Ekstrak status kepesertaan JKN (AKTIF / TIDAK AKTIF) — tidak menggagalkan validasi."""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from systems.validation.name_extraction import (
    _normalize,
    extract_jkn_participants,
    jkn_masked_name_matches,
    parse_jkn_status_label,
)

_STATUS_DISPLAY = {
    "aktif": "AKTIF",
    "tidak_aktif": "TIDAK AKTIF",
}


def _score_name(left: str, right: str) -> float:
    a = _normalize(left)
    b = _normalize(right)
    if not a or not b:
        return 0.0
    return float(max(fuzz.token_sort_ratio(a, b), fuzz.WRatio(a, b)))


def _pick_member(
    members: list[dict[str, str | None]],
    *,
    expected_name: str,
    candidate_name: str,
) -> dict[str, str | None] | None:
    if not members:
        return None
    keys = [n for n in (candidate_name, expected_name) if (n or "").strip()]
    if not keys:
        if len(members) == 1:
            return members[0]
        return None
    best: tuple[float, dict[str, str | None]] | None = None
    for member in members:
        name = str(member.get("name") or "")
        score = max(_score_name(name, k) for k in keys)
        if any(jkn_masked_name_matches(name, k) for k in keys):
            score = max(score, 92.0)
        if best is None or score > best[0]:
            best = (score, member)
    if best and best[0] >= 65.0:
        return best[1]
    return None


def evaluate_jkn_status(
    ocr_text: str,
    *,
    profile_id: str,
    expected_name: str = "",
    candidate_name: str = "",
) -> dict[str, Any] | None:
    """None jika bukan JKN. Status hanya dibaca, tidak memblokir lolos."""
    if (profile_id or "").strip().casefold() != "jkn":
        return None

    members = extract_jkn_participants(ocr_text)
    name_keys = (expected_name or "").strip() or (candidate_name or "").strip()
    picked = _pick_member(
        members,
        expected_name=expected_name,
        candidate_name=candidate_name,
    )
    status = picked.get("status") if picked else None
    if status is None and name_keys:
        blob = ocr_text or ""
        needle = (candidate_name or expected_name).strip()
        low = blob.casefold()
        idx = low.find(needle.casefold()) if needle else -1
        if idx >= 0:
            window = blob[idx : idx + max(180, len(needle) + 80)]
            status = parse_jkn_status_label(window)
    if status is None and not name_keys:
        if any(m.get("status") == "aktif" for m in members):
            status = "aktif"
            picked = next(m for m in members if m.get("status") == "aktif")
        elif any(m.get("status") == "tidak_aktif" for m in members):
            status = "tidak_aktif"
            picked = next(m for m in members if m.get("status") == "tidak_aktif")

    display = _STATUS_DISPLAY.get(status or "")
    who = (picked or {}).get("name") or expected_name or "peserta"
    if status == "aktif":
        reason = f"Status JKN terbaca: {who} AKTIF."
    elif status == "tidak_aktif":
        reason = f"Status JKN terbaca: {who} TIDAK AKTIF."
    else:
        reason = f"Status JKN tidak terbaca pada kartu {who}."

    return {
        "evaluated": True,
        "status": status,
        "status_display": display,
        "matched_name": (picked or {}).get("name"),
        "members": [
            {"name": m.get("name"), "status": m.get("status")} for m in members
        ],
        "reason": reason,
    }


def apply_jkn_status_extract(
    validation: dict[str, Any],
    *,
    ocr_text: str,
    profile_id: str | None = None,
    expected_name: str = "",
) -> dict[str, Any]:
    out = dict(validation)
    profile = (profile_id or out.get("document_profile_id") or "").strip().casefold()
    ident = out.get("identity") if isinstance(out.get("identity"), dict) else {}
    extraction = out.get("name_extraction") if isinstance(out.get("name_extraction"), dict) else {}
    name_ref = expected_name or str(ident.get("expected_normalized") or "") or ""
    candidate = str(extraction.get("candidate_raw") or "")
    info = evaluate_jkn_status(
        ocr_text,
        profile_id=profile,
        expected_name=name_ref,
        candidate_name=candidate,
    )
    if info is None:
        out.setdefault("jkn_status_label", None)
        return out

    out["jkn_status"] = {
        k: info[k]
        for k in ("status", "status_display", "matched_name", "members", "reason")
    }
    out["jkn_status_label"] = info["status"]
    out["jkn_status_display"] = info["status_display"]

    explanation = dict(out.get("explanation") or {})
    details = list(explanation.get("detail_lines") or [])
    if info["reason"] not in details:
        explanation["detail_lines"] = [info["reason"], *details]
    gates = dict(explanation.get("gates") or {})
    gates["jkn_status"] = {
        "status": info["status"],
        "status_display": info["status_display"],
        "matched_name": info["matched_name"],
        "blocking": False,
    }
    explanation["gates"] = gates
    out["explanation"] = explanation
    return out


# Alias lama — tidak memblokir validasi.
apply_jkn_aktif_gate = apply_jkn_status_extract
evaluate_jkn_aktif_gate = evaluate_jkn_status
