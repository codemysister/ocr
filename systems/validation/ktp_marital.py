"""Gate KTP: posisi Helper wajib status perkawinan BELUM KAWIN."""

from __future__ import annotations

import re
from typing import Any

_HELPER_EXACT = frozenset(
    {
        "helper",
        "posisi helper",
        "job helper",
        "helper warehouse",
        "helper gudang",
    }
)

_STATUS_LABELS: dict[str, str] = {
    "belum_kawin": "BELUM KAWIN",
    "kawin": "KAWIN",
    "cerai": "CERAI",
}

# OCR PP-OCR sering merusak "BELUM KAWIN" / label "Status Perkawinan".
_BELUM_KAWIN_COMPACT = re.compile(
    r"bel+u?n?m+k+a+[vwun]+i?n?"
    r"|bl+m+k+a+w?i?n?"
    r"|bel+u?n?m+ka(?:win|wn|vin|wim)?"
    r"|tdk+kawin"
    r"|tidakkawin"
)
_BELUM_KAWIN_WORDS = re.compile(
    r"\b(?:bel+u?n?m+|bl+m+|belom|belun)\s+ka(?:win|wn|vin|wim)?\b"
    r"|\btidak\s+kawin\b"
    r"|\bbelum\s*kawin\b"
)
_CERAI_RE = re.compile(r"\bcerai(?:\s+(?:hidup|mati))?\b")
_LABELED_STATUS_RE = re.compile(
    r"status\s*perkaw[a-z]{0,8}\s*[:|.]?\s*([a-z0-9 \-/]{2,56})",
)
_STATUS_WINDOW_RE = re.compile(
    r"(?:agama|\bgama\b|islam|isam)\b(.{0,220}?)(?:\bpekerjaan\b|\bpokerjaan\b|\bkewarganegaraan\b|\bberlaku\b)",
)


def is_helper_job_position(job_position: str) -> bool:
    s = " ".join((job_position or "").strip().casefold().split())
    if not s:
        return False
    if s in _HELPER_EXACT:
        return True
    tokens = re.split(r"[\s\-_/]+", s)
    return "helper" in tokens


def _soften_ocr(text: str) -> str:
    t = " ".join((text or "").casefold().split())
    t = t.replace("perkawinanr", "perkawinan ")
    t = re.sub(r"(perkawin[a-z]{0,4})(?=belum|kawin|cerai|blm|elwin)", r"\1 ", t)
    t = re.sub(r"\b8elum\b", "belum", t)
    t = re.sub(r"belun(?=k)", "belum", t)
    t = re.sub(r"\bbelun\b", "belum", t)
    t = re.sub(r"\bbelom\b", "belum", t)
    t = re.sub(r"\bisam\b", "islam", t)
    t = re.sub(r"\bpokerjaan\b", "pekerjaan", t)
    t = re.sub(r"\belwin\b", "belum kawin", t)
    t = re.sub(r"\bblwin\b", "belum kawin", t)
    t = re.sub(r"\bbelwin\b", "belum kawin", t)
    t = re.sub(r"\beel+um\b", "belum", t)
    t = re.sub(r"\bkav+in\b", "kawin", t)
    t = re.sub(r"\bkawln\b", "kawin", t)
    t = re.sub(r"\bkawim\b", "kawin", t)
    return t


def extract_ktp_marital_status(ocr_text: str) -> dict[str, Any]:
    """Ambil status perkawinan dari teks OCR KTP (nilai, bukan label field)."""
    n = _soften_ocr(ocr_text or "")
    compact = re.sub(r"[^a-z0-9]", "", n)

    labeled = ""
    m = _LABELED_STATUS_RE.search(n)
    if m:
        labeled = " ".join(m.group(1).split())

    status = _classify_marital_chunk(labeled) if labeled else None
    source = "labeled_field" if status else None
    if status is None:
        win = ""
        wm = _STATUS_WINDOW_RE.search(n)
        if wm:
            win = " ".join(wm.group(1).split())
        status = _classify_marital_chunk(win) if win else None
        if status:
            source = "agama_window"
    if status is None:
        status = _classify_marital_chunk(n, compact=compact)
        if status:
            source = "ocr_keywords"

    return {
        "status": status,
        "label": _STATUS_LABELS.get(status or "", None),
        "source": source,
    }


def _classify_marital_chunk(chunk: str, *, compact: str | None = None) -> str | None:
    t = _soften_ocr(chunk or "")
    if not t:
        return None
    c = compact if compact is not None else re.sub(r"[^a-z0-9]", "", t)
    if _CERAI_RE.search(t) or "cerai" in c:
        return "cerai"
    if (
        "belumkawin" in c
        or "belumka" in c
        or _BELUM_KAWIN_COMPACT.search(c)
        or _BELUM_KAWIN_WORDS.search(t)
    ):
        return "belum_kawin"
    stripped = re.sub(r"perkawinan", " ", t)
    if re.search(r"\bkawin\b", stripped):
        return "kawin"
    return None


def evaluate_helper_marital_gate(
    ocr_text: str,
    *,
    job_position: str,
    profile_id: str,
) -> dict[str, Any] | None:
    """
    None jika gate tidak berlaku.
    Jika berlaku: marital_pass True hanya bila status BELUM KAWIN terbaca.
    """
    if (profile_id or "").strip().casefold() != "ktp":
        return None
    if not is_helper_job_position(job_position):
        return None
    extracted = extract_ktp_marital_status(ocr_text)
    status = extracted.get("status")
    marital_pass = status == "belum_kawin"
    if marital_pass:
        reason = "Posisi Helper: status perkawinan BELUM KAWIN — lolos."
    elif status in {"kawin", "cerai"}:
        reason = (
            "Posisi Helper: status perkawinan harus BELUM KAWIN, "
            f"di KTP terbaca {extracted.get('label')}."
        )
    else:
        reason = (
            "Posisi Helper: status perkawinan harus BELUM KAWIN, "
            "tetapi nilai itu tidak terbaca di OCR KTP."
        )
    return {
        "evaluated": True,
        "job_position": "helper",
        "required_status": "BELUM KAWIN",
        "marital_pass": marital_pass,
        "extracted": extracted,
        "reason": reason,
    }


def apply_ktp_helper_marital_gate(
    validation: dict[str, Any],
    *,
    ocr_text: str,
    job_position: str,
    profile_id: str | None = None,
) -> dict[str, Any]:
    """Terapkan (atau terapkan ulang) gate Helper pada hasil validasi KTP."""
    out = dict(validation)
    profile = (profile_id or out.get("document_profile_id") or "").strip().casefold()
    gate = evaluate_helper_marital_gate(
        ocr_text,
        job_position=job_position,
        profile_id=profile,
    )
    out["job_position"] = (job_position or "").strip() or None
    if gate is None:
        out["marital_status"] = None
        out["marital_pass"] = None
        return out

    out["marital_status"] = gate["extracted"]
    out["marital_pass"] = gate["marital_pass"]
    out["helper_marital_gate"] = gate

    if not gate["marital_pass"]:
        out["document_matched"] = False
        explanation = dict(out.get("explanation") or {})
        details = [gate["reason"], *list(explanation.get("detail_lines") or [])]
        explanation["detail_lines"] = details
        explanation["summary"] = gate["reason"]
        blockers = list(explanation.get("primary_blockers") or [])
        if "HELPER_MARITAL" not in blockers:
            blockers.insert(0, "HELPER_MARITAL")
        explanation["primary_blockers"] = blockers
        gates = dict(explanation.get("gates") or {})
        gates["helper_marital"] = {
            "pass": False,
            "required_status": "BELUM KAWIN",
            "detected_status": (gate["extracted"] or {}).get("label"),
        }
        explanation["gates"] = gates
        out["explanation"] = explanation
        verdict = dict(out.get("verdict") or {})
        verdict["document_matched"] = False
        verdict["summary"] = gate["reason"]
        out["verdict"] = verdict
    else:
        explanation = dict(out.get("explanation") or {})
        details = list(explanation.get("detail_lines") or [])
        if gate["reason"] not in details:
            explanation["detail_lines"] = [gate["reason"], *details]
        gates = dict(explanation.get("gates") or {})
        gates["helper_marital"] = {
            "pass": True,
            "required_status": "BELUM KAWIN",
            "detected_status": (gate["extracted"] or {}).get("label"),
        }
        explanation["gates"] = gates
        out["explanation"] = explanation

    return out
