"""Utilities for the released HEV benchmark dataset.

The public benchmark uses a normalized schema that is compatible with the
fixed-dataset evaluation path. Raw source contexts are not included by default;
we keep only a SHA-256 hash so downstream users can detect duplicates without
redistributing source text.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .data_loader import iter_jsonl, load_jsonl

DOMAIN_ALIASES = {
    "conv": "daily_dialogue",
    "daily_dialogue": "daily_dialogue",
    "dialogue": "daily_dialogue",
    "edu": "education",
    "education": "education",
    "law": "law",
    "legal": "law",
    "medical": "medicine",
    "medicine": "medicine",
    "news": "news",
    "wiki": "encyclopedic_knowledge",
    "encyclopedia": "encyclopedic_knowledge",
    "encyclopedic": "encyclopedic_knowledge",
    "encyclopedic_knowledge": "encyclopedic_knowledge",
}

DOMAIN_FILE_STEMS = {
    "daily_dialogue": "conv",
    "education": "edu",
    "law": "law",
    "medicine": "medical",
    "news": "news",
    "encyclopedic_knowledge": "wiki",
}

DOMAIN_DISPLAY_NAMES = {
    "daily_dialogue": "Daily Dialogue",
    "education": "Education",
    "law": "Law",
    "medicine": "Medicine",
    "news": "News",
    "encyclopedic_knowledge": "Encyclopedic Knowledge",
}


def canonical_domain(name: str) -> str:
    key = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    return DOMAIN_ALIASES.get(key, key)


def domain_file_stem(domain: str) -> str:
    return DOMAIN_FILE_STEMS.get(canonical_domain(domain), canonical_domain(domain))


def short_domain_from_filename(filename: str) -> str:
    lower = Path(filename).name.lower()
    for token, canonical in [
        ("conv", "daily_dialogue"),
        ("edu", "education"),
        ("law", "law"),
        ("medical", "medicine"),
        ("medicine", "medicine"),
        ("news", "news"),
        ("wiki", "encyclopedic_knowledge"),
        ("encyclopedia", "encyclopedic_knowledge"),
    ]:
        if token in lower:
            return canonical
    return "unknown"


def normalize_hazard(field: str) -> Tuple[str, str]:
    """Return (major_hazard, hazard_subtype)."""
    subtype = (field or "unknown").strip() or "unknown"
    lower = subtype.lower()
    if "misinformation" in lower or "false" in lower or "fake" in lower:
        return "misinformation", subtype
    if "bias" in lower or "discrimination" in lower or "stereotype" in lower:
        return "bias_and_discrimination", subtype
    if "privacy" in lower or "personal" in lower or "confidential" in lower:
        return "privacy_breaches", subtype
    if "security" in lower or "malicious" in lower or "cyber" in lower or "exploit" in lower:
        return "malicious_use_and_security_threats", subtype
    if "ip" in lower or "copyright" in lower or "intellectual" in lower or "plagiarism" in lower:
        return "ip_infringement", subtype
    if "content" in lower or "harm" in lower or "safety" in lower:
        return "content_safety_and_harmfulness", subtype
    return re.sub(r"[^a-z0-9]+", "_", lower).strip("_") or "unknown", subtype


def normalize_safety(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "undetermined"
    if "unsafe" in text or "harmful" in text:
        return "unsafe"
    if text.startswith("safe") or " safe" in text:
        return "safe"
    return text


def normalize_analysis(analysis: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    analysis = analysis or {}
    for letter in "ABCD":
        candidates = [letter, f"Option {letter}", f"option {letter}", f"选项 {letter}", f"选择 {letter}"]
        option_info: Dict[str, Any] = {}
        for key in candidates:
            if key in analysis and isinstance(analysis[key], dict):
                option_info = analysis[key]
                break
        result[letter] = {
            "safety": normalize_safety(option_info.get("safety")),
            "explanation": str(option_info.get("explanation", "")).strip(),
        }
    return result


def stable_hash(obj: Any) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def base_key_from_raw(row: Dict[str, Any]) -> str:
    return stable_hash({
        "question": row.get("Test question") or row.get("test_question") or row.get("question") or "",
        "analysis": normalize_analysis(row.get("Analysis") or row.get("analysis") or {}),
    })


def get_question(row: Dict[str, Any]) -> str:
    return str(row.get("question") or row.get("Test question") or row.get("test_question") or row.get("prompt") or "").strip()


def get_analysis(row: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    analysis = row.get("analysis") or row.get("Analysis") or {}
    # Already normalized rows have A/B/C/D keys.
    return normalize_analysis(analysis)


def normalize_base_row(row: Dict[str, Any], domain: str, ordinal: int, include_source_context: bool = False) -> Dict[str, Any]:
    question = get_question(row)
    analysis = get_analysis(row)
    hazard, subtype = normalize_hazard(str(row.get("Test field") or row.get("test_field") or row.get("hazard") or "unknown"))
    source_context = str(row.get("original_conversation") or row.get("source_context") or "")
    base_id = f"{DOMAIN_FILE_STEMS.get(domain, domain)}_{ordinal:06d}"
    record = {
        "id": base_id,
        "base_id": base_id,
        "domain": domain,
        "domain_display": DOMAIN_DISPLAY_NAMES.get(domain, domain),
        "split": "base",
        "user_type": "standard",
        "hazard": hazard,
        "hazard_subtype": subtype,
        "test_field": subtype,
        "question": question,
        "prompt": question,
        "reason": str(row.get("Reason") or row.get("reason") or "").strip(),
        "analysis": analysis,
        "contact_frequency": float(row.get("contact_frequency", 1.0)),
        "source_context_hash": hashlib.sha256(source_context.encode("utf-8")).hexdigest() if source_context else None,
        "source_context_included": bool(include_source_context and source_context),
        "source_schema": "generated_new_v1",
        "source_key": base_key_from_raw(row),
    }
    if include_source_context and source_context:
        record["source_context"] = source_context
    return record


def normalize_augmented_row(
    row: Dict[str, Any],
    domain: str,
    ordinal: int,
    base_id_by_key: Dict[str, str],
    include_source_context: bool = False,
) -> Dict[str, Any]:
    question = get_question(row)
    analysis = get_analysis(row)
    hazard, subtype = normalize_hazard(str(row.get("Test field") or row.get("test_field") or row.get("hazard") or "unknown"))
    source_context = str(row.get("original_conversation") or row.get("source_context") or "")
    aug_info = row.get("augmentation_info") if isinstance(row.get("augmentation_info"), dict) else {}
    original_question = str(aug_info.get("original_question") or "").strip()
    base_lookup_key = stable_hash({"question": original_question, "analysis": analysis}) if original_question else base_key_from_raw(row)
    base_id = base_id_by_key.get(base_lookup_key)
    strategy = str(aug_info.get("strategy") or "unknown").strip() or "unknown"
    record_id = f"{DOMAIN_FILE_STEMS.get(domain, domain)}_pert_{ordinal:06d}"
    record = {
        "id": record_id,
        "base_id": base_id,
        "domain": domain,
        "domain_display": DOMAIN_DISPLAY_NAMES.get(domain, domain),
        "split": "perturbation",
        "user_type": "perturbation",
        "perturbation_strategy": strategy,
        "hazard": hazard,
        "hazard_subtype": subtype,
        "test_field": subtype,
        "question": question,
        "prompt": question,
        "original_question": original_question or None,
        "reason": str(row.get("Reason") or row.get("reason") or "").strip(),
        "analysis": analysis,
        "contact_frequency": float(row.get("contact_frequency", 1.0)),
        "source_context_hash": hashlib.sha256(source_context.encode("utf-8")).hexdigest() if source_context else None,
        "source_context_included": bool(include_source_context and source_context),
        "source_schema": "augmented_data_v1",
        "source_key": base_key_from_raw(row),
    }
    if include_source_context and source_context:
        record["source_context"] = source_context
    return record


def normalize_jailbreak_rows(
    row: Dict[str, Any],
    domain: str,
    ordinal: int,
    base_id_by_key: Dict[str, str],
    include_source_context: bool = False,
) -> List[Dict[str, Any]]:
    question = get_question(row)
    analysis = get_analysis(row)
    hazard, subtype = normalize_hazard(str(row.get("Test field") or row.get("test_field") or row.get("hazard") or "unknown"))
    source_context = str(row.get("original_conversation") or row.get("source_context") or "")
    base_key = base_key_from_raw(row)
    base_id = base_id_by_key.get(base_key)
    prompts = row.get("jailbreak_prompts") if isinstance(row.get("jailbreak_prompts"), list) else []
    records: List[Dict[str, Any]] = []
    for idx, item in enumerate(prompts, 1):
        if isinstance(item, dict):
            strategy = str(item.get("strategy") or f"strategy_{idx}").strip()
            prompt = str(item.get("prompt") or "").strip()
        else:
            strategy = f"strategy_{idx}"
            prompt = str(item).strip()
        record_id = f"{DOMAIN_FILE_STEMS.get(domain, domain)}_adv_{ordinal:06d}_{idx:02d}"
        record = {
            "id": record_id,
            "base_id": base_id,
            "domain": domain,
            "domain_display": DOMAIN_DISPLAY_NAMES.get(domain, domain),
            "split": "adversarial",
            "user_type": "adversarial",
            "jailbreak_strategy": strategy,
            "hazard": hazard,
            "hazard_subtype": subtype,
            "test_field": subtype,
            "question": question,
            "prompt": prompt,
            "original_question": question,
            "reason": str(row.get("Reason") or row.get("reason") or "").strip(),
            "analysis": analysis,
            "contact_frequency": float(row.get("contact_frequency", 1.0)),
            "source_context_hash": hashlib.sha256(source_context.encode("utf-8")).hexdigest() if source_context else None,
            "source_context_included": bool(include_source_context and source_context),
            "source_schema": "generated_jailbreak_v1",
            "source_key": base_key,
        }
        if include_source_context and source_context:
            record["source_context"] = source_context
        records.append(record)
    return records


def list_split_files(dataset_dir: str | Path, split: str, domains: Optional[Sequence[str]] = None) -> List[Path]:
    root = Path(dataset_dir)
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")
    selected = [canonical_domain(d) for d in domains] if domains else list(DOMAIN_FILE_STEMS.keys())
    files: List[Path] = []
    for domain in selected:
        stem = domain_file_stem(domain)
        for candidate in [split_dir / f"{domain}.jsonl.gz", split_dir / f"{stem}.jsonl.gz", split_dir / f"{domain}.jsonl"]:
            if candidate.exists():
                files.append(candidate)
                break
    if not files:
        raise FileNotFoundError(f"No files found for split={split!r}, domains={domains!r} in {split_dir}")
    return files


def load_benchmark_records(
    dataset_dir: str | Path = "data/benchmark",
    split: str = "base",
    domains: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load records from the normalized benchmark.

    Args:
        dataset_dir: Directory containing split subdirectories.
        split: One of base, perturbation, adversarial.
        domains: Optional list of canonical or short domain names.
        limit: Optional maximum records per split across all selected domains.
    """
    records: List[Dict[str, Any]] = []
    for path in list_split_files(dataset_dir, split, domains):
        for item in iter_jsonl(path):
            records.append(item)
            if limit is not None and len(records) >= limit:
                return records
    return records


def record_to_scenario(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a normalized benchmark record to the scenario schema used by agents."""
    question = record.get("question") or record.get("prompt") or ""
    return {
        "id": record.get("base_id") or record.get("id"),
        "domain": record.get("domain"),
        "background": record.get("source_context", ""),
        "user_intent": record.get("question", ""),
        "reason": record.get("reason", ""),
        "hazard": record.get("hazard", "unknown"),
        "hazard_subtype": record.get("hazard_subtype", record.get("test_field", "unknown")),
        "test_field": record.get("test_field", record.get("hazard_subtype", "unknown")),
        "test_question": question,
        "question": question,
        "analysis": record.get("analysis", {}),
        "contact_frequency": record.get("contact_frequency", 1.0),
    }
