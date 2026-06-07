"""Prepare normalized HEV-Sandbox data from the original generation archives.

This repository already includes normalized data under ``data/benchmark``. Use
this script only if you need to rebuild it from the original archives:
``generated_new.zip``, ``generated_jailbreak.zip``, ``augmented_data.zip``, and
optionally ``results.zip``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hev_sandbox.dataset import (
    DOMAIN_NAMES,
    normalize_analysis,
    normalize_hazard,
    save_jsonl_gz,
    source_item_hash,
    stable_hash,
)
from hev_sandbox.utils import save_json


def read_zip_jsonl(zip_path: Path):
    data = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".jsonl"):
                continue
            match = re.search(r"generated_tests_([a-z]+)", name)
            if not match:
                continue
            domain = match.group(1)
            rows = []
            with archive.open(name) as handle:
                for line in handle:
                    if line.strip():
                        rows.append(json.loads(line))
            data[domain] = rows
    return data


def write_checksums(root: Path, output: Path):
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"checksums.sha256"}:
            h = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            lines.append(f"{h.hexdigest()}  {path.relative_to(root.parent)}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_base(generated_new: Path, output_dir: Path):
    source = read_zip_jsonl(generated_new)
    base_id_by_key = {}
    base_q_to_ids = defaultdict(list)
    counts = {}
    for domain, rows in sorted(source.items()):
        out = []
        for i, row in enumerate(rows, 1):
            field = (row.get("Test field") or "").strip()
            hazard, hazard_name = normalize_hazard(field)
            question = (row.get("Test question") or "").strip()
            analysis = normalize_analysis(row.get("Analysis"))
            item_id = f"{domain}_{i:06d}"
            key = source_item_hash(question, analysis)
            item = {
                "id": item_id,
                "base_id": item_id,
                "domain": domain,
                "domain_name": DOMAIN_NAMES.get(domain, domain),
                "split": "base",
                "user_type": "standard",
                "hazard": hazard,
                "hazard_name": hazard_name,
                "hazard_subtype": field,
                "question": question,
                "prompt": question,
                "reason": (row.get("Reason") or "").strip(),
                "analysis": analysis,
                "source_context_hash": stable_hash(row.get("original_conversation")),
                "source_context_included": False,
                "source_item_hash": key,
            }
            base_id_by_key[key] = item_id
            base_q_to_ids[question].append(item_id)
            out.append(item)
        save_jsonl_gz(out, output_dir / "base" / f"{domain}.jsonl.gz")
        counts[domain] = len(out)
    return source, base_id_by_key, base_q_to_ids, counts


def normalize_adversarial(generated_jailbreak: Path, output_dir: Path, base_id_by_key, base_q_to_ids):
    source = read_zip_jsonl(generated_jailbreak)
    counts = {}
    strategies = Counter()
    for domain, rows in sorted(source.items()):
        out = []
        for row in rows:
            question = (row.get("Test question") or "").strip()
            analysis = normalize_analysis(row.get("Analysis"))
            key = source_item_hash(question, analysis)
            base_id = base_id_by_key.get(key)
            if not base_id and len(base_q_to_ids.get(question, [])) == 1:
                base_id = base_q_to_ids[question][0]
            field = (row.get("Test field") or "").strip()
            hazard, hazard_name = normalize_hazard(field)
            prompts = []
            for j, prompt_info in enumerate(row.get("jailbreak_prompts") or [], 1):
                if isinstance(prompt_info, dict):
                    strategy = str(prompt_info.get("strategy") or f"strategy_{j}").strip()
                    prompt = str(prompt_info.get("prompt") or "").strip()
                else:
                    strategy = f"strategy_{j}"
                    prompt = str(prompt_info).strip()
                strategies[strategy] += 1
                prompts.append({"id": f"{base_id}_adv_{j:02d}", "strategy": strategy, "prompt": prompt})
            out.append({
                "id": base_id,
                "base_id": base_id,
                "domain": domain,
                "domain_name": DOMAIN_NAMES.get(domain, domain),
                "split": "adversarial",
                "user_type": "adversarial",
                "hazard": hazard,
                "hazard_name": hazard_name,
                "hazard_subtype": field,
                "question": question,
                "jailbreak_prompts": prompts,
                "reason": (row.get("Reason") or "").strip(),
                "analysis": analysis,
                "source_context_hash": stable_hash(row.get("original_conversation")),
                "source_context_included": False,
                "source_item_hash": key,
            })
        save_jsonl_gz(out, output_dir / "adversarial" / f"{domain}.jsonl.gz")
        counts[domain] = len(out)
    return counts, dict(strategies)

def normalize_perturbation(augmented_data: Path, output_dir: Path, base_id_by_key, base_q_to_ids):
    source = read_zip_jsonl(augmented_data)
    counts = {}
    strategies = Counter()
    for domain, rows in sorted(source.items()):
        out = []
        for i, row in enumerate(rows, 1):
            info = row.get("augmentation_info") or {}
            original_question = info.get("original_question", "") if isinstance(info, dict) else ""
            question_for_key = original_question or row.get("Test question") or ""
            analysis = normalize_analysis(row.get("Analysis"))
            key = source_item_hash(question_for_key, analysis)
            base_id = base_id_by_key.get(key)
            if not base_id and len(base_q_to_ids.get(question_for_key, [])) == 1:
                base_id = base_q_to_ids[question_for_key][0]
            field = (row.get("Test field") or "").strip()
            hazard, hazard_name = normalize_hazard(field)
            strategy = str(info.get("strategy") if isinstance(info, dict) else "unknown" or "unknown").strip()
            strategies[strategy] += 1
            question = (row.get("Test question") or "").strip()
            out.append({
                "id": f"{domain}_pert_{i:06d}",
                "base_id": base_id,
                "domain": domain,
                "domain_name": DOMAIN_NAMES.get(domain, domain),
                "split": "perturbation",
                "user_type": "perturbation",
                "hazard": hazard,
                "hazard_name": hazard_name,
                "hazard_subtype": field,
                "question": question,
                "prompt": question,
                "original_question": original_question,
                "augmentation_strategy": strategy,
                "reason": (row.get("Reason") or "").strip(),
                "analysis": analysis,
                "source_context_hash": stable_hash(row.get("original_conversation")),
                "source_context_included": False,
                "source_item_hash": key,
            })
        save_jsonl_gz(out, output_dir / "perturbation" / f"{domain}.jsonl.gz")
        counts[domain] = len(out)
    return counts, dict(strategies)


def main():
    parser = argparse.ArgumentParser(description="Normalize HEV source archives")
    parser.add_argument("--generated-new", "--generated_new", dest="generated_new", required=True)
    parser.add_argument("--generated-jailbreak", "--generated_jailbreak", dest="generated_jailbreak")
    parser.add_argument("--augmented-data", "--augmented_data", dest="augmented_data")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="data/benchmark")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, base_map, base_q_to_ids, base_counts = normalize_base(Path(args.generated_new), output_dir)
    manifest = {"base": {"counts_by_domain": base_counts, "total": sum(base_counts.values())}}
    if args.generated_jailbreak:
        counts, strategies = normalize_adversarial(Path(args.generated_jailbreak), output_dir, base_map, base_q_to_ids)
        manifest["adversarial"] = {"counts_by_domain": counts, "total": sum(counts.values()), "strategies": strategies}
    if args.augmented_data:
        counts, strategies = normalize_perturbation(Path(args.augmented_data), output_dir, base_map, base_q_to_ids)
        manifest["perturbation"] = {"counts_by_domain": counts, "total": sum(counts.values()), "strategies": strategies}
    save_json(manifest, output_dir / "dataset_manifest.rebuilt.json")
    write_checksums(output_dir, output_dir / "checksums.sha256")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
