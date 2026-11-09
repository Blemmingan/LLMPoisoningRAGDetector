"""Download the raw files of every knowledge-base source.

No scraping: each source publishes plain files (Parquet, JSON, YAML or a single HTML page).
Files land in `data/knowledge_base/raw/<source>/` (ignored by git) together with a
`manifest.json` recording URL, size, SHA-256 and download time, so every teammate can
rebuild exactly the same knowledge base with `corpus-forensics kb download`.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

HF_API = "https://huggingface.co/api/datasets"
HF_RESOLVE = "https://huggingface.co/datasets"
GH_RAW = "https://raw.githubusercontent.com"


@dataclass(frozen=True)
class RemoteFile:
    """A single file to download."""

    url: str
    filename: str
    needs_hf_token: bool = False


@dataclass(frozen=True)
class HuggingFaceDataset:
    """A Hugging Face dataset whose Parquet files are discovered through the API."""

    repo_id: str
    gated: bool = False


SOURCES: dict[str, list[RemoteFile] | HuggingFaceDataset] = {
    "deepset_prompt_injections": HuggingFaceDataset("deepset/prompt-injections"),
    "hackaprompt": HuggingFaceDataset("hackaprompt/hackaprompt-dataset", gated=True),
    "bipia": [
        RemoteFile(f"{GH_RAW}/microsoft/BIPIA/main/benchmark/{name}", name)
        for name in (
            "text_attack_train.json",
            "text_attack_test.json",
            "code_attack_train.json",
            "code_attack_test.json",
        )
    ]
    + [RemoteFile(f"{GH_RAW}/microsoft/BIPIA/main/LICENSE", "LICENSE")],
    "mitre_atlas": [
        RemoteFile(f"{GH_RAW}/mitre-atlas/atlas-data/main/dist/ATLAS.yaml", "ATLAS.yaml"),
        RemoteFile(f"{GH_RAW}/mitre-atlas/atlas-data/main/LICENSE", "LICENSE"),
    ],
    "owasp_llm_top10": [
        RemoteFile("https://genai.owasp.org/llmrisk/llm01-prompt-injection/", "llm01.html"),
        RemoteFile(
            "https://genai.owasp.org/llmrisk/llm072025-system-prompt-leakage/", "llm07.html"
        ),
    ],
}


def list_downloadable_sources() -> list[str]:
    """Names of the sources that can be downloaded."""
    return list(SOURCES)


def _hf_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _resolve_files(
    spec: list[RemoteFile] | HuggingFaceDataset, client: httpx.Client, hf_token: str | None
) -> list[RemoteFile]:
    """Turn a source spec into the concrete list of files to fetch."""
    if isinstance(spec, list):
        return spec
    if spec.gated and not hf_token:
        raise ValueError(
            f"{spec.repo_id} is gated on Hugging Face: set CF_HF_TOKEN in .env "
            "(read token from an account that accepted the dataset conditions)."
        )
    response = client.get(f"{HF_API}/{spec.repo_id}", headers=_hf_headers(hf_token))
    response.raise_for_status()
    siblings = [s["rfilename"] for s in response.json().get("siblings", [])]
    parquet = sorted(f for f in siblings if f.endswith(".parquet"))
    if not parquet:
        raise ValueError(f"No Parquet files found in {spec.repo_id}")
    return [
        RemoteFile(
            url=f"{HF_RESOLVE}/{spec.repo_id}/resolve/main/{path}",
            filename=Path(path).name,
            needs_hf_token=spec.gated,
        )
        for path in parquet
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_source(
    source: str,
    dest_root: Path | str,
    hf_token: str | None = None,
    force: bool = False,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Download every file of `source` into `dest_root/<source>/`.

    Files that already exist are kept unless `force` is True.

    Returns:
        Summary with the destination directory and per-file status.
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown source '{source}'. Available: {list_downloadable_sources()}")

    dest = Path(dest_root) / source
    dest.mkdir(parents=True, exist_ok=True)
    own_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=300)

    files_status: list[dict[str, Any]] = []
    try:
        for remote in _resolve_files(SOURCES[source], http, hf_token):
            target = dest / remote.filename
            if target.exists() and not force:
                files_status.append({"file": remote.filename, "status": "skipped"})
                continue
            headers = _hf_headers(hf_token) if remote.needs_hf_token else {}
            tmp = target.with_suffix(target.suffix + ".part")
            with http.stream("GET", remote.url, headers=headers) as response:
                response.raise_for_status()
                with open(tmp, "wb") as f:
                    for block in response.iter_bytes():
                        f.write(block)
            tmp.replace(target)
            files_status.append(
                {
                    "file": remote.filename,
                    "status": "downloaded",
                    "url": remote.url,
                    "bytes": target.stat().st_size,
                    "sha256": _sha256(target),
                    "downloaded_at": datetime.now(UTC).isoformat(),
                }
            )
    finally:
        if own_client:
            http.close()

    manifest_path = dest / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in files_status:
        if entry["status"] == "downloaded":
            manifest[entry["file"]] = {k: v for k, v in entry.items() if k not in ("status",)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {"source": source, "dest": str(dest), "files": files_status}
