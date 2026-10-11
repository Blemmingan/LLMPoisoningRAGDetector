"""Evaluation dataset loader.

ASSUMED DATASET FORMAT:
An evaluation dataset folder contains:
1. Source documents (e.g. .txt, .md, .html, .pdf)
2. A `labels.jsonl` ground-truth file where each line is a JSON object with:
   - "documento": filename (e.g., "sample_01.txt")
   - "categoria": "limpio" | "envenenado" | "hard_negative"
   - "malicioso": bool
   Optional fields:
   - "tipo_de_ataque": string (matching TipoDeAtaque enum values)
   - "tecnica_de_ocultamiento": string or list of strings (matching TecnicaDeOcultamiento)
   - "fragmento_exacto": string (exact injected payload)
   - "posicion": "inicio" | "medio" | "fin"
   - "sutileza": "obvio" | "parafraseado" | "disfrazado"

This loader is tolerant of missing optional fields.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from corpus_forensics.schemas import (
    CategoriaDocumento,
    PosicionAtaque,
    SutilezaAtaque,
    TecnicaDeOcultamiento,
    TipoDeAtaque,
)


class EvalLabel(BaseModel):
    """Ground truth annotation record for an evaluation document."""

    documento: str
    categoria: CategoriaDocumento
    malicioso: bool
    tipo_de_ataque: TipoDeAtaque | None = None
    tecnica_de_ocultamiento: list[TecnicaDeOcultamiento] = Field(default_factory=list)
    fragmento_exacto: str | None = None
    posicion: PosicionAtaque | None = None
    sutileza: SutilezaAtaque | None = None


class EvalSample(BaseModel):
    """Evaluation sample pairing the actual document file path with its ground truth."""

    file_path: Path
    label: EvalLabel


class EvalDatasetLoader:
    """Loads evaluation datasets adhering to the assumed directory layout."""

    def __init__(self, dataset_dir: Path | str) -> None:
        """Initialize evaluation dataset loader.

        Args:
            dataset_dir: Directory containing documents and labels.jsonl.
        """
        self.dataset_dir = Path(dataset_dir)
        self.labels_path = self.dataset_dir / "labels.jsonl"

    def load(self) -> list[EvalSample]:
        """Load and parse evaluation samples.

        Returns:
            List of EvalSample pairs.

        Raises:
            FileNotFoundError: If dataset directory or labels.jsonl does not exist.
        """
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Evaluation directory not found: {self.dataset_dir}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Evaluation labels file not found: {self.labels_path}")

        samples: list[EvalSample] = []
        with open(self.labels_path, encoding="utf-8") as f:
            for _line_num, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    raw_data = json.loads(clean_line)
                    label = self._parse_label(raw_data)
                    doc_path = self.dataset_dir / label.documento
                    samples.append(EvalSample(file_path=doc_path, label=label))
                except Exception:
                    # Tolerant parsing: skip or report malformed line
                    continue

        return samples

    def _parse_label(self, data: dict) -> EvalLabel:
        """Parse dictionary tolerantly into EvalLabel."""
        cat_str = str(data.get("categoria", "limpio")).lower()
        try:
            categoria = CategoriaDocumento(cat_str)
        except ValueError:
            categoria = CategoriaDocumento.LIMPIO

        tipo_str = data.get("tipo_de_ataque")
        tipo = None
        if tipo_str:
            try:
                tipo = TipoDeAtaque(str(tipo_str).lower())
            except ValueError:
                tipo = TipoDeAtaque.OTRO

        tecnica_raw = data.get("tecnica_de_ocultamiento")
        tecnicas: list[TecnicaDeOcultamiento] = []
        if isinstance(tecnica_raw, list):
            for t in tecnica_raw:
                try:
                    tecnicas.append(TecnicaDeOcultamiento(str(t).lower()))
                except ValueError:
                    pass
        elif isinstance(tecnica_raw, str):
            try:
                tecnicas.append(TecnicaDeOcultamiento(tecnica_raw.lower()))
            except ValueError:
                pass

        pos_str = data.get("posicion")
        pos = None
        if pos_str:
            try:
                pos = PosicionAtaque(str(pos_str).lower())
            except ValueError:
                pos = None

        sut_str = data.get("sutileza")
        sut = None
        if sut_str:
            try:
                sut = SutilezaAtaque(str(sut_str).lower())
            except ValueError:
                sut = None

        return EvalLabel(
            documento=str(data.get("documento", "")),
            categoria=categoria,
            malicioso=bool(data.get("malicioso", False)),
            tipo_de_ataque=tipo,
            tecnica_de_ocultamiento=tecnicas,
            fragmento_exacto=data.get("fragmento_exacto"),
            posicion=pos,
            sutileza=sut,
        )
