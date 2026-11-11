"""Builds tiny PDFs by hand for loader tests (no PDF-writing dependency needed).

Each segment is raw content-stream operators for one text object; the page is US Letter
(612 x 792 pt) with Helvetica as /F1 and an ExtGState /GS0 with fill alpha 0.
"""

from pathlib import Path


def text_segment(text: str, x: float, y: float, *, size: float = 12, prefix: str = "") -> str:
    """One text object: optional state operators (e.g. '3 Tr', '1 1 1 rg', '/GS0 gs')."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"BT {prefix} /F1 {size} Tf {x} {y} Td ({escaped}) Tj ET"


def write_pdf(path: Path, segments: list[str]) -> Path:
    """Write a one-page PDF drawing the given text segments."""
    stream = "\n".join(segments).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> /ExtGState << /GS0 6 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /ExtGState /ca 0 >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(bytes(out))
    return path
