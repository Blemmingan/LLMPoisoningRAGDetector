"""HTML text hidden by CSS (benign markers only)."""

from pathlib import Path

import pytest

from corpus_forensics.ingestion.html_loader import HtmlLoader, hiding_reason

PAGE = """<!DOCTYPE html><html><head><style>
/* hiding rules */
.sneaky { display: none; }
p.ghost, .other { color: #ffffff; }
#away { position: absolute; left: -9999px; }
@media screen { .tiny { font-size: 0px; } }
.dark { color: #fff; background-color: #222; }
</style></head><body>
<h1>Lighthouses</h1>
<p>The lighthouse guides ships along the coast.</p>
<div class="sneaky">HIDDEN MARKER ONE <span>nested</span></div>
<p class="ghost">HIDDEN MARKER TWO</p>
<p id="away">HIDDEN MARKER THREE</p>
<p class="tiny">HIDDEN MARKER FOUR</p>
<p style="opacity: 0">HIDDEN MARKER FIVE</p>
<p hidden>HIDDEN MARKER SIX</p>
<p class="dark">White text on a dark band stays visible.</p>
</body></html>"""


def test_css_hidden_text_goes_to_hidden_channel(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text(PAGE, encoding="utf-8")
    doc = HtmlLoader().load(path)
    assert "lighthouse guides ships" in doc.visible_text
    assert "White text on a dark band stays visible." in doc.visible_text
    for marker, reason in [
        ("HIDDEN MARKER ONE nested", "display_none"),
        ("HIDDEN MARKER TWO", "texto_blanco"),
        ("HIDDEN MARKER THREE", "fuera_de_pantalla"),
        ("HIDDEN MARKER FOUR", "fuente_cero"),
        ("HIDDEN MARKER FIVE", "opacidad_cero"),
        ("HIDDEN MARKER SIX", "atributo_hidden"),
    ]:
        assert marker.split(" nested")[0] not in doc.visible_text
        assert f"({reason}) {marker}" in doc.hidden_content
    assert doc.metadata["hidden_elements_count"] == 6


@pytest.mark.parametrize(
    ("css", "reason"),
    [
        ("display:none", "display_none"),
        ("visibility: hidden;", "visibility_hidden"),
        ("opacity:0;", "opacidad_cero"),
        ("font-size:0", "fuente_cero"),
        ("text-indent:-9999px", "fuera_de_pantalla"),
        ("clip: rect(0 0 0 0)", "recorte"),
        ("color: white; background: #fff", "texto_blanco"),
        ("color: rgba(0,0,0,0)", "texto_blanco"),
        ("color: white; background: black", None),
        ("opacity: 0.5", None),
        ("font-size: 0.9em", None),
        ("margin-left: -10px", None),
    ],
)
def test_hiding_reason(css: str, reason: str | None) -> None:
    assert hiding_reason(css) == reason
