"""Garde-fous de qualité de l'interface web.

Ces tests ne vérifient pas un goût : ils vérifient des propriétés mesurables du
front statique, celles qui se dégradent silencieusement au fil des retouches.

1. Contraste WCAG AA (4.5:1) des couples de couleurs réellement utilisés.
2. Absence des marqueurs qui trahissent une interface générée sans intention :
   dégradés décoratifs, ombres portées empilées, emoji tenant lieu d'identité,
   capitales espacées en série, polices distantes non chargées.
3. Accessibilité de base : langue, titre, lien d'évitement, focus visible,
   région vivante, respect de `prefers-reduced-motion`.
4. Cohérence structurelle : tout identifiant lu par `app.js` existe dans le
   document, et toute classe d'état produite par le renderer est stylée.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "web" / "static"
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "styles.css").read_text(encoding="utf-8")
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
RENDERER_JS = (STATIC / "renderer.js").read_text(encoding="utf-8")


# ── Contraste ───────────────────────────────────────────────────────────────


def _relative_luminance(hex_color: str) -> float:
    value = hex_color.lstrip("#")
    channels = [int(value[index : index + 2], 16) / 255 for index in (0, 2, 4)]

    def linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linear(channel) for channel in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(foreground: str, background: str) -> float:
    light, dark = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)), reverse=True
    )
    return (light + 0.05) / (dark + 0.05)


def _token(name: str) -> str:
    match = re.search(rf"--{name}:\s*(#[0-9a-fA-F]{{6}})\s*;", CSS)
    assert match is not None, f"jeton --{name} introuvable dans styles.css"
    return match.group(1)


@pytest.mark.parametrize(
    ("foreground", "background"),
    [
        ("ink", "paper"),
        ("ink", "surface"),
        ("ink-2", "surface"),
        ("ink-2", "paper"),
        ("ink-3", "surface"),
        ("ink-3", "paper"),
        ("ink-3", "surface-2"),
        ("accent", "surface"),
        ("accent", "accent-soft"),
        ("warn", "warn-soft"),
        ("error", "error-soft"),
    ],
)
def test_text_colors_meet_wcag_aa(foreground: str, background: str) -> None:
    """Tout texte réel dépasse 4.5:1, y compris le texte secondaire."""
    ratio = contrast(_token(foreground), _token(background))
    assert ratio >= 4.5, f"{foreground} sur {background} : {ratio:.2f}:1 < 4.5:1"


def test_primary_button_text_is_readable() -> None:
    assert contrast("#ffffff", _token("accent")) >= 4.5


def test_control_borders_are_visible() -> None:
    """WCAG 1.4.11 : la limite d'un champ ou d'un bouton atteint 3:1, sinon le
    contrôle disparaît pour un œil qui distingue mal les gris."""
    for background in ("surface", "paper"):
        ratio = contrast(_token("control-line"), _token(background))
        assert ratio >= 3.0, f"bordure sur {background} : {ratio:.2f}:1 < 3:1"


def test_bar_fill_stands_out_from_its_track() -> None:
    assert contrast(_token("accent"), _token("surface-2")) >= 3.0


# ── Marqueurs d'interface générée ───────────────────────────────────────────


@pytest.mark.parametrize("marker", ["linear-gradient", "radial-gradient", "conic-gradient"])
def test_no_decorative_gradients(marker: str) -> None:
    assert marker not in CSS


def test_no_stacked_shadows() -> None:
    """Les séparations se font par filet, jamais par ombre portée décorative."""
    assert "box-shadow" not in CSS


def test_no_emoji_as_identity() -> None:
    emoji = re.compile("[\U0001f300-\U0001faff\u2600-\u27bf]")
    assert not emoji.search(HTML), "emoji trouvé dans index.html"
    assert not emoji.search(APP_JS)


def test_no_uppercase_eyebrows() -> None:
    """Les libellés ne sont pas criés en capitales : la hiérarchie passe par
    la taille et la couleur."""
    assert "text-transform: uppercase" not in CSS
    assert "text-transform:uppercase" not in CSS


def test_no_remote_fonts_or_imports() -> None:
    assert "@import" not in CSS
    assert "fonts.googleapis" not in CSS + HTML
    assert not re.search(r"<link[^>]+href=\"https?://", HTML)


def test_no_placeholder_inter_font_stack() -> None:
    """Une police déclarée en tête mais jamais chargée rend un texte différent
    d'un poste à l'autre : la pile est explicitement système."""
    assert re.search(r"--font-ui:\s*\n?\s*ui-sans-serif", CSS)
    assert '"Inter"' not in CSS


def test_single_accent_family() -> None:
    """Un seul accent, décliné en deux valeurs : pas de bleu + vert côte à côte."""
    accents = set(re.findall(r"--accent(?:-hover|-soft)?:\s*(#[0-9a-fA-F]{6})", CSS))
    assert len(accents) == 3, accents
    for blue in ("#1f4e79", "#2e7d32", "#1565c0"):
        assert blue not in CSS


# ── Accessibilité de base ───────────────────────────────────────────────────


def test_document_language_and_title() -> None:
    assert re.search(r'<html lang="fr">', HTML)
    assert re.search(r"<title>[^<]{20,}</title>", HTML)


def test_viewport_and_description() -> None:
    assert 'name="viewport"' in HTML
    assert 'name="description"' in HTML


def test_skip_link_targets_main_region() -> None:
    assert 'class="skip-link" href="#reponse"' in HTML
    assert 'id="reponse"' in HTML


def test_focus_is_always_visible() -> None:
    assert re.search(r":focus-visible\s*\{[^}]*outline:\s*2px solid", CSS)
    # Aucune règle ne supprime l'anneau de focus.
    assert "outline: none" not in CSS


def test_conversation_is_a_live_region() -> None:
    assert 'role="log"' in HTML
    assert 'aria-live="polite"' in HTML


def test_scroll_regions_are_keyboard_reachable() -> None:
    """Une zone qui défile sans `tabindex` est inatteignable au clavier."""
    for element_id in ('id="chat-log"', 'id="surface-scroll"'):
        start = HTML.index(element_id)
        tag = HTML[HTML.rindex("<div", 0, start) : HTML.index(">", start) + 1]
        assert 'tabindex="0"' in tag, f"{element_id} n'est pas focusable"


def test_busy_and_progress_states_are_exposed() -> None:
    assert 'setAttribute("aria-busy"' in APP_JS
    assert 'role="progressbar"' in HTML
    assert "aria-pressed" in HTML


def test_motion_respects_user_preference() -> None:
    assert "prefers-reduced-motion" in CSS
    assert "prefers-reduced-motion" in APP_JS
    # Toute animation déclarée doit être sous condition de préférence.
    animations = CSS.count("animation:")
    guarded = CSS.count("prefers-reduced-motion: no-preference")
    assert animations >= 1 and guarded >= 1


def test_input_has_visible_label() -> None:
    assert re.search(r'<label class="sr-only" for="question">', HTML)
    assert re.search(r'id="question"', HTML)


def test_no_inline_style_in_document() -> None:
    assert not re.search(r"<[^>]+\sstyle=", HTML)


def test_no_noscript_blank_page() -> None:
    assert "<noscript>" in HTML


# ── Cohérence structurelle ──────────────────────────────────────────────────


def test_every_dom_id_used_by_app_js_exists() -> None:
    used = set(re.findall(r"getElementById\(\"([^\"]+)\"\)", APP_JS))
    declared = set(re.findall(r'id="([^"]+)"', HTML))
    missing = sorted(used - declared)
    assert not missing, f"identifiants absents de index.html : {missing}"


def test_template_is_not_used_for_console_output() -> None:
    """Le gabarit de trace doit rester aligné avec ce que construit app.js."""
    assert 'id="trace-template"' in HTML
    assert "trace-template" in APP_JS or "trace-list" in APP_JS


@pytest.mark.parametrize(
    "selector",
    [
        ".a2ui-card",
        ".a2ui-kpi",
        ".kpi-value",
        ".kpi-hint",
        ".a2ui-chart",
        ".chart-svg",
        ".grid-line",
        ".axis-label",
        ".bar-track",
        ".bar-fill",
        ".bar-value",
        ".map-svg",
        ".map-dept",
        ".map-point.is-sensitive",
        ".privacy-badge",
        ".a2ui-notice.is-warning",
        ".a2ui-notice.is-error",
        ".a2ui-select select",
        ".data-table",
        ".a2ui-placeholder",
        ".a2ui-divider",
        ".a2ui-button.is-primary",
    ],
)
def test_catalogue_components_are_styled(selector: str) -> None:
    """Chaque composant du catalogue a un style : aucun rendu par défaut du
    navigateur ne doit apparaître dans une réponse."""
    assert selector in CSS, f"aucune règle pour {selector}"


def test_renderer_adds_alignment_classes_not_inline_styles() -> None:
    assert 'classList.add("is-num")' in RENDERER_JS
    assert "style.textAlign" not in RENDERER_JS


def test_no_pill_buttons() -> None:
    """Le rayon 999 px est réservé aux pastilles d'état, pas aux boutons."""
    buttons = re.findall(r"\.[a-z0-9_-]*(?:button|toggle|input)[^{]*\{[^}]*\}", CSS)
    assert all("border-radius: var(--pill)" not in rule for rule in buttons)


def test_every_css_token_is_defined() -> None:
    used = set(re.findall(r"var\(--([a-z0-9-]+)", CSS))
    defined = set(re.findall(r"--([a-z0-9-]+):", CSS))
    assert not sorted(used - defined), f"jetons non définis : {sorted(used - defined)}"


def test_every_class_in_document_is_styled() -> None:
    classes: set[str] = set()
    for attribute in re.findall(r'class="([^"]+)"', HTML):
        classes.update(attribute.split())
    missing = sorted(name for name in classes if f".{name}" not in CSS)
    assert not missing, f"classes sans règle CSS : {missing}"
