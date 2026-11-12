"""Floutage des espèces sensibles (composant de confidentialité)."""

from __future__ import annotations

import pytest

from lpo_data_mcp import privacy


def test_is_sensitive_handles_booleans_numbers_and_labels():
    assert privacy.is_sensitive(True, protected_values=("protégée",)) is True
    assert privacy.is_sensitive(1, protected_values=("protégée",)) is True
    assert privacy.is_sensitive(0, protected_values=("protégée",)) is False
    assert privacy.is_sensitive(None, protected_values=("protégée",)) is False
    assert privacy.is_sensitive("protégée", protected_values=("protégée",)) is True
    assert privacy.is_sensitive("PROTÉGÉE", protected_values=("protégée",)) is True
    assert privacy.is_sensitive("non protégée", protected_values=("protégée",)) is False


def test_auto_mode_blurs_only_sensitive_rows(monkeypatch):
    monkeypatch.delenv("LPO_ALLOW_EXACT_COORDINATES", raising=False)
    assert privacy.resolve_mode("auto", sensitive=True) == "blurred"
    assert privacy.resolve_mode("auto", sensitive=False) == "exact"


def test_exact_mode_is_downgraded_for_sensitive_species_by_default(monkeypatch):
    monkeypatch.delenv("LPO_ALLOW_EXACT_COORDINATES", raising=False)
    assert privacy.resolve_mode("exact", sensitive=True) == "blurred"
    monkeypatch.setenv("LPO_ALLOW_EXACT_COORDINATES", "true")
    assert privacy.resolve_mode("exact", sensitive=True) == "exact"


def test_blur_rounds_to_grid(monkeypatch):
    monkeypatch.setenv("LPO_BLUR_DECIMALS", "1")
    lat, lon = privacy.apply_privacy(lat=48.7329, lon=-3.9871, sensitive=True, mode="blurred")
    assert (lat, lon) == (48.7, -4.0)
    assert privacy.grid_size_km() == 11


def test_department_mode_drops_coordinates():
    lat, lon = privacy.apply_privacy(lat=48.7, lon=-4.0, sensitive=True, mode="department")
    assert lat is None and lon is None


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError):
        privacy.resolve_mode("public", sensitive=False)


def test_nan_coordinates_are_not_exposed():
    lat, lon = privacy.apply_privacy(lat=float("nan"), lon=1.0, sensitive=False, mode="exact")
    assert lat is None and lon is None


def test_privacy_summary_is_explicit(monkeypatch):
    monkeypatch.setenv("LPO_BLUR_DECIMALS", "1")
    text = privacy.privacy_summary("blurred", sensitive_rows=3)
    assert "3" in text and "11 km" in text
