"""Smoke tests for TRI-X-CDSS package imports."""

from trix_cdss import FRAMEWORK_NAME, __version__, perform_dizziness_triage


def test_package_exports_are_available() -> None:
    """Package-level exports remain importable for users."""
    assert FRAMEWORK_NAME == "TRI-X-CDSS"
    assert isinstance(__version__, str)
    assert callable(perform_dizziness_triage)
