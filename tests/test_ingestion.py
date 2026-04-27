"""
Tests the ingestion pipeline

"""

from types import SimpleNamespace
from ingestion import clean_id

def test_clean_id_removes_arxiv_version_suffix():
    assert clean_id("2401.12345v2") == "2401.12345"

def test_clean_id_leaves_unversioned_id_unchanged():
    assert clean_id("2401.12345") == "2401.12345"