"""
Sanity checks for deployment compose files
"""

from pathlib import Path


def test_prod_compose_disables_dev_magic_link_response():
    content = Path("docker-compose.prod.yml").read_text(encoding="utf-8")
    assert "ALLOW_DEV_MAGIC_LINK_RESPONSE" in content
    assert '"0"' in content or "'0'" in content
