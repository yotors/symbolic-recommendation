"""Path-based test markers for fast and integration selections."""

from pathlib import Path


_TESTS_ROOT = Path(__file__).parent


def pytest_collection_modifyitems(items):
    for item in items:
        try:
            relative = Path(item.path).relative_to(_TESTS_ROOT)
        except ValueError:
            continue
        if relative.parts[0] in {"integration", "unit"}:
            item.add_marker(relative.parts[0])
