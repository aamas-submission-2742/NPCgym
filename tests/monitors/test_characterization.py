"""Fixed observable outcomes for every parameter-free registered monitor."""

import pytest
from monitor_characterization import CASES, FACTORIES, assert_characterization


def test_characterizations_cover_all_registered_monitors():
    assert set(CASES) == set(FACTORIES)
    assert len(CASES) == 32


@pytest.mark.parametrize("identifier", sorted(CASES))
@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_registered_monitor_characterization(identifier, ending):
    assert_characterization(identifier, FACTORIES[identifier], ending)
