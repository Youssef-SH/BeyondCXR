from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


def test_rsna_campaign_cli_import_is_lazy_and_main_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli_name = "beyondcxr.training.rsna_campaign_cli"
    campaign_name = "beyondcxr.training.rsna_campaign"
    monkeypatch.delitem(sys.modules, cli_name, raising=False)
    monkeypatch.delitem(sys.modules, campaign_name, raising=False)

    cli = importlib.import_module(cli_name)

    assert campaign_name not in sys.modules
    campaign = ModuleType(campaign_name)
    campaign.main = lambda: 7  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, campaign_name, campaign)
    assert cli.main() == 7
