"""Bybit V5 REST base URLs (official docs: "Integration Guidance").

Adapters receive one of these explicitly; nothing selects mainnet by default.
"""

from __future__ import annotations

from typing import Final

BYBIT_TESTNET_REST_URL: Final = "https://api-testnet.bybit.com"
BYBIT_MAINNET_REST_URL: Final = "https://api.bybit.com"
