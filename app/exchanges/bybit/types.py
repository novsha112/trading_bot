"""JSON types shared by Bybit request builders and the private transport.

Kept apart from ``private_rest`` so pure request mapping does not depend on the
HTTP transport.
"""

from __future__ import annotations

JsonValue = str | int | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
"""JSON-native values only: numbers with financial meaning are sent as strings."""
