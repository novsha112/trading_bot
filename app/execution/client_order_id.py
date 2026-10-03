"""Stable, exchange-neutral namespace of managed client order ids
(docs/ARCHITECTURE.md 13.5).

Wire format, version 1 (ASCII, canonical, one representation per value)::

    tb1_<namespace>_<order-token>

* ``tb`` + version digits: the managed family prefix; ``tb1`` is the only
  supported version;
* ``namespace``: the stable identity of one installation / bot instance,
  ``[a-z0-9]{1,12}``; configured and injected explicitly (never generated at
  startup), so the same configuration owns the same ids across restarts;
* ``order-token``: the unique part of one order, ``[a-z0-9]{1,32}``; the
  generator's responsibility (the namespace does not guarantee uniqueness).

``_`` is the only delimiter and appears in no segment, so parsing is
unambiguous. Nothing about strategy, symbol, side or price is encoded; neither
is ``intent_id``. The length bounds are local resource-safety bounds of this
format (``len("tb1_") + 12 + 1 + 32 = 49``), not any exchange's limit: a future
exchange adapter checks the ids against its own capability. The default
generator produces 16 hex characters, so with a namespace of up to 12
characters an id has at most 33 characters.

Parsing (``parse_client_order_id``) never raises for a ``str`` or ``None`` and
distinguishes:

* ``ABSENT``: no client id at all (None);
* ``UNMANAGED``: an ordinary string outside the managed family;
* ``MANAGED``: a valid v1 id (namespace and token available);
* ``MALFORMED_MANAGED``: it starts like a managed id (``tb<digits>`` followed by
  ``_`` or the end, ignoring the ASCII case of ``tb`` and leading
  whitespace / control characters) but is not a valid canonical v1 id;
* ``UNSUPPORTED_MANAGED_VERSION``: a canonical managed prefix of another version.

A malformed or unsupported managed-looking id is deliberately NOT unmanaged: a
damaged id of our family must later fail closed, never pass as foreign.
``classify_client_order_id(value, namespace=...)`` maps this to ownership by one
namespace: OURS, OTHER (unmanaged or another valid namespace), ABSENT,
MALFORMED_MANAGED, UNSUPPORTED_MANAGED_VERSION. It classifies the id string
only; what that means for an exchange order is the recovery layer's decision.

Pure: no I/O, persistence, exchange access or clock. Client ids and namespaces
are not secrets.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from app.domain.errors import DomainValidationError
from app.domain.intents import PlaceOrderIntent

FAMILY_PREFIX: Final = "tb"
SUPPORTED_VERSION: Final = 1
DELIMITER: Final = "_"
NAMESPACE_MAX_LENGTH: Final = 12
ORDER_TOKEN_MAX_LENGTH: Final = 32
DEFAULT_TOKEN_BYTES: Final = 8
"""Entropy of a default order token: 64 bits as 16 lowercase hex characters."""

_NAMESPACE: Final = re.compile(rf"[a-z0-9]{{1,{NAMESPACE_MAX_LENGTH}}}", re.ASCII)
_ORDER_TOKEN: Final = re.compile(rf"[a-z0-9]{{1,{ORDER_TOKEN_MAX_LENGTH}}}", re.ASCII)
# The managed family as it may appear, also damaged: any ASCII case of "tb",
# ASCII digits, then the delimiter or the end of the string.
_FAMILY: Final = re.compile(r"(?i:tb)([0-9]+)(?:_|\Z)", re.ASCII)
_IGNORED_LEADING: Final = "".join(chr(code) for code in range(33)) + "\x7f"


def _exact_str(value: object, field: str) -> str:
    if type(value) is not str:
        raise DomainValidationError(f"{field} must be a str, got {type(value).__name__}")
    return value


@dataclass(frozen=True, slots=True)
class ClientOrderNamespace:
    """The stable namespace of one installation / bot instance."""

    token: str

    def __post_init__(self) -> None:
        token = _exact_str(self.token, "namespace")
        if not _NAMESPACE.fullmatch(token):
            raise DomainValidationError(
                f"namespace must be 1-{NAMESPACE_MAX_LENGTH} characters of [a-z0-9], got {token!r}"
            )

    def build(self, order_token: str) -> str:
        """The managed client order id of ``order_token`` in this namespace."""
        token = _exact_str(order_token, "order_token")
        if not _ORDER_TOKEN.fullmatch(token):
            raise DomainValidationError(
                f"order_token must be 1-{ORDER_TOKEN_MAX_LENGTH} characters of [a-z0-9], "
                f"got {token!r}"
            )
        return f"{FAMILY_PREFIX}{SUPPORTED_VERSION}{DELIMITER}{self.token}{DELIMITER}{token}"

    def owns(self, client_order_id: str | None) -> bool:
        """True only for a valid managed id of exactly this namespace."""
        return classify_client_order_id(client_order_id, namespace=self) is (
            ClientOrderIdOwnership.OURS
        )


class ClientOrderIdFormat(StrEnum):
    """What the wire format of a client order id says (namespace-independent)."""

    ABSENT = "absent"
    UNMANAGED = "unmanaged"
    MANAGED = "managed"
    MALFORMED_MANAGED = "malformed_managed"
    UNSUPPORTED_MANAGED_VERSION = "unsupported_managed_version"


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedClientOrderId:
    """The parts of a valid managed client order id."""

    version: int
    namespace: ClientOrderNamespace
    order_token: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ClientOrderIdParse:
    """Result of ``parse_client_order_id``; ``parsed`` only for MANAGED."""

    format: ClientOrderIdFormat
    parsed: ParsedClientOrderId | None = None


def parse_client_order_id(value: str | None) -> ClientOrderIdParse:
    """Parse a client order id; never raises for a ``str`` or ``None``.

    Raises:
        DomainValidationError: ``value`` is neither a ``str`` nor ``None``.
    """
    if value is None:
        return ClientOrderIdParse(format=ClientOrderIdFormat.ABSENT)
    text = _exact_str(value, "client_order_id")
    family = _FAMILY.match(text.lstrip(_IGNORED_LEADING))
    if family is None:
        return ClientOrderIdParse(format=ClientOrderIdFormat.UNMANAGED)
    version_text = family.group(1)
    canonical_prefix = text.startswith(FAMILY_PREFIX)
    if canonical_prefix and version_text != str(SUPPORTED_VERSION) and version_text[0] != "0":
        return ClientOrderIdParse(format=ClientOrderIdFormat.UNSUPPORTED_MANAGED_VERSION)
    parts = text.split(DELIMITER)
    if (
        not canonical_prefix
        or version_text != str(SUPPORTED_VERSION)
        or len(parts) != 3
        or parts[0] != f"{FAMILY_PREFIX}{SUPPORTED_VERSION}"
        or not _NAMESPACE.fullmatch(parts[1])
        or not _ORDER_TOKEN.fullmatch(parts[2])
    ):
        return ClientOrderIdParse(format=ClientOrderIdFormat.MALFORMED_MANAGED)
    return ClientOrderIdParse(
        format=ClientOrderIdFormat.MANAGED,
        parsed=ParsedClientOrderId(
            version=SUPPORTED_VERSION,
            namespace=ClientOrderNamespace(parts[1]),
            order_token=parts[2],
        ),
    )


class ClientOrderIdOwnership(StrEnum):
    """Ownership of a client order id string by one namespace."""

    OURS = "ours"
    OTHER = "other"
    """Unmanaged, or a valid managed id of another namespace."""
    ABSENT = "absent"
    MALFORMED_MANAGED = "malformed_managed"
    UNSUPPORTED_MANAGED_VERSION = "unsupported_managed_version"


def classify_client_order_id(
    value: str | None, *, namespace: ClientOrderNamespace
) -> ClientOrderIdOwnership:
    """Wire-format ownership of ``value`` by ``namespace`` (see the module doc)."""
    if type(namespace) is not ClientOrderNamespace:
        raise DomainValidationError("namespace must be a ClientOrderNamespace")
    result = parse_client_order_id(value)
    parsed = result.parsed  # set exactly for MANAGED
    if parsed is not None:
        if parsed.namespace == namespace:
            return ClientOrderIdOwnership.OURS
        return ClientOrderIdOwnership.OTHER
    return _OWNERSHIP[result.format]


_OWNERSHIP: Final = {
    ClientOrderIdFormat.ABSENT: ClientOrderIdOwnership.ABSENT,
    ClientOrderIdFormat.UNMANAGED: ClientOrderIdOwnership.OTHER,
    ClientOrderIdFormat.MALFORMED_MANAGED: ClientOrderIdOwnership.MALFORMED_MANAGED,
    ClientOrderIdFormat.UNSUPPORTED_MANAGED_VERSION: (
        ClientOrderIdOwnership.UNSUPPORTED_MANAGED_VERSION
    ),
}


def default_order_token() -> str:
    """A new unpredictable order token (``secrets``, OS randomness)."""
    return secrets.token_hex(DEFAULT_TOKEN_BYTES)


class NamespacedClientOrderIdGenerator:
    """Managed client order ids of one injected namespace (satisfies the
    placement coordinator's ``ClientOrderIdGenerator``).

    The namespace is given, never created here; every id takes exactly one new
    token from ``token_source`` (default: ``default_order_token``). The intent
    is not encoded (``intent_id`` is never a token), so a new intent always gets
    a new id. Token uniqueness is the token source's responsibility.
    """

    __slots__ = ("_namespace", "_token_source")

    def __init__(
        self,
        *,
        namespace: ClientOrderNamespace,
        token_source: Callable[[], str] = default_order_token,
    ) -> None:
        if type(namespace) is not ClientOrderNamespace:
            raise DomainValidationError("namespace must be a ClientOrderNamespace")
        if not callable(token_source):
            raise DomainValidationError("token_source must be callable")
        self._namespace = namespace
        self._token_source = token_source

    @property
    def namespace(self) -> ClientOrderNamespace:
        return self._namespace

    def next_id(self, *, intent: PlaceOrderIntent) -> str:
        """A new managed id; ``intent`` is accepted for the protocol only."""
        return self._namespace.build(self._token_source())
