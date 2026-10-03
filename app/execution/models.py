"""Execution-local records (not domain models).

``SubmissionOutcome`` classifies the transport result of a placement request;
``SafetyBlockRecord`` is the durable reason of a submission refused by the
local safety gate before any send. ``PositionBaselineRecord`` is the immutable
audit record of an explicitly accepted position baseline.
``ExchangeOrderState`` is a confirmed exchange report about one order, normalized
for the account state (which does not depend on exchange DTOs).

``PlacementRecord`` links one ``PlaceOrderIntent`` to the outcome registered for
it: the Risk decision and, when approved, the ``client_order_id`` of the local
``Order(NEW)`` reservation. It keeps the intent itself (immutable) as the identity
used to detect a conflicting reuse of an ``intent_id``; the domain ``Order`` does
not carry ``intent_id``.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Final

from app.domain.enums import OrderStatus
from app.domain.errors import DomainValidationError
from app.domain.intents import PlaceOrderIntent
from app.domain.order_state import EXCHANGE_REPORTED_STATUSES
from app.domain.validation import (
    require_enum,
    require_non_negative,
    require_positive,
    require_text,
    require_utc,
)
from app.risk.models import RiskDecision, TradingState


class SubmissionOutcome(StrEnum):
    """What the transport outcome of one placement request proves."""

    NOT_SENT = "not_sent"
    """Definitely never reached the exchange -> FAILED."""
    REJECTED = "rejected"
    """The exchange definitively refused it -> REJECTED."""
    AMBIGUOUS = "ambiguous"
    """It may exist on the exchange -> UNKNOWN (stays active)."""


class SubmissionBlockStage(StrEnum):
    """Where the safety gate stopped a submission (both before any send)."""

    BEFORE_WRITE_AHEAD = "before_write_ahead"
    """The order was still NEW: NEW -> FAILED."""
    BEFORE_SEND = "before_send"
    """SUBMITTING was durable, the final pre-send check refused: SUBMITTING -> FAILED."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SafetyBlockRecord:
    """Durable reason of a safety-blocked submission: the order was FAILED by the
    local safety gate and was DEFINITELY NOT SENT (no exchange was called)."""

    client_order_id: str
    effective_state: TradingState
    """The effective trading state that refused the send."""
    stage: SubmissionBlockStage
    blocked_at: datetime
    """Local time of the FAILED transition (the order's new ``updated_at``)."""

    def __post_init__(self) -> None:
        require_text(self.client_order_id, "client_order_id")
        require_enum(self.effective_state, TradingState, "effective_state")
        require_enum(self.stage, SubmissionBlockStage, "stage")
        require_utc(self.blocked_at, "blocked_at")


AUDIT_REASON_MAX_LENGTH: Final = 500
BASELINE_ID_MAX_LENGTH: Final = 64


def require_audit_reason(value: object, field: str = "reason") -> str:
    """A canonical audit reason, kept exactly as given (never normalized).

    A ``str`` of 1..``AUDIT_REASON_MAX_LENGTH`` characters without surrounding
    whitespace and without any control / format / unassigned character (Unicode
    category ``C*``): a non-canonical reason is rejected, not trimmed. The
    reason is free text that is persisted and shown: it must never contain a
    secret (nothing here can detect one).
    """
    if type(value) is not str:
        raise DomainValidationError(f"{field} must be a str, got {type(value).__name__}")
    if not value or value != value.strip():
        raise DomainValidationError(
            f"{field} must be non-empty text without surrounding whitespace"
        )
    if len(value) > AUDIT_REASON_MAX_LENGTH:
        raise DomainValidationError(
            f"{field} must be at most {AUDIT_REASON_MAX_LENGTH} characters, got {len(value)}"
        )
    if any(unicodedata.category(char).startswith("C") for char in value):
        raise DomainValidationError(f"{field} must not contain control or format characters")
    return value


def require_baseline_id(value: object, field: str = "baseline_id") -> str:
    """A baseline id: canonical text of at most ``BASELINE_ID_MAX_LENGTH``
    printable ASCII characters (no spaces)."""
    if type(value) is not str:
        raise DomainValidationError(f"{field} must be a str, got {type(value).__name__}")
    if not 1 <= len(value) <= BASELINE_ID_MAX_LENGTH or not all(
        "!" <= char <= "~" for char in value
    ):
        raise DomainValidationError(
            f"{field} must be 1-{BASELINE_ID_MAX_LENGTH} printable ASCII characters "
            f"without spaces, got {value!r}"
        )
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class PositionBaselineRecord:
    """Immutable audit record of an explicit position baseline acceptance.

    It records that the operator / recovery layer explicitly accepted the
    exchange-observed runtime quantity ``qty`` of ``symbol`` as the durable
    position baseline at ``accepted_at``; it became durable at
    ``account_revision`` together with the known durable position. It is NOT a
    lasting proof of the exchange position: every restart still needs an
    exchange reconciliation. Append-only: a later acceptance adds a new record.
    """

    baseline_id: str
    symbol: str
    qty: Decimal
    """The accepted signed quantity (exact, finite; 0 is a known flat baseline)."""
    reason: str
    accepted_at: datetime
    account_revision: int
    """The revision at which the acceptance became durable (>= 1)."""

    def __post_init__(self) -> None:
        require_baseline_id(self.baseline_id)
        require_text(self.symbol, "symbol")
        if type(self.qty) is not Decimal or not self.qty.is_finite():
            raise DomainValidationError(f"qty must be an exact, finite Decimal, got {self.qty!r}")
        require_audit_reason(self.reason)
        require_utc(self.accepted_at, "accepted_at")
        if type(self.account_revision) is not int or self.account_revision < 1:
            raise DomainValidationError(
                f"account_revision must be an int >= 1, got {self.account_revision!r}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class PlacementRecord:
    """The registered result of one intent; one record per ``intent_id``."""

    intent: PlaceOrderIntent
    """The registered intent; a later registration of the same ``intent_id`` is a
    replay only when it is equal to this one (field by field)."""
    decision: RiskDecision
    """The Risk decision registered for the intent (approved or rejected)."""
    client_order_id: str | None
    """The reserved order's id when approved; None when rejected."""

    def __post_init__(self) -> None:
        if type(self.intent) is not PlaceOrderIntent:
            raise DomainValidationError("intent must be a PlaceOrderIntent")
        if type(self.decision) is not RiskDecision:
            raise DomainValidationError("decision must be a RiskDecision")
        if self.decision.intent_id != self.intent.intent_id:
            raise DomainValidationError(
                f"decision intent_id {self.decision.intent_id} differs from "
                f"intent_id {self.intent.intent_id}"
            )
        if self.decision.approved:
            require_text(self.client_order_id, "client_order_id")
        elif self.client_order_id is not None:
            raise DomainValidationError("a rejected placement has no client_order_id")

    @property
    def intent_id(self) -> str:
        return self.intent.intent_id

    @property
    def approved(self) -> bool:
        return self.decision.approved


@dataclass(frozen=True, slots=True, kw_only=True)
class ExchangeOrderState:
    """What the exchange confirms about one order: identity, status, cumulative
    execution and its time. Consistency with the local order (quantity, fill
    invariants) is checked when it is applied."""

    client_order_id: str
    exchange_order_id: str | None
    status: OrderStatus
    filled_qty: Decimal
    """Cumulative executed quantity on the exchange."""
    avg_fill_price: Decimal | None
    """None exactly when nothing is filled."""
    exchange_ts: datetime

    def __post_init__(self) -> None:
        require_text(self.client_order_id, "client_order_id")
        if self.exchange_order_id is not None:
            require_text(self.exchange_order_id, "exchange_order_id")
        status = require_enum(self.status, OrderStatus, "status")
        if status not in EXCHANGE_REPORTED_STATUSES:
            raise DomainValidationError(f"status {status.value} is not an exchange-reported status")
        filled = require_non_negative(self.filled_qty, "filled_qty")
        if filled == 0:
            if self.avg_fill_price is not None:
                raise DomainValidationError("avg_fill_price must be None without fills")
        else:
            require_positive(self.avg_fill_price, "avg_fill_price")
        require_utc(self.exchange_ts, "exchange_ts")
