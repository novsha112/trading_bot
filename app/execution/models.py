"""Execution-local records (not domain models).

``SubmissionOutcome`` classifies the transport result of a placement request;
``SafetyBlockRecord`` is the durable reason of a submission refused by the
local safety gate before any send.
``ExchangeOrderState`` is a confirmed exchange report about one order, normalized
for the account state (which does not depend on exchange DTOs).

``PlacementRecord`` links one ``PlaceOrderIntent`` to the outcome registered for
it: the Risk decision and, when approved, the ``client_order_id`` of the local
``Order(NEW)`` reservation. It keeps the intent itself (immutable) as the identity
used to detect a conflicting reuse of an ``intent_id``; the domain ``Order`` does
not carry ``intent_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Final

from app.domain.enums import OrderStatus
from app.domain.errors import DomainValidationError
from app.domain.intents import PlaceOrderIntent
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


# Statuses an exchange can actually report about an order. Local lifecycle
# statuses (NEW, SUBMITTING, CANCELING, UNKNOWN) and FAILED ("never sent") are
# never exchange facts.
EXCHANGE_REPORTED_STATUSES: Final = frozenset(
    {
        OrderStatus.OPEN,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    }
)


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
