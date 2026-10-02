"""Execution-local records (not domain models).

``SubmissionOutcome`` classifies the transport result of a placement request.

``PlacementRecord`` links one ``PlaceOrderIntent`` to the outcome registered for
it: the Risk decision and, when approved, the ``client_order_id`` of the local
``Order(NEW)`` reservation. It keeps the intent itself (immutable) as the identity
used to detect a conflicting reuse of an ``intent_id``; the domain ``Order`` does
not carry ``intent_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.domain.errors import DomainValidationError
from app.domain.intents import PlaceOrderIntent
from app.domain.validation import require_text
from app.risk.models import RiskDecision


class SubmissionOutcome(StrEnum):
    """What the transport outcome of one placement request proves."""

    NOT_SENT = "not_sent"
    """Definitely never reached the exchange -> FAILED."""
    REJECTED = "rejected"
    """The exchange definitively refused it -> REJECTED."""
    AMBIGUOUS = "ambiguous"
    """It may exist on the exchange -> UNKNOWN (stays active)."""


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
