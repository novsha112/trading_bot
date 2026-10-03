"""Managed client order id namespace: wire format, parsing, ownership, generator."""

from __future__ import annotations

import ast
import dataclasses
import inspect
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.domain.enums import OrderType, Side, TimeInForce
from app.domain.errors import DomainValidationError
from app.domain.intents import PlaceOrderIntent
from app.execution import client_order_id as module
from app.execution.account_state import InMemoryAccountState
from app.execution.client_order_id import (
    ClientOrderIdFormat,
    ClientOrderIdOwnership,
    ClientOrderNamespace,
    NamespacedClientOrderIdGenerator,
    ParsedClientOrderId,
    classify_client_order_id,
    default_order_token,
    parse_client_order_id,
)
from app.execution.safety import SafetyController
from app.persistence.memory import InMemoryAccountStateStore
from app.risk.models import RiskPolicy, SymbolRiskLimits, TradingState
from app.services.placement import ClientOrderIdGenerator, PlacementCoordinator

OWN = ClientOrderIdOwnership
FMT = ClientOrderIdFormat
NS = ClientOrderNamespace("bot01")
OTHER_NS = ClientOrderNamespace("bot02")
T0 = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)


class SubStr(str):
    __slots__ = ()


def intent(intent_id: str = "i-1") -> PlaceOrderIntent:
    return PlaceOrderIntent(
        intent_id=intent_id,
        strategy_id="grid-1",
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("100"),
        qty=Decimal("1"),
        time_in_force=TimeInForce.GTC,
        reduce_only=False,
        tag=None,
        created_at=T0,
    )


class Tokens:
    """A scripted token source that counts its calls."""

    def __init__(self, *tokens: str) -> None:
        self.tokens = list(tokens)
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return self.tokens.pop(0)


# --- namespace ------------------------------------------------------------------------------


@pytest.mark.parametrize("token", ["a", "bot01", "0", "abcdefghij12"])
def test_valid_namespaces(token: str) -> None:
    namespace = ClientOrderNamespace(token)
    assert namespace.token == token
    with pytest.raises(dataclasses.FrozenInstanceError):
        namespace.token = "x"  # type: ignore[misc]


@pytest.mark.parametrize(
    "token",
    [
        "",
        "Bot01",  # never silently lowercased
        "BOT",
        "bot_01",  # the delimiter
        "bot-01",
        " bot01",
        "bot01 ",
        "bo t",
        "bot\n",
        "bot\x00",
        "b\u043et",  # Cyrillic o
        "bot\u0661",  # Arabic-Indic digit
        "abcdefghij123",  # 13 characters
        "\u00df",
    ],
)
def test_invalid_namespace_tokens_are_rejected(token: str) -> None:
    with pytest.raises(DomainValidationError, match="namespace"):
        ClientOrderNamespace(token)


@pytest.mark.parametrize("token", [1, b"bot", True, None, SubStr("bot")])
def test_namespace_must_be_an_exact_str(token: Any) -> None:
    with pytest.raises(DomainValidationError, match="must be a str"):
        ClientOrderNamespace(token)


def test_build_is_canonical_and_versioned() -> None:
    assert NS.build("a1b2") == "tb1_bot01_a1b2"
    assert NS.build("a1b2").isascii()


@pytest.mark.parametrize(
    "token",
    [
        "",
        "a_b",
        "A1",
        "a-b",
        " a",
        "a ",
        "\x7f",
        "\u0442\u043e\u043a\u0435\u043d",
        "a" * 33,
        "ab\t",
    ],
)
def test_invalid_order_tokens_are_rejected(token: str) -> None:
    with pytest.raises(DomainValidationError, match="order_token"):
        NS.build(token)


@pytest.mark.parametrize("token", [1, b"abc", True, None, SubStr("abc")])
def test_order_token_must_be_an_exact_str(token: Any) -> None:
    with pytest.raises(DomainValidationError, match="must be a str"):
        NS.build(token)


# --- parsing / classification ---------------------------------------------------------------


@pytest.mark.parametrize("token", ["a", "0", "deadbeefcafe0123", "z" * 32])
def test_round_trip(token: str) -> None:
    client_order_id = NS.build(token)

    result = parse_client_order_id(client_order_id)

    assert result.format is FMT.MANAGED
    assert result.parsed == ParsedClientOrderId(version=1, namespace=NS, order_token=token)
    assert classify_client_order_id(client_order_id, namespace=NS) is OWN.OURS
    assert NS.owns(client_order_id)


def test_another_valid_namespace_is_other() -> None:
    client_order_id = NS.build("abc")
    assert classify_client_order_id(client_order_id, namespace=OTHER_NS) is OWN.OTHER
    assert not OTHER_NS.owns(client_order_id)
    assert parse_client_order_id(client_order_id).format is FMT.MANAGED


@pytest.mark.parametrize(
    "value",
    [
        "tb1",  # nothing after the prefix
        "tb1_",
        "tb1_bot01",  # missing token
        "tb1_bot01_",
        "tb1__abc",  # missing namespace
        "tb1_bot01_abc_def",  # extra segment
        "tb1_bot01__abc",
        "tb1_bot-01_abc",  # invalid namespace character
        "tb1_bot01_ab-c",  # invalid token character
        "tb1_BOT01_abc",  # uppercase namespace
        "tb1_bot01_ABC",  # uppercase token
        "TB1_bot01_abc",  # uppercase prefix
        "Tb1_bot01_abc",
        "tb01_bot01_abc",  # non-canonical version
        " tb1_bot01_abc",  # leading whitespace
        "tb1_bot01_abc ",  # trailing whitespace
        "\ttb1_bot01_abc",
        "\x00tb1_bot01_abc",  # control character
        "tb1_bot01_ab\nc",
        "tb1_b\u043et01_abc",  # Cyrillic o in the namespace
        "tb1_bot01_\u0430bc",  # Cyrillic a in the token
        "tb1_bot01_" + "a" * 33,  # token too long
        "tb1_abcdefghij123_abc",  # namespace too long
    ],
)
def test_damaged_managed_ids_are_malformed_never_other(value: str) -> None:
    assert parse_client_order_id(value).format is FMT.MALFORMED_MANAGED
    assert parse_client_order_id(value).parsed is None
    assert classify_client_order_id(value, namespace=NS) is OWN.MALFORMED_MANAGED
    assert not NS.owns(value)


@pytest.mark.parametrize(
    "value", ["tb2_bot01_abc", "tb2", "tb9_x", "tb10_bot01_abc", "tb2_anything at all"]
)
def test_other_versions_are_unsupported_never_other(value: str) -> None:
    assert parse_client_order_id(value).format is FMT.UNSUPPORTED_MANAGED_VERSION
    assert classify_client_order_id(value, namespace=NS) is OWN.UNSUPPORTED_MANAGED_VERSION


@pytest.mark.parametrize(
    "value",
    [
        "abc",
        "external-123",
        "manual",
        "tb",
        "tbx_1",
        "tb_bot01_abc",
        "tb12x",
        "atb1_bot01_abc",
        "1234567890",
        "SIM-0000000001",
        "",
        "\u0442\u04311_bot01_abc",  # Cyrillic lookalike prefix: not our family
    ],
)
def test_unrelated_strings_are_other(value: str) -> None:
    assert parse_client_order_id(value).format is FMT.UNMANAGED
    assert classify_client_order_id(value, namespace=NS) is OWN.OTHER


def test_absent_client_id_is_absent_not_an_error() -> None:
    assert parse_client_order_id(None).format is FMT.ABSENT
    assert classify_client_order_id(None, namespace=NS) is OWN.ABSENT
    assert not NS.owns(None)


@pytest.mark.parametrize("value", [1, b"tb1_bot01_abc", True, SubStr("tb1_bot01_abc")])
def test_non_str_input_is_a_programmer_error(value: Any) -> None:
    with pytest.raises(DomainValidationError, match="client_order_id"):
        parse_client_order_id(value)


def test_classification_requires_a_namespace() -> None:
    with pytest.raises(DomainValidationError, match="ClientOrderNamespace"):
        classify_client_order_id("abc", namespace="bot01")  # type: ignore[arg-type]


# --- generator ------------------------------------------------------------------------------


def test_generator_uses_the_injected_namespace_and_one_token_per_id() -> None:
    tokens = Tokens("aaa", "bbb")
    generator = NamespacedClientOrderIdGenerator(namespace=NS, token_source=tokens)

    first = generator.next_id(intent=intent("i-1"))
    second = generator.next_id(intent=intent("i-1"))  # same intent: still a new id

    assert (first, second) == ("tb1_bot01_aaa", "tb1_bot01_bbb")
    assert tokens.calls == 2
    assert generator.namespace is NS
    assert "i-1" not in first


def test_generator_satisfies_the_coordinator_protocol() -> None:
    generator: ClientOrderIdGenerator = NamespacedClientOrderIdGenerator(namespace=NS)
    assert NS.owns(generator.next_id(intent=intent()))


def test_restart_with_the_same_namespace_keeps_ownership() -> None:
    before = NamespacedClientOrderIdGenerator(namespace=ClientOrderNamespace("bot01"))
    issued = before.next_id(intent=intent())
    # A "restart": a new generator from the same configured namespace.
    after = NamespacedClientOrderIdGenerator(namespace=ClientOrderNamespace("bot01"))

    assert after.namespace == before.namespace
    assert after.namespace.owns(issued)
    assert not ClientOrderNamespace("bot02").owns(issued)


def test_default_tokens_are_secure_random_hex() -> None:
    tokens = {default_order_token() for _ in range(200)}
    assert len(tokens) == 200
    assert all(len(token) == 16 and int(token, 16) >= 0 for token in tokens)
    source = inspect.getsource(module)
    assert "secrets.token_hex" in source
    for banned in ("import random", "uuid1", "time.time", "datetime.now", "hash("):
        assert banned not in source
    ids = {
        NamespacedClientOrderIdGenerator(namespace=NS).next_id(intent=intent()) for _ in range(50)
    }
    assert len(ids) == 50


def test_generator_never_creates_a_namespace() -> None:
    source = inspect.getsource(NamespacedClientOrderIdGenerator)
    assert "ClientOrderNamespace(" not in source
    with pytest.raises(DomainValidationError, match="namespace"):
        NamespacedClientOrderIdGenerator(namespace="bot01")  # type: ignore[arg-type]
    with pytest.raises(DomainValidationError, match="token_source"):
        NamespacedClientOrderIdGenerator(namespace=NS, token_source="x")  # type: ignore[arg-type]


def test_an_invalid_token_from_the_source_is_rejected() -> None:
    generator = NamespacedClientOrderIdGenerator(namespace=NS, token_source=lambda: "a_b")
    with pytest.raises(DomainValidationError, match="order_token"):
        generator.next_id(intent=intent())


@pytest.mark.asyncio
async def test_coordinator_reserves_namespaced_ids() -> None:
    account = InMemoryAccountState(account_scope_id="acct-1", store=InMemoryAccountStateStore())
    async with account.account_lock() as locked:
        await locked.set_position_qty("BTCUSDT", Decimal("0"))
    safety = SafetyController(account_state=account)
    safety.mark_hydrated()
    safety.mark_exchange_reconciled()
    safety.request_state(TradingState.RUNNING)

    class Clock:
        def now(self) -> datetime:
            return T0

    coordinator = PlacementCoordinator(
        account_state=account,
        safety=safety,
        policy=RiskPolicy(
            policy_id="p",
            max_open_orders=None,
            symbols={
                "BTCUSDT": SymbolRiskLimits(
                    max_order_qty=None, max_order_notional=None, max_position_qty=Decimal("10")
                )
            },
        ),
        clock=Clock(),
        client_order_id_generator=NamespacedClientOrderIdGenerator(
            namespace=NS, token_source=Tokens("aa", "bb")
        ),
    )

    first = await coordinator.place(intent=intent("i-1"))
    second = await coordinator.place(intent=intent("i-2"))
    replay = await coordinator.place(intent=intent("i-1"))

    assert (first.client_order_id, second.client_order_id) == ("tb1_bot01_aa", "tb1_bot01_bb")
    assert replay == first


# --- boundaries -----------------------------------------------------------------------------


def test_module_is_pure_and_exchange_neutral() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert {name for name in imported if name.startswith("app.")} <= {
        "app.domain.errors",
        "app.domain.intents",
    }
    assert "bybit" not in source.lower()
    assert "36" not in source
