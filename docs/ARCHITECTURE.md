# Архітектура торгового бота

> Статус: чернетка v0.1 (2026-09-30). Реалізації ще немає.
> Основні правила проєкту — у `CLAUDE.md`. Якщо цей документ суперечить `CLAUDE.md`, пріоритет має `CLAUDE.md`, а документ треба виправити.
> План реалізації — у `docs/ROADMAP.md`.

---

## 0. Обсяг v1 і базові принципи

### Обсяг першої версії

| Параметр | Значення v1 |
|---|---|
| Біржа | Bybit, API v5 |
| Ринок | USDT perpetual futures (у Bybit v5 — `category=linear`, перевірити) |
| Стратегія | Grid: Long / Short / Neutral |
| Процес | один Python-процес, asyncio |
| Акаунт | один виділений субакаунт Bybit тільки для бота |
| Інстанси стратегії | один інстанс Grid на один символ |

Моделі та інтерфейси одразу містять `strategy_id`, `symbol` і `exchange`, щоб пізніше додати кілька стратегій, символів і бірж без переписування ядра. Підтримувати кілька інстансів у v1 не будемо.

### Принципи

1. **Modular monolith.** Один процес, чіткі модулі, залежності тільки «зверху вниз». Без мікросервісів, брокерів повідомлень і Redis на старті.
2. **Functional core, imperative shell.** Стратегія, Risk Manager, state machine ордерів, портфель і grid-математика — синхронний детермінований код без I/O. `async` тільки на межах: біржа, WebSocket, БД, нотифікації. Тоді один і той самий код легко тестувати і повторно використовувати в backtest.
3. **Single writer.** Усі зміни торгового стану проходять через одну чергу подій і обробляються послідовно одним циклом (`TradingEngine`). Стратегія, ризик і портфель не змінюються паралельно з різних задач, тому не потрібні локи і немає гонок.
4. **Біржа — джерело правди.** Локальний стан — це робочий кеш і журнал, а не істина (див. розділи 12–13).
5. **Гроші в `Decimal`.** Ціни, кількості, суми, комісії — тільки `Decimal`. `float` заборонений у торговій логіці; у метриках бектесту (Sharpe тощо) допустимий.
6. **Час тільки через `Clock`.** Усі часові мітки в UTC. Кожна ринкова подія несе `exchange_ts` (час біржі) і `received_ts` (час отримання). Логіка ніколи не викликає `datetime.now()` напряму, лише `Clock.now()`.
7. **Нічого неявного.** Поведінка при виході ціни з діапазону, дії kill switch і реакція на розбіжності задаються явно в конфігурації, без «розумних дефолтів», які можуть торгувати.

---

## 1. Загальна архітектура

```text
┌──────────────────────────────── Джерела подій ────────────────────────────────┐
│  MarketDataFeed (public WS / REST / replay)    PrivateStream (orders, fills,  │
│                                                positions, wallet)             │
└───────────────┬──────────────────────────────────────────┬────────────────────┘
                │ MarketEvent                              │ OrderUpdate / Fill
                ▼                                          ▼
        ┌──────────────────────── Event Queue (asyncio.Queue) ─────────────────┐
        └──────────────────────────────────┬───────────────────────────────────┘
                                           ▼
                                ┌────────────────────┐
                                │   TradingEngine    │  single writer loop
                                └─────────┬──────────┘
          ┌───────────────────────────────┼───────────────────────────────┐
          ▼                               ▼                               ▼
   Portfolio.apply()             Strategy.on_event()             OrderManager.apply()
   (positions, PnL)                      │
                                         │ list[Intent]
                                         ▼
                                 InstrumentPreflight ──► reject → RiskEvent
                                         │ (tick/step/min/max/min_notional)
                                         ▼
                                 RiskManager.evaluate()  ──► reject → RiskEvent
                                         │ approved
                                         ▼
                                 ExecutionEngine.submit()
                                         │ (async I/O task)
                                         ▼
                                 ExchangeAdapter  ──►  Bybit | SimulatedExchange
                                         │
                                         └── результат повертається як подія в Event Queue

Паралельно: Persistence (журнал усього), Monitoring (health, метрики),
Notifications (неблокувальні), KillSwitch (незалежний шлях до адаптера).
```

### Модулі та залежності

```text
app/
├── domain/          моделі, enums, state machine ордера, Clock protocol, помилки. Без залежностей.
├── config/          Pydantic Settings + YAML. Залежить від domain.
├── exchanges/       protocols адаптерів + bybit/, simulated/, fake/. Залежить від domain.
├── market_data/     фіди, stale-детектор, replay. Залежить від domain, exchanges (protocol).
├── strategies/      base.py + grid/. Залежить ТІЛЬКИ від domain.
├── risk/            RiskManager, ліміти, KillSwitch, TradingState, оцінка ліквідації. domain, portfolio.
├── execution/       OrderManager, ExecutionEngine, стан акаунта (ордери + позиції + revision під одним lock). domain, exchanges, risk (лише моделі), portfolio (правила позиції).
├── portfolio/       правила позицій, баланси, PnL з fills (без власного стану й lock). domain.
├── persistence/     БД, таблиці, репозиторії, міграції. domain.
├── backtesting/     runner, завантаження історії, метрики, звіт.
├── paper_trading/   зв'язка live market data + SimulatedExchange, збереження стану симуляції.
├── monitoring/      health checks, метрики, статус.
├── notifications/   Notifier protocol, Telegram, неблокувальний dispatcher.
└── services/        TradingEngine, bootstrap (збирання залежностей під режим),
                     recovery, reconciliation, preflight.
```

Жорсткі правила залежностей:
- `strategies/` не імпортує `exchanges/`, `execution/`, `persistence/`, `config/`. Параметри стратегія отримує як готову доменну модель.
- `risk/` не імпортує `execution/` чи `services/`; `execution/` може зберігати результати Risk (`RiskDecision`), але не викликає `evaluate` — їх поєднує coordinator у `services/`. `services/` поки може імпортувати лише `domain`, `risk`, `execution` (без адаптерів бірж, симулятора, persistence і сторонніх бібліотек); `execution/` не імпортує `services/`.
- `domain/` не імпортує нічого з `app/`, крім власних модулів `app.domain`, і з зовнішнього світу — лише стандартну бібліотеку Python (без Pydantic, structlog, SDK бірж, HTTP/WS, БД).
- Тільки `execution/` і `KillSwitch` викликають торгові методи адаптера.
- Тільки `services/bootstrap` знає, який режим активний і які реалізації підставити.

Правила імпортів між пакетами і правило «`domain` — тільки stdlib» перевіряє архітектурний тест `tests/unit/test_architecture_imports.py` (аналіз `ast`, без виконання модулів; нерозв'язні relative imports вважаються порушенням).

---

## 2. Відповідальність модулів

| Модуль | Відповідає за | НЕ відповідає за |
|---|---|---|
| `domain` | Типи даних, інваріанти, таблицю переходів статусів ордера, округлення до tick/step | I/O, конфігурацію |
| `config` | Завантаження та валідацію `.env` + YAML, режими, профілі | Бізнес-логіку |
| `exchanges` | Перетворення доменних запитів в API біржі й назад; auth; rate limit; нормалізацію помилок | Рішення, що торгувати |
| `market_data` | Підписки, reconnect, heartbeat, stale-детекцію, валідацію послідовності | Торгові рішення |
| `strategies` | Генерацію `Intent` на основі ринку і власного стану | Виклики API, ризик-ліміти акаунта |
| `risk` | Затвердження / відхилення intents (V1 — без зміни intent); ліміти; TradingState; KillSwitch | Генерацію торгових ідей; перевірки `InstrumentSpec` (окремий preflight) |
| `execution` | Життєвий цикл ордерів, ідемпотентність, submit/cancel, обробку невідомих результатів | Вибір ціни чи обсягу |
| `portfolio` | Позиції, баланси, realized/unrealized PnL, комісії, funding | Відправку ордерів |
| `persistence` | Збереження та читання стану і журналу подій | Бізнес-рішення |
| `backtesting` | Прогін історії через ті самі Strategy/Risk/Execution; метрики | Окрему «бектестову» стратегію |
| `paper_trading` | Live-дані + симуляція виконання | Окрему логіку стратегії |
| `monitoring` | Health, метрики, статус | Торгові рішення |
| `notifications` | Доставку повідомлень, яка ніколи не блокує торгівлю | Будь-що критичне для торгівлі |
| `services` | Оркестрацію, запуск, відновлення, reconciliation, preflight | Деталі API біржі |

---

## 3. Потік даних

### 3.1 Нормальний цикл

```text
1. MarketDataFeed отримує тікер / стакан / трейд  → MarketEvent → Event Queue
2. TradingEngine бере подію:
   a. MarketDataCache оновлюється (last, mark, bid/ask, data_age)
   b. RiskManager перевіряє глобальні умови (stale data, TradingState)
   c. Strategy.on_market(event, ctx) → list[Intent]
3. Для кожного Intent:
   a. InstrumentPreflight: tick / step / min / max qty / min notional за InstrumentSpec
      (невідповідний intent відхиляється до Risk)
   b. RiskManager.evaluate(intent, snapshot, policy) → RiskDecision (розділ 9.0)
   c. approved → ExecutionEngine.submit(intent) без змін intent
      rejected → RiskEvent (журнал, лог; сповіщення, якщо критично)
4. ExecutionEngine:
   a. створює Order(NEW, client_order_id), персистить
   b. переводить у SUBMITTING, персистить (write-ahead)
   c. запускає async-задачу trading_client.place_order(OrderRequest)
   d. результат задачі (ack / reject / timeout) → подія в Event Queue
5. PrivateStream приносить OrderUpdate і Fill → Event Queue
6. TradingEngine: OrderManager.apply(update), Portfolio.apply(fill),
   Strategy.on_fill(fill) → нові Intents (наприклад, парний ордер сітки)
```

### 3.2 Signal проти Order Intent

`CLAUDE.md` допускає, що стратегія генерує або Signal, або Order Intent. Для Grid окремий Signal нічого не додає: стратегія одразу знає, які лімітні ордери їй потрібні. Тому в v1 стратегія повертає **intents**:

- `PlaceOrderIntent` — розмістити ордер;
- `CancelOrderIntent` — скасувати ордер за `client_order_id`.

Модель `Signal` (напрямок і сила сигналу без конкретних ордерів) додамо, коли з'являться directional-стратегії (Trend, Breakout). Між Signal і Intent тоді стоятиме position sizer. Зараз це було б зайвою абстракцією.

### 3.3 Хто і що може змінювати в Intent

| Хто | Може | Не може |
|---|---|---|
| Strategy | Задати ціну, кількість, сторону, `reduce_only`, `time_in_force` (включно з `POST_ONLY`), тег рівня сітки | Звертатися до біржі |
| RiskManager | Затвердити або відхилити (V1) | Змінювати intent: кількість (ні збільшити, ні зменшити), ціну, сторону, `reduce_only`. Обрізання розміру — можлива майбутня окрема функція |
| ExecutionEngine | Нічого не змінює в суті ордера | Округлювати, змінювати ціну чи кількість |

**Округлення до `tick_size` / `qty_step`** робить стратегія через доменні хелпери `InstrumentSpec`. Рівні сітки мусять бути детермінованими і однаковими після рестарту. Окремий instrument preflight (не RiskManager) тільки **перевіряє** відповідність і відхиляє невідповідний intent; RiskManager ці перевірки не дублює. Тихого округлення на нижчих рівнях немає, бо воно ховає помилки розрахунку розміру.

---

## 4. Доменні моделі

Усі моделі — `frozen` dataclasses або Pydantic-моделі. Гроші в `Decimal`, час — aware datetime в UTC.

### Довідкові

| Модель | Ключові поля |
|---|---|
| `InstrumentSpec` | `exchange`, `symbol`, `category`, `base`, `quote`, `settle`, `tick_size`, `qty_step`, `min_qty`, `max_qty`, `min_notional`, `max_leverage`, `funding_interval`, `status` |
| `FeeSchedule` | `maker_rate`, `taker_rate`, `source` (api / config), `fetched_at` |
| `RiskLimitTier` | `max_position_value`, `maintenance_margin_rate`, `max_leverage` (для оцінки ліквідації) |

### Ринкові

| Модель | Ключові поля |
|---|---|
| `Ticker` | `last`, `mark`, `index`, `best_bid`, `best_ask`, `funding_rate`, `next_funding_time`, `exchange_ts`, `received_ts` |
| `OrderBookTop` | `bid`, `bid_qty`, `ask`, `ask_qty`, `update_id`, `seq`, timestamps |
| `PublicTrade` | `price`, `qty`, `side`, `trade_id`, `exchange_ts` |
| `Kline` | `open_time`, `close_time`, OHLC, `volume`, `is_closed` |
| `FundingRate` | `rate`, `funding_time` |

### Торгові

| Модель | Ключові поля |
|---|---|
| `PlaceOrderIntent` | `intent_id`, `strategy_id`, `symbol`, `side`, `order_type`, `price`, `qty`, `time_in_force` (`GTC` / `IOC` / `FOK` / `POST_ONLY`; окремого `post_only` немає), `reduce_only`, `tag` (наприклад, `grid:L07:buy`), `created_at` |
| `CancelOrderIntent` | `intent_id`, `strategy_id`, `client_order_id`, `reason` |
| `RiskDecision` | `intent_id`, `snapshot_id`, `policy_id`, `approved` (так / ні; `approved ⇔ reasons == ()`), `reasons` (коди `RiskReason`), `exposure` (розклад на зменшення / збільшення і worst-case). V1 без REDUCED / `approved_qty` (розділ 9.0) |
| `Order` | `client_order_id`, `exchange_order_id?`, `strategy_id`, `symbol`, `side`, `order_type`, `price`, `qty`, `time_in_force`, `reduce_only`, `status`, `filled_qty`, `avg_fill_price`, `created_at`, `updated_at`, `last_exchange_update_ts`, `version`. Зв'язок з `intent_id` — відкрите питання Phase 5 |
| `OrderUpdate` | нормалізований звіт біржі про ордер: `client_order_id`, `exchange_order_id`, `status`, `cum_filled_qty`, `avg_price`, `reject_reason`, `exchange_ts` |
| `Fill` | `exec_id` (унікальний), `client_order_id`, `exchange_order_id`, `symbol`, `side`, `price`, `qty`, `fee?`, `fee_asset?`, `is_maker?`, `exchange_ts` (див. нижче про невідомі метадані) |
| `Position` | `symbol`, `side` / знакова `qty`, `entry_price`, `mark_price`, `unrealized_pnl?`, `realized_pnl` (gross), `leverage`, `margin_mode`, `position_margin`, `maintenance_margin`, `liquidation_price?`, `updated_at` (див. нижче про невідомий unrealized PnL) |
| `Balance` | `asset`, `wallet_balance`, `equity`, `available`, `margin_used`, `updated_at` |
| `FundingPayment` | `symbol`, `amount` (знакова), `rate`, `position_qty`, `ts` |

**Невідомі метадані `Fill`.** `Fill` — завжди **підтверджене** виконання: `exec_id`, id ордера, `symbol`, `side`, `price`, `qty`, `exchange_ts` відомі. Nullable-поля не роблять fill умовним, статусу «UNKNOWN fill» немає.

- `fee` / `fee_asset` — пара: обидва `None` (комісія невідома / не моделювалась) або обидва задані. `fee=Decimal("0")` + asset — **відома** нульова комісія; `None` ніколи не означає нуль, sentinel-рядки й `0` для «невідомо» заборонені. Знак: від'ємна комісія — rebate.
- `is_maker=None` — роль ліквідності невідома. Незалежна від даних комісії: допустимі будь-які поєднання.
- Поля обов'язкові при створенні (без defaults): джерело явно заявляє «невідомо».
- Реальні адаптери зберігають фактичні `fee`, `fee_asset` і роль ліквідності, якщо біржа їх надає; `None` — лише для джерел, які цієї інформації справді не мають (наприклад, симулятор без моделі комісій і книги).
- **Правило PnL:** будь-який розрахунок net PnL, сумарних комісій, fee-adjusted return чи прибутковості після комісій **fail closed** або явно позначає результат як incomplete, якщо хоча б один задіяний `Fill.fee is None`. `None` ніколи не трактується як нуль. (Portfolio / PnL ще не реалізовані.)

**Невідомий unrealized PnL у `Position`.**

- `unrealized_pnl: Decimal | None`, без default: джерело явно передає або оцінку, або `None` — «не обчислено / невідомо». `None` ніколи не трактується як нуль і не робить невідомою саму позицію: `qty`, сторона, `entry_price`, `realized_pnl` і `updated_at` лишаються відомими.
- FLAT (`qty == 0`): `entry_price is None` і **рівно** `unrealized_pnl == 0` (без відкритої позиції нереалізований PnL справді нульовий); `None` чи ненульове значення — помилка.
- OPEN: `entry_price > 0`; `unrealized_pnl` — `None` або скінченний `Decimal` будь-якого знаку, включно з нулем (відома нульова оцінка).
- `mark_price` і `unrealized_pnl` незалежні: допустимі оцінка без mark price і mark price без оцінки; domain не вгадує походження даних.
- `realized_pnl` — **gross** реалізований торговий PnL з різниці цін виконання, **до** комісій, funding, процентів і rebates; ці грошові потоки обліковуються окремо майбутніми шарами accounting / portfolio. Відповідність конкретного поля Bybit цій семантиці не стверджується — її перевірять за офіційною документацією перед реалізацією live account adapter.
- **Правило оцінки:** розрахунок, якому потрібна поточна оцінка відкритої позиції (equity, сумарний unrealized, drawdown, поточна прибутковість з урахуванням комісій, ризик-перевірки, що залежать від unrealized / equity), при `unrealized_pnl is None` **fail closed** або явно повертає incomplete / unknown результат. Загального заборонного правила для exposure немає: метрика fail closed лише тоді, коли для її конкретної формули бракує потрібних даних (наприклад, exposure з `qty` і ціни не потребує `unrealized_pnl`). Ці розрахунки ще не реалізовані.
- Simulation API: `get_position(symbol) -> Position | None`, де `None` — по символу ще не було fills / стану позиції; так `updated_at` не вигадується. `updated_at` — час останньої зміни опублікованого стану: пізніший із часу останнього fill і часу mark (див. 5.4, «Mark-to-market»).

### Стратегія Grid

| Модель | Ключові поля |
|---|---|
| `GridConfig` | `symbol`, `mode` (LONG / SHORT / NEUTRAL), `lower`, `upper`, `levels`, `spacing` (ARITHMETIC / GEOMETRIC), `order_qty` або `capital_allocation`, `leverage`, `out_of_range_policy`, `start_policy` |
| `GridLevel` | `index`, `price`, `buy_client_order_id?`, `sell_client_order_id?`, `state` |
| `GridState` | `grid_id`, `config_hash`, `levels[]`, `inventory_qty`, `completed_cycles`, `realized_grid_pnl`, `status` |
| `GridProfitability` | `gross_per_cycle`, `fees_per_cycle`, `est_slippage`, `est_funding_per_day`, `net_per_cycle`, `breakeven_spacing` |

### Системні

| Модель | Призначення |
|---|---|
| `TradingState` | RUNNING / REDUCE_ONLY / PAUSED / HALTED (див. розділ 9) |
| `BotEvent` | Журнал: старт/стоп, reconnect, помилки (категорія `error`), зміни стану |
| `RiskEvent` | Відхилення intent, досягнення ліміту, спрацювання kill switch |
| `ReconciliationEvent` | Знайдена розбіжність, дія, результат |
| `Clock` (protocol) | `now()`, у backtest — симульований |

---

## 5. Exchange Adapter

### 5.1 Контракти (`app/exchanges/`)

Три невеликі async-протоколи замість одного великого клієнта: публічні дані, акаунт і торгівля мають різні вимоги (ключі, rate limit, хто має доступ). Протоколи структурні (`typing.Protocol`), відповідність адаптерів перевіряє mypy.

```text
MarketDataClient   (без ключів)
  get_instrument(symbol) -> InstrumentSpec
  get_ticker(symbol) -> Ticker

AccountClient      (з ключами)
  get_balances() -> tuple[Balance, ...]
  get_positions() -> tuple[Position, ...]

TradingClient      (з ключами; лише execution і KillSwitch)
  place_order(order: OrderRequest) -> OrderAck
  cancel_order(order: OrderRef) -> None
  get_order(order: OrderRef) -> OrderUpdate | None
  get_open_orders(*, symbol) -> tuple[OrderUpdate, ...]
```

Методи додаються разом з першим споживачем. Заплановані (ще не існують): klines і історія funding, risk-limit tiers, fills, ставки комісій, налаштування акаунта й права ключа, `set_leverage`, `cancel_all_orders`, а також стріми (публічний market data і приватний `PrivateStream` з `OrderUpdate` / `Fill` / `Position` / `Balance`).

`CLAUDE.md` пропонує один `ExchangeAdapter`. Розділення на протоколи — уточнення, а не заміна: їх можна зібрати в один фасад.

**Межа і DTO.** Через протоколи проходять лише доменні типи й три DTO межі (`app/exchanges/models.py`). Сирі відповіді, JSON, SDK-об'єкти й біржові назви (статуси, ідентифікатори) лишаються в адаптері, який перекладає їх явно.

| DTO | Поля | Сенс |
|---|---|---|
| `OrderRequest` | `client_order_id`, `symbol`, `side`, `order_type`, `price?`, `qty`, `time_in_force`, `reduce_only` | Запит на розміщення. Будує execution зі збереженого `Order`; життєвого циклу (статус, виконання, версія, часи) в ньому немає. `client_order_id` існує до мережевого виклику — це ключ ідемпотентності; адаптер ніколи не генерує власний |
| `OrderRef` | `symbol`, `client_order_id`, `exchange_order_id?` | Посилання на ордер для скасування й запиту. `client_order_id` обов'язковий (стабільний ідентифікатор для reconciliation); `exchange_order_id` відомий лише після ACK, тож усе працює й без нього |
| `OrderAck` | `client_order_id`, `exchange_order_id`, `exchange_ts?` | Підтвердження прийняття запиту, **не статус**: ордер лишається `SUBMITTING`, доки `OrderUpdate` не підтвердить стан. Запис `exchange_order_id` — оновлення метаданих (Phase 5), не перехід state machine |

Ланцюжок (Phase 5): затверджений intent → збережений `Order` з `client_order_id` → `OrderRequest` → `TradingClient`. `Order` і intents у протоколи не передаються.

**Семантика результатів.** `cancel_order` повертає `None`: прийняття запиту не означає `CANCELED`; результат (у гонці й `FILLED`) приходить через `OrderUpdate`. `get_order` повертає `None` лише тоді, коли біржа підтвердила, що такого ордера немає; порожній `get_open_orders` — підтверджено, що відкритих немає. Неможливість отримати відповідь — завжди виняток.

### 5.2 Помилки (`app/exchanges/errors.py`)

Критично важливо знати, **чи міг запит бути прийнятий біржею**. Категорії результату (`NotSent`, `Rejected`, `DuplicateOrder`, `AmbiguousResult`, а для читання — `Response`) — сиблінги, жодна не є підкласом іншої, тож обробка однієї не може проковтнути іншу:

```text
ExchangeError
├── ExchangeNotSentError
│   └── ExchangeRequestValidationError
├── ExchangeRejectedError
│   └── ExchangeAuthenticationError
├── ExchangeDuplicateOrderError
├── ExchangeAmbiguousResultError
└── ExchangeResponseError
```

| Помилка | Значення | Повтор |
|---|---|---|
| `ExchangeNotSentError` | Відсутність відправки **доведено до входу в мережеву операцію з можливими побічними ефектами**: локальна перевірка, відмова локального rate limiter, очікування з'єднання з пулу. Для мутуючих запитів збій встановлення з'єднання сюди **не** належить (див. нижче); для читання — належить | Може повторити політика викликача |
| `ExchangeRequestValidationError` | Підвид `NotSent`: запит відхилено локально **до будь-якого виклику транспорту**, бо його неможливо виразити в документованому форматі біржі (непідтримана комбінація полів, значення поза документованим синтаксисом) | Повтор того самого запиту безглуздий — виправити запит |
| `ExchangeRejectedError` | Біржа відповіла й однозначно відмовила, нічого не прийнято (зокрема явна відмова через rate limit) | Не повторюється наосліп |
| `ExchangeAuthenticationError` | Біржа відхилила автентифікацію чи права (не для відсутніх локальних ключів — це помилка конфігурації) | Ні; HALTED + сповіщення |
| `ExchangeDuplicateOrderError` | **Окрема категорія, не `Rejected`**: конфлікт ідентичності — `client_order_id` уже належить існуючому біржовому ордеру з іншими умовами. Цей запит нічого не створив, **але ордер з таким id існує**. Generic `except ExchangeRejectedError` його не ловить | Ні; reconciliation існуючого ордера |
| `ExchangeAmbiguousResultError` | Мутуючий запит міг бути прийнятий, підтвердженого результату немає (timeout чи розрив після відправки, або адаптер не може довести, що запит не пішов) | **Ніколи наосліп**: спершу reconciliation |
| `ExchangeResponseError` | **Лише для читання:** запит відправлено (або міг бути), але валідної відповіді немає (timeout, розрив, 5xx, зламаний чи неочікуваний JSON). Читання не має побічних ефектів, тому це не `Ambiguous`. Мутуючі запити його ніколи не кидають | Може повторити політика викликача |

- Timeout ніколи не класифікується як `NotSent`, якщо адаптер не може цього довести. Для мутуючого запиту після початку HTTP-операції будь-який невизначений збій транспорту чи HTTP — `Ambiguous`; доказ «не відправлено» не будується на внутрішній реалізації транспортної бібліотеки.
- Rate limit: відмова локального limiter до мережі → `NotSent`; явна відповідь біржі з відмовою → `Rejected`; timeout чи розрив після можливої відправки → `Ambiguous`. Окремого `RateLimitError` немає.
- Після `ExchangeAmbiguousResultError` на `place_order` ордер стає `UNKNOWN`, і стан встановлюється через `get_order(OrderRef(symbol, client_order_id))`: знайдено → фактичний статус, підтверджено «немає» → `FAILED` (розділ 7.3). Повторна відправка з тим самим `client_order_id` у v1 не робиться.
- Контракт reconciliation для `ExchangeDuplicateOrderError` (симулятор зараз; у майбутньому — реальний адаптер, якщо біржа поверне документовану помилку дубля `orderLinkId`). Викликач:
  1. не повторює `place_order` наосліп;
  2. виконує `get_order(OrderRef(symbol, client_order_id))`;
  3. перевіряє існуючий біржовий ордер (умови, статус, `exchange_order_id`) і застосовує його фактичний стан;
  4. не трактує дубль як доказ стану `REJECTED` / `FAILED`; розбіжність умов — баг локального стану (подія reconciliation, за потреби зупинка торгівлі).
- Повідомлення помилок не містять ключів, підписів, сирих заголовків і тіл відповідей.

### 5.3 Політика retry

Retry-механізму ще немає (наступні кроки Phase 2); правила для нього:

| Операція | Retry | Примітка |
|---|---|---|
| Читання (`get_*`) | так, exponential backoff + jitter, обмежена кількість спроб | безпечно |
| `place_order` | **ні** | `Ambiguous` → `UNKNOWN` → перевірка за `client_order_id`. Ніколи не повторювати сліпо |
| `cancel_order` | так, обмежено, лише після `NotSent` | `Rejected` («order not found» тощо) → перевірити стан через `get_order` |
| `cancel_all_orders` (заплановано) | так, обмежено | потім перевірити `get_open_orders` |
| `set_leverage` (заплановано) | так | ідемпотентна операція |

Rate limiter (token bucket) живе в адаптері, окремо для кожної групи ендпоінтів. Якщо біржа повертає заголовки зі станом лімітів, адаптер їх використовує (формат перевірити в документації).

### 5.4 Реалізації

**Наявна: `app/exchanges/bybit/market_data.py` — `BybitMarketDataClient`** (публічний REST V5, без ключів і підписів, без WebSocket і приватного API). Залежить лише від `httpx` (правило архітектури окремо для `exchanges.bybit`; ядро контрактів лишається на stdlib).

- Ендпоінти (офіційні docs `bybit-exchange/docs`, `docs/v5`): `GET /v5/market/instruments-info` і `GET /v5/market/tickers` з `category=linear&symbol=...`; обгортка `retCode` / `retMsg` / `result` / `retExtInfo` / `time`.
- Підтримується лише USDT linear perpetual: `contractType=LinearPerpetual`, `quoteCoin=settleCoin=USDT`, `status=Trading`, `isPreListing=false` (поле документоване для всіх linear-інструментів і має бути присутнім як bool; відсутність — помилка відповіді, а не `false`). `category=linear` містить і USDC-контракти та ф'ючерси — вони відхиляються (`ExchangeRejectedError`).
- `InstrumentSpec`: `tickSize`, `qtyStep`, `minOrderQty`, `minNotionalValue`; `max_qty` ← `maxOrderQty` (ліміт limit / post-only ордера). `maxMktOrderQty` (окремий ліміт market-ордерів) перевіряється як додатний `Decimal`, але в `InstrumentSpec` не проєктується, доки не з'явиться споживач market-ордерів.
- `Ticker`: `lastPrice` обов'язковий; `markPrice`, `bid1Price`, `ask1Price`, `fundingRate`, `nextFundingTime` — поле має бути присутнє, порожній рядок означає `None`; `nextFundingTime="0"` теж `None` (ніколи не 1970-01-01). У тікера немає власної мітки часу: `exchange_ts` — це `time` обгортки, тобто **серверний час відповіді Bybit, а не час ринкової події**; `received_ts` — з `Clock` після отримання відповіді.
- Числа — лише з JSON-рядків простого десяткового вигляду, напряму в `Decimal` (JSON-числа на їх місці, `NaN`, експонента відхиляються). Мілісекунди — через `utc_from_ms`.
- Відповідь має містити рівно один елемент саме для запитаного символу (без нормалізації регістру); порожній список — `ExchangeRejectedError` («not found»).
- Помилки: з'єднання не встановлено → `NotSent`; `retCode` 10001 (помилка параметрів) і HTTP 400 / 404 → `ExchangeRejectedError`; `retCode` 10000 / 10006 / 10016 / 429, **будь-який невідомий `retCode`**, HTTP 403 / 408 / 429 / 5xx / інші не-2xx, інші транспортні збої, зламаний JSON чи схема → `ExchangeResponseError`. Повідомлення без тіл відповідей; `retMsg` обрізається.
- `httpx.AsyncClient` і `base_url` передаються ззовні (адаптер не закриває клієнт). Константи `BYBIT_TESTNET_REST_URL` / `BYBIT_MAINNET_REST_URL` — в `exchanges/bybit/endpoints.py`; вибір URL за режимом робить майбутній composition layer, у адаптері значення за замовчуванням немає.

**Наявний: `app/exchanges/bybit/private_rest.py` — `BybitPrivateRestTransport`** (лише підпис і транспорт; `TradingClient`, ордери й акаунт ще не реалізовані).

- Підпис за офіційними docs (`docs/v5/guide.mdx`): заголовки `X-BAPI-API-KEY`, `X-BAPI-TIMESTAMP` (UTC ms), `X-BAPI-SIGN`, `X-BAPI-RECV-WINDOW` (за замовчуванням 5000; максимум у docs не задано); рядок для підпису — `timestamp + api_key + recv_window + queryString` (GET) або `+ jsonBodyString` (POST); HMAC_SHA256, lowercase hex.
- Детермінованість: query (порядок викликача, percent-encoding UTF-8 один раз) і JSON-тіло (порядок ключів викликача, компактні роздільники, ASCII, без NaN) формуються один раз і відправляються саме в тому вигляді, в якому підписані. Значення: у query — `str` / `int`, у тілі — JSON-native без `float` / `Decimal` (числа передає адаптер у документованому рядковому вигляді).
- Секрети: `BybitCredentials` (власна обгортка без pydantic, `repr` маскований) будує composition із `EnvSettings`; секрет лише підписує і ніколи не відправляється; `retMsg` у помилках очищується від ключа й секрету; тіла відповідей у помилки не потрапляють; транспорт нічого не логує.
- Час — лише з injected `Clock` (UTC → цілі мілісекунди, без `float`).
- Шляхи — лише відносні `/v5/...` (без `..`, `//`, `?`, `#`, `%`), тож ключ не може піти на інший host; `base_url` перевіряється як у публічному адаптері; `httpx.AsyncClient` — ззовні, не закривається.
- API: `get(path, params=...)` — лише читання, ніколи не `Ambiguous`; `post_mutating(path, body=...)` — для мутуючих запитів. Обидва повертають `BybitResponse(result, ret_ext_info, server_time)`.
- Класифікація мутуючого POST (пріоритет — не допустити дубль ордера, а не зекономити reconciliation):
  - `NotSent` — лише якщо відсутність відправки доведено до входу в HTTP-операцію: локальна перевірка шляху й тіла, серіалізація, `httpx.PoolTimeout` (за публічною семантикою httpx — очікування з'єднання з пулу).
  - Будь-який інший збій транспорту (`ConnectError`, `ConnectTimeout`, read / write / protocol / proxy, невідомі) і **будь-який не-2xx HTTP-статус** (400, 401, 403, 404, 408, 429, 5xx, 3xx…) → `Ambiguous`: HTTP-помилка не є документованим бізнес-результатом Bybit.
  - HTTP 2xx з валідною обгорткою: `retCode` 10003 / 10004 / 10005 / 10007 / 10010 / 33004 → `ExchangeAuthenticationError`; 10001 (помилка параметрів) і 10002 (час поза `recv_window`, захист від replay) → `ExchangeRejectedError`; 10000 / 10006 / 10016 / 429 і невідомі коди → `Ambiguous`; зламаний JSON чи обгортка → `Ambiguous`.
  - Після `Ambiguous` — reconciliation за `client_order_id`, ніколи сліпий повтор.
- Класифікація GET (лише читання, без побічних ефектів, ніколи `Ambiguous`): збій з'єднання чи пулу → `NotSent`; HTTP 401 і auth-коди → `ExchangeAuthenticationError`; HTTP 400 / 404 і 10001 / 10002 → `ExchangeRejectedError`; усе інше → `ExchangeResponseError`.

**Наявний: `app/exchanges/bybit/order_mapping.py` — `map_order_request(order: OrderRequest) -> dict[str, JsonValue]`** — чистий mapping у тіло `POST /v5/order/create` (офіційні docs, гілка `master`: `docs/v5/order/create-order.mdx`, `docs/v5/enum.mdx`). Без транспорту, мережі, логування й `httpx`; `TradingClient` / place / cancel / get ще не реалізовані.

- Обсяг: USDT linear perpetual, one-way. Поля й порядок: `category="linear"`, `symbol`, `side` (`Buy`/`Sell`), `orderType` (`Limit`/`Market`), `qty`, `price` (лише для `Limit`), `timeInForce`, `reduceOnly` (JSON bool), `orderLinkId` (= `client_order_id`). Не відправляються: `positionIdx` (обов'язковий лише в hedge mode), TP/SL, trigger, SMP, MMP, slippage, `orderFilter`, broker, RPI.
- `timeInForce`: `GTC` / `IOC` / `FOK` / `PostOnly` для `Limit` (завжди явно, без покладання на дефолт `GTC`). Для `Market` docs кажуть «Market order will always use IOC» — тому лише `IOC` (відправляється явно); `Market` + `GTC` / `FOK` / `PostOnly` неможливо виразити без мовчазної заміни біржею → `ExchangeRequestValidationError`. Bybit виконує market як IOC-limit із власним порогом slippage (поле `slippageTolerance` не використовуємо).
- Decimal → канонічний рядок у простій нотації без контексту `decimal`: без експоненти, без `float`, без округлення, без хвостових нулів дробової частини (`1.500` → `"1.5"`, `1E+8` → `"100000000"`, `1E-8` → `"0.00000001"`); рівні значення дають однаковий текст. Лише точний тип `Decimal`, скінченний, `> 0`. Обмеження біржі й обмеження реалізації розділені: `MAX_DECIMAL_TEXT_LENGTH = 1024` — **внутрішня межа безпеки ресурсів, а не ліміт Bybit API** (Bybit довжину не документує); перевіряється за `Decimal.as_tuple()` до побудови рядка, тож патологічні порядки (`1E+999999999`) відхиляються без гігабайтної алокації з повідомленням «decimal representation exceeds local safety limit». Вирівнювання до `tickSize` / `qtyStep` — pre-trade перевірка, не mapping.
- `orderLinkId`: `[A-Za-z0-9_-]{1,36}` за docs; ніколи не обрізається й не нормалізується. `symbol`: docs — лише «Symbol name, like `BTCUSDT`, uppercase only», набір символів не задано; тому перевіряється тільки документована властивість: непорожній, без пробілів по краях, `symbol == symbol.upper()` (lowercase → помилка, без нормалізації). Пунктуацію mapper не відхиляє. Чи є символ саме USDT linear perpetual, mapping не знає — це гарантує `InstrumentSpec` з публічного адаптера на етапі pre-trade.
- Поля `OrderRequest` перевіряються повторно (enum — саме член enum, а не рівний йому рядок `StrEnum`; `reduceOnly` — саме `bool`), бо mapping — остання межа перед підписом. Кожен виклик повертає новий `dict`, яким володіє викликач; транспорт одразу серіалізує й підписує його. Frozen DTO не вводився: незмінність після підпису забезпечує транспорт (тіло серіалізується один раз).
- `JsonValue` винесено в `exchanges/bybit/types.py`, тож mapping не залежить від `private_rest`; тест архітектури має для `order_mapping` / `types` окремий allowlist (лише `app.domain`, `app.exchanges.models`, `app.exchanges.errors`, `app.exchanges.bybit.types`; без third-party).

**Наявний: `app/exchanges/simulated.py` — `SimulatedExchange(*, clock: Clock, instruments: tuple[InstrumentSpec, ...] = ())`** — детермінований exchange-neutral `TradingClient` без мережі, ключів і Bybit. Є авторитетним біржовим станом свого екземпляра; викликачі звіряються з ним так само, як з реальною біржею.

- Внутрішній стан — приватний frozen-запис `_SimulatedOrder` (прийнятий `OrderRequest`, `exchange_order_id`, біржовий статус, `created_at` / `updated_at`), а не доменний `Order`: локальний lifecycle (`SUBMITTING`, `UNKNOWN`, `strategy_id`, `version`) належить execution-шару, біржа його не знає.
- `exchange_order_id` — непрозорий рядок із монотонної послідовності екземпляра (`SIM-0000000001`, …); номер видається лише прийнятим ордерам. Ширина не є частиною контракту. Жодних UUID і випадковості: однаковий сценарій → однаковий результат.
- Реєстр інструментів: приватна копія `symbol → InstrumentSpec`, незмінна протягом життя симулятора (оновлення / делістинг метаданих — окремий майбутній дизайн). Дубль symbol → `ValueError` у конструкторі (як інші помилки конфігурації конструкторів; ієрархія `ExchangeError` описує результати запитів). Невідомий symbol, зокрема з порожнім реєстром, → `ExchangeRejectedError` (fail closed).
- Pre-trade перевірки нового LIMIT-ордера (доменні helper-и `is_price_aligned`, `is_qty_aligned`, `meets_min_qty`, `meets_min_notional` — точні, у приватному `Context`, незалежні від глобального): ціна кратна `tick_size`, qty кратна `qty_step`, `min_qty <= qty <= max_qty`, точний `price * qty >= min_notional` (номінал, без комісій, плеча й маржі). Нічого не виправляється автоматично. Порушення → `ExchangeRejectedError` (запит коректний, біржа його розглянула й відхилила, ордера немає): без id, без читання годинника, без збереження. `ExchangeRequestValidationError` лишається для некоректних аргументів API.
- Порядок: спершу існуючий `client_order_id` (ідентичний повтор → оригінальний `OrderAck`, інші умови → `ExchangeDuplicateOrderError`), і лише для нового id — перевірки інструмента. Відхилений ордер не створено, тож його `client_order_id` можна використати знову коректним запитом.
- `place_order`: LIMIT GTC / POST_ONLY → OPEN. Fail closed (`ExchangeRejectedError`, нічого не створено): MARKET (немає моделі ціни виконання), LIMIT IOC / FOK (ніколи не лежать у книзі — результат залежить від matching).
- **Reduce-only** (LIMIT GTC / POST_ONLY) — детермінована політика симулятора, **не** твердження про точну реалізацію reduce-only конкретною біржею:
  - Розміщення: після перевірки існуючого `client_order_id` (ідентичний повтор → оригінальний `OrderAck` незалежно від поточної позиції; інші умови → Duplicate) і звичайних pre-trade перевірок інструмента новий reduce-only мусить зменшувати поточну позицію: SELL — long, BUY — short. Немає позиції, FLAT або той самий бік → `ExchangeRejectedError` без наслідків (id, годинник, стан). Розмір ордера **може** перевищувати позицію: обмеження діє під час виконання.
  - Виконання: авторитетна позиція — та, що підготовлена в цьому batch після попередніх fills (`PositionBatch` ledger; основний ledger не змінюється до commit). Кількість fill = `min(залишок ордера, залишок бюджету, reducible)`, далі DOWN до `qty_step`; reducible — розмір long для SELL, short для BUY, інакше 0. Розвороту через нуль не буває.
  - Auto-cancel: перехрещений reduce-only, що вже не може зменшувати (позиція FLAT або з іншого боку — до або після власного fill), стає CANCELED; виконані частини, `cum_filled_qty` і середня ціна зберігаються, окремого fill немає. Частково виконаний ордер, який ще може зменшувати позицію, лишається PARTIALLY_FILLED.
  - Auto-cancel не витрачає `exec_id` і бюджет `available_qty`; наступні ордери batch використовують залишок. Він є зміною стану: годинник читається один раз на batch, у якому є fill **або** auto-cancel (`updated_at` = час batch); batch без змін годинник не читає.
  - Звичайні ордери не обмежуються reducible і можуть відкривати, нарощувати, закривати й розвертати позицію; окремого пріоритету для reduce-only немає — лише порядок `(created_at, client_order_id)`.
  - Атомарність: fills, записи ордерів (включно з auto-cancel), позиції й послідовність `exec_id` готуються послідовно на робочих копіях і фіксуються разом; помилка на будь-якому ордері batch не змінює нічого (годинник, якщо вже прочитано, транзакційним станом не є).
- Ідемпотентність: `client_order_id` унікальний у межах екземпляра (для всіх символів). Повтор із рівними за значенням умовами (Decimal порівнюється за значенням) повертає оригінальний `OrderAck` (той самий id і `exchange_ts`) у будь-якому стані ордера, нічого не створює й не відкриває знову; інші умови → `ExchangeDuplicateOrderError`, існуючий ордер не змінюється.
- `cancel_order`: OPEN → CANCELED (`exchange_ts` = момент скасування). Невідомий ордер чи вже фінальний → `ExchangeRejectedError` без змін стану, однаково при кожному повторі (відповідає контракту `TradingClient`: «already final or unknown»).
- `get_order`: пошук за `symbol` + `client_order_id`; інший символ — інший простір → `None`. Якщо передано `exchange_order_id`, що суперечить ордеру, → `ExchangeRejectedError`, а не `None`: ордер існує, і `None` дозволив би reconciliation вирішити «не розміщено» з ризиком дубля.
- `get_open_orders(symbol)`: лише активні ордери символу, порядок `(created_at, client_order_id)`.
- **Виконання (simulation-only вхід, не в `TradingClient`, не market data):** `fill_crossed_limit_orders(*, symbol, execution_price, available_qty=None) -> tuple[Fill, ...]`.
  - `execution_price` і `available_qty` (якщо задано) — точний `Decimal`, скінченний, `> 0`; `symbol` — та сама перевірка, що в `get_open_orders`. Некоректний вхід → `ExchangeRequestValidationError` до будь-яких змін (стан, послідовності й годинник не зачіпаються).
  - Учасники: OPEN і PARTIALLY_FILLED ордери символу. Перетин: BUY — `execution_price <= limit`, SELL — `execution_price >= limit`; рівність виконує. CANCELED / FILLED і інші символи не беруть участі.
  - `execution_price` **не** перевіряється на кратність `tick_size`: це зовнішнє спостереження, а не ціна нового ордера (змішувати валідність market data з pre-trade перевірками ордера не варто).
  - Ордер без зареєстрованого інструмента під час виконання — пошкоджений стан симулятора → помилка інваріанту до будь-яких змін (не пропуск і не rejection).
  - Модель price improvement: `Fill.price = execution_price` (BUY limit 100 при 95 → fill за 95). Книги, bid/ask, глибини, spread, slippage і латентності немає — ціну й ліквідність повністю задає викликач.
  - `available_qty=None` — необмежена ліквідність: кожен ордер отримує весь залишок (`qty − filled_qty`). Інакше — **загальний** обсяг на весь batch: ордери в порядку `(created_at, client_order_id)` отримують `min(залишок, ще доступно)`, **округлене DOWN до `qty_step`**, доки обсяг не вичерпано; частка, що округлюється до нуля, fill не дає (і `exec_id` не витрачає), невикористаний залишок бюджету не переноситься. `available_qty` не мусить бути кратним кроку — це зовнішній бюджет ліквідності, а не кількість ордера. Залишок ордера завжди кратний кроку (прийнята qty і всі попередні fills кратні), тож останній fill закриває його точно; некратний залишок — порушення інваріанту (помилка до commit). Максимум один fill на ордер за виклик. Це детерміноване правило симуляції, а не біржовий matching priority.
  - Lifecycle: OPEN → PARTIALLY_FILLED / FILLED; PARTIALLY_FILLED → PARTIALLY_FILLED (більший `filled_qty`) / FILLED; ніколи `filled_qty > qty`. `Fill.qty` — кількість саме цього виконання, не накопичена.
  - Середня ціна: точний накопичений notional (`Σ price·qty`, зберігається у записі) / накопичена кількість. Перший fill — рівно `execution_price`. Ділення — у явному `Context`: `AVERAGE_PRICE_PRECISION = 40` значущих цифр, `ROUND_HALF_EVEN`, незалежно від глобального контексту; округлюється лише нескінченний дріб (наприклад, 299/3), і похибка не накопичується, бо наступне середнє рахується з точного notional. До tick не округлюється. Кількості й notional — точні (80 цифр, інакше fail closed).
  - Метадані: без політики комісій `fee=None`, `fee_asset=None`, `is_maker=None`; з політикою — див. «Комісії» нижче.
  - `exec_id` — окрема послідовність екземпляра `SIM-EXEC-0000000001`, …, рівно один номер на кожен створений fill; ордери без fill, виклики без fills і невалідні виклики номери не витрачають.
  - Час: `clock.now()` читається рівно один раз на batch, що щось виконує (без виконань — жодного разу); усі fills і змінені ордери batch мають цей `exchange_ts` / `updated_at`.
  - Атомарність: спершу локально будуються розподіл, усі `Fill`, нові записи ордерів і наступне значення послідовності; лише потім одним кроком фіксуються ордери й послідовність. Помилка під час підготовки не лишає жодних змін і пропусків у `exec_id`. Запис, що порушує інваріанти (OPEN: 0 виконано; PARTIALLY_FILLED: `0 < filled < qty`; FILLED: `filled = qty`; CANCELED: `filled < qty`; середня ціна є тоді й лише тоді, коли є виконання), → помилка до commit, без «ремонту».
  - PARTIALLY_FILLED — активний: є в `get_open_orders` і скасовується. Скасування зберігає `cum_filled_qty` і `avg_fill_price`, `exchange_ts` = момент скасування; залишок більше не виконується. FILLED / CANCELED — фінальні (повторне скасування → `ExchangeRejectedError`).
  - Ідемпотентність після fill: однаковий повтор `place_order` повертає оригінальний `OrderAck` без змін стану; інші умови → `ExchangeDuplicateOrderError`.
- **Позиції** — `app/exchanges/simulated_positions.py`, `SimulatedPositionLedger` (лише stdlib + `app.domain`; не знає ордерів, інструментів, контрактів біржі й годинника). Читання — simulation-only `SimulatedExchange.get_position(*, symbol) -> Position | None` (не в `TradingClient`): некоректний аргумент → `ExchangeRequestValidationError`, символ без fills (зареєстрований чи ні) → `None`; годинник не читається.
  - One-way / net: одна знакова позиція на символ; BUY додає `+qty`, SELL — `−qty`.
  - Збільшення тієї самої сторони: `entry_price` — середньозважена ціна виконань. Внутрішня собівартість відкритої частини і realized PnL зберігаються як точні раціональні числа (`fractions.Fraction`), тож частинне закриття з нескінченним середнім (наприклад, 310/3) не лишає похибки; публікуються `entry_price` і `realized_pnl`, округлені до 40 значущих цифр `ROUND_HALF_EVEN` у явному `Context` (та сама точність, що в середньої ціни fill). Глобальний контекст не використовується; кількості — точні `Decimal`. До tick / step нічого не округлюється.
  - Протилежна сторона: спершу закривається відкритий обсяг (`(exit − basis)·closed` для long, `(basis − exit)·closed` для short), собівартість зменшується пропорційно; рівне закриття → FLAT (`entry_price=None`, `unrealized_pnl=0`); надлишок — розворот, нова сторона з базою за ціною цього fill.
  - `realized_pnl` — gross, накопичується через FLAT і нові позиції. Комісії fill (навіть відомі) і funding **не** враховуються.
  - Без mark: відкрита позиція — `mark_price=None`, `unrealized_pnl=None`; FLAT — `unrealized_pnl=0`. З mark — див. «Mark-to-market» нижче.
  - Потік fills має власний водяний знак (час останнього застосованого fill, внутрішній `last_fill_at`): fill, старіший за нього, → `PositionAccountingError` (внутрішній інваріант, `RuntimeError`, не помилка біржі) без змін; рівний час дозволено. Mark цей знак не рухає.
  - Ідемпотентність за `exec_id`: ідентичний повтор нічого не змінює; той самий `exec_id` з іншим payload → `PositionAccountingError`.
  - Атомарність: `begin_batch()` дає робочу копію, читання якої бачать її власні підготовлені fills (`signed_qty`, `apply`, `prepared()`); `prepare(fills)` — те саме для готового списку (у порядку fills batch, без пересортування); `commit(prepared)` встановлює стан лише для тієї версії ledger, на якій його підготовлено. У `fill_crossed_limit_orders` підготовка позицій іде після побудови fills і записів ордерів; потім одним кроком фіксуються позиції, ордери й послідовність `exec_id`. Помилка будь-якої підготовки не змінює нічого.
- **Mark-to-market** — simulation-only вхід `SimulatedExchange.set_mark_price(*, symbol, mark_price) -> Position | None` (не в `TradingClient`).
  - Mark — **явний** вхід симуляції; ніколи не виводиться з ціни виконання, ліміту, entry, тикера чи свічки. Зберігається в симуляторі окремо від собівартості (`symbol → MarkQuote(price, at)`), може надійти **до** першої позиції (тоді повертається `None`, позиція не створюється) і переживає FLAT; наступне відкриття оцінюється за збереженим mark.
  - Перевірка: некоректний symbol чи ціна (точний скінченний `Decimal` > 0) → `ExchangeRequestValidationError`; незареєстрований інструмент → `ExchangeRejectedError`; в обох випадках годинник не читається. Успішне оновлення читає `clock.now()` рівно один раз.
  - Оцінка — у ledger позицій з **точної** бази (`Fraction`), не з округленого `entry_price`: LONG `mark·qty − cost`, SHORT `cost − mark·|qty|`, FLAT `0` (з відомим `mark_price`). Публікація — 40 значущих цифр `ROUND_HALF_EVEN`, незалежно від глобального контексту; до tick не округлюється. Відомий нуль лишається відомим.
  - Часові потоки розділені: mark, старіший за збережений, → помилка інваріанту без змін (рівний час дозволено); mark не рухає водяний знак fills, тож новіший mark не робить старіший fill «застарілим», а захист від застарілих fills не послаблено. `Position.updated_at` = пізніший із часу останнього fill і часу mark. Повтор тієї самої ціни з новим часом — нова подія (оновлює `updated_at`), ідемпотентності за ціною немає.
  - Mark не змінює кількість, базу, realized, cash, ордери, fills і жодні id. Cash і далі `starting + realized − fees`; unrealized і equity туди не входять.
  - Атомарність: batch fills перевіряє, що опублікована позиція (за збереженим mark) обчислюється, **до** commit — помилка оцінки не змінює ордери, позиції, cash і `exec_id`. `set_mark_price` спершу рахує переоцінену позицію, а лише потім зберігає mark; помилка лишає старий mark і стару позицію.
- **Equity (read model)** — simulation-only `SimulatedExchange.get_equity_state() -> EquityState | None` (не в `TradingClient`). `EquityState(asset, cash, unrealized_pnl, equity)` — незмінний simulation-local тип у `simulated_accounting`; доменний `Balance` не використовується (його `available` / margin-семантика ширша, вигадувати її не можна).
  - **Похідна** модель, ніде не зберігається: `equity = точний cash + точний сумарний unrealized` під час читання. Джерела істини — ledger cash (`exact_cash()`, до округлення), ledger позицій (`exact_unrealized_total(marks)` — сума з точної бази через ту саму єдину функцію оцінки `_exact_unrealized`; формули long / short ніде не дублюються) і збережені mark.
  - Повнота: облік cash вимкнено → `None`; будь-яка OPEN позиція без mark → `None` (невідома оцінка ніколи не стає нулем, навіть якщо інші позиції відомі). FLAT позиції mark не потребують; без позицій equity = cash.
  - Публікація: кожне поле з точних значень, 40 значущих цифр `ROUND_HALF_EVEN` у явному `Context`, незалежно від глобального; до tick / точності активу не округлюється. Опубліковані `cash + unrealized_pnl` можуть відрізнятися від `equity` в останньому знаку, внутрішня рівність точна; поля штучно не підганяються.
  - Актив — актив обліку cash; усі інструменти котируються в ньому (перевірено конструктором), FX немає.
  - Mark змінює unrealized і equity, але не cash. Закриття переносить realized у cash: на mark без комісії equity не змінюється (часткове закриття LONG 10 @ 100, mark 110, SELL 4 @ 110: cash 10000 → 10040, unrealized 100 → 60, equity 10100 → 10100); закриття гірше за mark знижує equity на різницю; комісії й rebates впливають на equity **лише через cash**.
  - Читання чисте: годинник не читається, стан, mark, id не змінюються, повторні читання ідентичні. Окремого rollback немає: помилка batch fills чи оновлення mark не змінює джерела, тож equity лишається попередньою.
- **Комісії** — `app/exchanges/simulated_fees.py` (лише stdlib + `app.domain`; implementation-модуль, який імпортує тільки симулятор). Вмикаються **явно**: `SimulatedExchange(..., fees=SimulatedFeePolicy(schedule=TradingFeeSchedule(maker_rate, taker_rate, fee_asset), liquidity_role=...))`; один параметр, тож напівналаштованого стану немає. Без політики (`fees=None`) кожен fill має `fee=None`, `fee_asset=None`, `is_maker=None` — комісія **невідома**, а не нульова. Ставок за замовчуванням немає.
  - `LiquidityRole` (MAKER / TAKER) — параметр моделі симуляції, а не доменний факт: книги в момент розміщення немає, тож роль не виводиться ні з GTC, ні з POST_ONLY; одна задана роль діє для всіх fills (`is_maker=True` для MAKER, `False` для TAKER). POST_ONLY сам по собі maker не доводить.
  - Формула (linear): `fee = ціна виконання fill × кількість fill × ставка ролі` — окремо для кожного (часткового) fill; не ціна ліміту, не середня ціна ордера, не entry позиції. Обчислюється точно в явному `Context` (120 цифр, `Inexact` → помилка), незалежно від глобального контексту; округлення до точності активу при розрахунку на біржі **не моделюється**.
  - Ставки — точні скінченні `Decimal` будь-якого знаку, без штучного діапазону: нуль → відома нульова комісія (з активом), від'ємна ставка → від'ємна комісія (rebate). `fee_asset` задається явно (`require_text`), з символу не виводиться.
  - Комісія рахується під час підготовки fill і входить в атомарний batch: помилка розрахунку не змінює ордери, позиції, auto-cancel і послідовність `exec_id`. Auto-cancel reduce-only комісії не створює. Повтор `place_order` і повторне виконання вже FILLED ордера нових комісій не дають. Політика незмінна протягом життя симулятора.
  - `Position.realized_pnl` лишається gross: ledger комісії ігнорує (облік комісій і балансів — окремий майбутній шар).
- **Облік cash** — `app/exchanges/simulated_accounting.py` (лише stdlib + `app.domain`; implementation-модуль, який імпортує тільки симулятор; не імпортує ні симулятор, ні ledger позицій, ні комісії). Вмикається **явно**: `SimulatedExchange(..., cash=SimulatedCashConfig(asset="USDT", starting_cash=Decimal(...)))`; без нього стану cash немає, стартового капіталу за замовчуванням немає.
  - Модель derivatives-style, один актив обліку на симулятор: `cash = starting_cash + gross_realized_pnl − trading_fees`. Три компоненти зберігаються окремо (`cash` похідний, тож рівність виконується за побудовою). Відкриття чи збільшення позиції **не** рухає cash на notional — лише на комісію. Від'ємна комісія (rebate) зменшує `trading_fees` і збільшує cash; окремого «кошика» rebates немає.
  - `starting_cash` — точний скінченний `Decimal` ≥ 0 (нуль дозволено); `asset` — `require_text`, з символу не виводиться.
  - Джерело realized PnL — **точний** delta кожного fill з ledger позицій (`PositionBatch.apply` повертає `Fraction` до публічного округлення); математика закриття / розвороту в cash не дублюється. Публікуються значення з 40 значущими цифрами `ROUND_HALF_EVEN`; округлення при розрахунку на біржі не моделюється.
  - Комісія — лише з fill. Інваріанти конструктора (fail fast): облік cash вимагає політики комісій (інакше `fee=None` зробив би cash невідомим), `fee_asset` розкладу = активу cash, `quote_asset` кожного інструмента = активу cash. Конвертації валют немає. Невідома комісія чи комісія в іншому активі під час виконання → `CashAccountingError` до commit.
  - Часткове закриття, розворот (realized лише закритої частини, комісія на весь fill) і reduce-only — без окремої математики; auto-cancel не рухає cash.
  - Атомарність: cash готується в тому самому batch після позиції кожного fill (`CashBatch`), фіксується разом із позиціями, ордерами й послідовністю `exec_id`; помилка будь-якої підготовки не змінює нічого. Повтор того самого `exec_id` ідемпотентний, інший payload → `CashAccountingError`.
  - Читання — simulation-only `get_cash_state() -> CashState | None` (не в `TradingClient`; годинник не читається). Поле називається `cash`, **не** equity: unrealized PnL не входить. Доменний `Balance` не використовується, бо вимагав би вигаданих `equity` / `available`; майбутня equity fail closed, доки оцінка відкритих позицій невідома.
- Глобальний контекст `decimal` процесу **не** є частиною семантики симуляції: уся арифметика `Decimal` іде в явних `Context`, над `Fraction` або є контекстно-незалежною. Зміна знака й модуль `Decimal` — лише через `copy_negate()` / `copy_abs()` (точні, без округлення); `-x` і `abs(x)` над `Decimal` заборонені, бо округлюють за глобальним контекстом (регресія: при `prec=2` reduce-only виконував 3.4 замість 3.333 і розвертав позицію). Тести `test_simulated_decimal_context.py` проганяють сценарії з кількостями на кшталт 3.333 у звичайному і низькоточному контексті.
- Час — лише з injected `Clock` (у тестах `ManualClock`). `Ambiguous` ніколи не виникає (транспорту немає). MARKET-виконання, slippage, доменний `Balance` / available, плече, маржа, ліквідація, funding, депозити / виведення ще не реалізовані.
- Правило архітектури: `exchanges.simulated` — implementation (ядро й Bybit не можуть його імпортувати); сам він може імпортувати лише stdlib, `app.domain`, `app.exchanges.models`, `app.exchanges.errors`.

Цільові реалізації:

| Реалізація | Призначення |
|---|---|
| `exchanges/bybit/` | Реальна біржа: mainnet і testnet (різні base URL) |
| `exchanges/simulated` | Реалістична модель виконання для backtest і paper (зараз — лише детермінований lifecycle ордера, без виконання) |
| `exchanges/fake/` | Тестовий дубль зі скриптованими збоями для unit/failure-тестів |

Спільний **contract test suite** (однакові тести поведінки адаптера) проганяється проти `fake`, `simulated` і, вручну, проти Bybit testnet.

### 5.5 Вибір бібліотеки для Bybit

Рішення ще не прийняте, його приймаємо у фазі 2 після короткого spike (див. ROADMAP). Кандидати:

- **pybit** — офіційний SDK Bybit. Потрібно перевірити, чи є нативний asyncio (є ризик, що HTTP синхронний, а WS працює на потоках).
- **ccxt** (`ccxt.async_support`, WS через ccxt pro) — корисний для майбутніх Binance/OKX, але нормалізація може ховати Bybit-специфіку (`orderLinkId`, `positionIdx`, статуси).
- **Власний тонкий клієнт** (httpx + websockets) лише для потрібних ~15 ендпоінтів v5: повний контроль над timeout/retry/помилками, але більше коду для підтримки.

---

## 6. Market Data

### 6.1 Компоненти

```text
BybitPublicStream ──► MarketDataFeed ──► MarketDataCache ──► Event Queue
        │                    │
        │                    └── StaleDataMonitor (data_age, last_update, connection_status)
        └── ReconnectPolicy (backoff) + Heartbeat (ping/pong)
```

### 6.2 Потоки для v1

| Потік | Навіщо |
|---|---|
| Ticker (last, mark, funding) | Mark price для ризику й ліквідації, funding |
| Orderbook top (L1) | Спред, очікувана ціна виконання, стан ринку |
| Public trades | Paper-симуляція виконання лімітних ордерів |
| Klines (REST) | Бектест, аналіз стартових умов |

Private stream (ордери, виконання, позиції, гаманець) — окремий модуль в `exchanges/bybit/`, але подає події в ту саму Event Queue.

### 6.3 Свіжість і коректність

- Для кожного потоку та символу відстежуються `last_update` (received_ts), `data_age = clock.now() - last_update` і `connection_status` (CONNECTING / CONNECTED / STALE / DISCONNECTED).
- Поріг застарілості налаштовується в конфігурації. Коли дані `STALE`, RiskManager блокує **нові ордери, що збільшують позицію**. Reduce-only і скасування дозволені.
- Heartbeat: ping з інтервалом, який вимагає документація Bybit (перевірити). Якщо pong не приходить — reconnect.
- Для стакана перевіряємо `update_id` / `seq`, якщо біржа їх надає. Пропуск або розрив послідовності → повторний snapshot. Точну семантику snapshot/delta в Bybit перевіряємо в документації.
- Після reconnect **приватного** стріму події могли загубитися, тому це тригер для REST-reconciliation (розділ 13).
- Події з `exchange_ts` у майбутньому (понад допустимий clock skew) або з різким розривом логуються і не використовуються для рішень.

---

## 7. Execution Engine

### 7.0 Стан акаунта, резервування й серіалізація (V1, foundation)

- **`Order(NEW)` — локальна резервація.** Схвалений intent одразу стає доменним `Order(status=NEW)` у локальному стані акаунта **до** будь-якого мережевого виклику; NEW входить в активні статуси, тож наступний snapshot уже враховує його як pending exposure (повний `qty`) і в лічильнику ордерів акаунта.
- **Один власник ризик-релевантного стану — `InMemoryAccountState` (`app/execution/account_state.py`).** Ордери за `client_order_id`, індекс `intent_id → PlacementRecord` (execution-локальна незмінна модель: intent, `RiskDecision`, `client_order_id` або `None`; `intent_id` у доменний `Order` не додається), знакові позиції за символом, застосовані fills за `exec_id`, точний notional кожного ордера і **одна** `revision` — під **одним** `asyncio.Lock`. Окремого registry чи position book зі своїм lock немає, тож ордери й позицію неможливо прочитати з різних моментів. Межа серіалізації — акаунт (не символ: `max_open_orders` глобальний); V1: один процес-writer на акаунт; lock звільняється до мережевого виклику.
- **Lock API:** `async with account.account_lock() as locked:` дає синхронний handle `LockedAccountState` (views, `position_qty`, `register_approved`, `register_rejected`, `mark_submitting`, `apply_fill`, `set_position_qty`); мутації існують лише на handle й lock повторно не беруть, тож deadlock неможливий; повторний вхід у lock тією ж задачею → `AccountLockError`, handle після звільнення lock непридатний.
- **Позиції:** `None` (немає запису) = невідома / не звірена; `0` = відомо FLAT; `> 0` LONG; `< 0` SHORT. Відсутній запис ніколи не читається як нуль. До появи reconciliation позицію явно задає `set_position_qty` (під lock). Правила зміни позиції — чисті функції `app/portfolio/positions.py` без власного стану.
- **Fills застосовуються атомарно до ордера й позиції.** `apply_fill(fill, at=...)`: ордер за `client_order_id` (fill без нього — чужий ордер — поки не підтримується), symbol / side / відомий `exchange_order_id` мають збігатися; fill приймають лише SUBMITTING, OPEN, PARTIALLY_FILLED, CANCELING, UNKNOWN (NEW ще не відправлявся, термінальні фінальні). Накопичення `filled_qty`, точного notional і `avg_fill_price` — спільне правило `app/domain/fill_math.py` (те саме, що в симуляторі); новий статус — FILLED при повному виконанні, інакше CANCELING лишається CANCELING, решта → PARTIALLY_FILLED (наявна state machine). Позиція: BUY `+qty`, SELL `−qty`, точно; невідома лишається невідомою; reduce-only fill, що для відомої позиції не зменшує її (немає позиції, той самий бік) або розвертає, — пошкоджений стан → `FillApplicationError`. Усе готується до першої мутації: при будь-якій помилці ордер, позиція, notional, множина fills і revision не змінюються. Повтор ідентичного `exec_id` нічого не змінює; той самий `exec_id` з іншими даними → `FillConflictError`.
- **Ідемпотентність intent:** один запис на `intent_id`; повтор з рівним (поле в поле) intent повертає той самий запис без нового `Order`, `client_order_id` і зміни `revision`; інші дані під тим самим `intent_id` або зайнятий `client_order_id` → `PlacementConflictError`. `client_order_id` стан акаунта не генерує: його передає викликач, тут лише валідація й унікальність.
- **Revision — версія всього локального ризик-релевантного стану акаунта** (а не лише ордерів): починається з 0; +1 за резервацію NEW, перехід ордера (`mark_submitting`: write-ahead `NEW → SUBMITTING`, нічого не відправляється), застосований fill і зміну позиції; пізніше — за reconciliation і зміну `TradingState`. Відхилений запис, повтор intent чи fill і встановлення позиції в те саме значення → без змін. Нова реєстрація має назвати поточну revision (`expected_revision`), інакше `StaleRevisionError`.
- **`PlacementCoordinator` (`app/services/placement.py`) володіє атомарною межею `snapshot → evaluate → reserve`.** Під одним lock акаунта: перевірка повтору за `intent_id` (до оцінки, генерації id і читання Clock; рівний intent → той самий `PlacementRecord`, інші дані → `PlacementConflictError`) → revision R, **позиція символу**, активні ордери символу й лічильник акаунта з утримуваного handle → `build_risk_snapshot` з `snapshot_id = "<account_scope_id>:<R>"` (`account_scope_id` без `:`) → `evaluate` → відмова: `register_rejected` (revision R) / схвалення: `ClientOrderIdGenerator.next_id` → `clock.now()` → `register_approved` (`Order(NEW)` видимий до unlock, revision R + 1). Під lock немає мережі й `await`. Coordinator **більше не приймає позицію від викликача**: вона читається з того самого стану, що й ордери; `TradingState` поки передається викликачем. Неможливість точно побудувати snapshot (`ExposureCalculationError`) → `PlacementPreparationError` (не рішення Risk, нічого не записано); некоректні входи й порушення контракту evaluator, помилки генератора / Clock, некоректний або зайнятий id, час резервації раніше за intent — передаються далі без часткових змін. Coordinator завершується на `PlacementRecord`: submit — окремий наступний крок.
- **Біржа лишається джерелом істини.** Локальний стан — надмножина (включно з NEW / SUBMITTING / UNKNOWN, яких біржа ще не бачить); після reconciliation він узгоджується з біржею, а не навпаки. Відправка однієї спроби (`OrderSubmitter`, 7.2), застосування звітів біржі й одноразове розв'язання UNKNOWN (7.3) реалізовані; cancel, повтори / grace, recovery, persistence і повна reconciliation — ще ні.

### 7.1 Складові

- `ClientOrderIdGenerator` — унікальний, персистентний, з префіксом бота. Формат приблизно `{bot_prefix}{strategy_short}{time_base36}{counter}`. Обмеження довжини і дозволені символи для `orderLinkId` перевірити в документації Bybit (очікувано до 36 символів). Префікс дає змогу відрізнити «наші» ордери від сторонніх.
- `OrderManager` — синхронний: зберігає `Order`, застосовує переходи state machine, відкидає застарілі й дубльовані оновлення.
- `ExecutionEngine` — асинхронна оболонка: запускає I/O-задачі й повертає їхні результати як події.

### 7.2 Потік submit

```text
approved intent
  → Order(status=NEW, client_order_id=gen()) → persist
  → status=SUBMITTING → persist                 # write-ahead: після рестарту відомо, що запит міг піти
  → async task: trading_client.place_order(OrderRequest з Order)
       ack          → зберегти exchange_order_id; статус лишається SUBMITTING,
                      доки не прийде OrderUpdate (WS) або REST get_order
       rejected     → REJECTED
       unknown      → UNKNOWN → резолвер
       not sent     → FAILED
  → OrderUpdate з WS/REST → OPEN / PARTIALLY_FILLED / FILLED / CANCELED ...
  → якщо за N секунд після ack немає підтвердження → REST get_order
```

REST-відповідь «ордер прийнято» **не** переводить ордер ні в `FILLED`, ні навіть в `OPEN`. Статус підтверджує тільки звіт про стан ордера.

**Реалізовано (V1 foundation, без persistence): `OrderSubmitter` (`app/execution/submitter.py`).** `submit(client_order_id=...)` відправляє лише наявну резервацію `Order(NEW)`; запит будує чистий `order_request_from_order(order)` (`app/execution/requests.py`) — усі поля з `Order`, нічого не генерується й не береться з intent.

- **Write-ahead до мережі.** Під lock акаунта: ордер має бути NEW (інакше `OrderAlreadySubmittedError`, мережа не викликається: повторний submit і два конкурентні submit дають рівно один запит), побудова `OrderRequest`, читання Clock, `NEW → SUBMITTING` (+1 revision). Збій на цьому кроці лишає NEW і нічого не відправляє.
- **Мережа поза lock**: `await client.place_order(request)` рівно один раз; автоматичних повторів немає.
- **Класифікація лише за типом винятку** (не за текстом): `ExchangeNotSentError` (включно з `ExchangeRequestValidationError`) → `FAILED`; `ExchangeRejectedError` (включно з `ExchangeAuthenticationError`) → `REJECTED`; усе інше — `ExchangeAmbiguousResultError`, `ExchangeDuplicateOrderError`, будь-яка інша помилка чи скасування задачі (запит міг дійти) → `UNKNOWN`, що лишається активною експозицією до резолвера. Після запису результату вихідний виняток передається далі. `TradingState` тут не змінюється (напр., при помилці автентифікації) — це задача майбутнього safety controller.
- **Ack — збагачення метаданих, а не перехід в OPEN.** Ack має належати цьому `client_order_id`; `exchange_order_id` записується доменною операцією `record_exchange_order_id` (статус той самий, version + 1, вже відомий інший id не замінюється) для будь-якого відправленого стану, включно зі станом, уже просунутим fills. Ack для NEW / FAILED / REJECTED, чужий `client_order_id`, інший уже відомий `exchange_order_id` чи не-ack → `OrderAckMismatchError`; SUBMITTING стає `UNKNOWN` (резервація не звільняється).
- **Пріоритет станів під гонками:** підтверджений біржею прогрес (fills) > транспортний результат > локальний маркер SUBMITTING. Результат запиту вирішує лише ордер у SUBMITTING; якщо fill уже просунув ордер, `ambiguous` нічого не змінює, а `not sent` / `rejected` суперечать фактам → `SubmissionOutcomeConflictError` (стан збережено, транспортна помилка — у `__context__`). Стан ніколи не відкочується.
- **Час.** Час відправки — одне читання Clock на кроці write-ahead (`updated_at` у SUBMITTING). Час результату / ack читається під lock після мережі, але збій Clock, некоректне чи раніше `updated_at` значення замінюється на `updated_at` ордера, тож відомий результат записується завжди.

### 7.3 Резолвер `UNKNOWN`

1. Запитати `get_order(client_order_id)` з backoff кілька разів протягом налаштованого вікна.
2. Знайдено → перевести в фактичний статус.
3. Не знайдено після вичерпання вікна → `FAILED` + `BotEvent(warning)`.
4. Поки хоча б один ордер у стані `UNKNOWN`, стратегія **не** отримує дозволу на новий ордер для того самого рівня сітки. Ризик-перевірки рахують такий ордер як потенційно активний (експозиція).
5. Якщо резолвер не може отримати відповідь від біржі — `PAUSED`.

Повторної відправки з тим самим `client_order_id` у v1 немає. Нову спробу стратегія ініціює сама на наступному кроці, з новим ID, тільки після того як старий ордер у фінальному стані.

**Реалізовано (V1 foundation, одна спроба): застосування звітів біржі й `UnknownOrderReconciler` (`app/execution/reconciliation.py`).**

- **Нормалізація.** `exchange_state_from_update(OrderUpdate)` → execution-локальний `ExchangeOrderState` (поле в поле, нічого не виправляється). Біржовими фактами є лише OPEN, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED, EXPIRED; локальні статуси (NEW, SUBMITTING, CANCELING, UNKNOWN) і FAILED («не відправлено») біржа повідомити не може → помилка. Стан акаунта від exchange DTO не залежить.
- **Застосування (`LockedAccountState.apply_exchange_state`)**, усе перевіряється до зміни: ордер існує й був прийнятий (не NEW / FAILED); відомий `exchange_order_id` збігається (відсутній — записується); виконана кількість звіту **дорівнює** локально застосованим fills — менша → застарілий звіт (`ExchangeStateMismatchError`), більша → `MissingFillsError`; **fills ніколи не синтезуються зі звіту**, тож Order і позиція лишаються узгодженими (спочатку треба застосувати бракуючі fills, потім повторити звіт); відома середня ціна збігається з локальною **точно** (локальний облік не замінюється). TODO: інтеграція з live-біржею має визначити нормалізацію середньої ціни за точністю біржі / інструмента, перш ніж послаблювати точне порівняння; epsilon / допуск зараз не використовується. Інший статус вирішують наявна state machine й інваріанти `Order` — недосяжний статус (PARTIALLY_FILLED → OPEN, будь-яка зміна термінального, REJECTED з fills) → mismatch; state machine глобально не послаблюється. Позиція тут ніколи не змінюється. Звіт, що лише підтверджує статус, може записати новий `exchange_order_id`; ідентичний звіт — no-op. Revision +1 лише за реальну зміну.
- **UNKNOWN — це reconciliation, а не повторна відправка.** `reconcile(client_order_id=...)`: під lock ордер має бути UNKNOWN (інакше `OrderNotUnknownError`, без читання) → поза lock **рівно один** `get_order(OrderRef(symbol, client_order_id, exchange_order_id))` → під lock звіт застосовується до **поточного** локального ордера. Без повторів, backoff, grace period, циклів і фонових задач.
- **Помилки читання** передаються далі; ордер лишається UNKNOWN, revision не змінюється. **Not found** (`None`) класифікується відносно ордера в момент обробки відповіді (lock береться знову) і нічого не змінює: «досі не розв'язано» (`OrderStillUnknownError`; ордер лишається UNKNOWN і активною експозицією; правило «не знайдено → FAILED» потребує окремої grace-політики) — **лише якщо ордер досі UNKNOWN з тим самим `exchange_order_id`**. Якщо за час читання локально з'явився новіший біржовий факт (fill → PARTIALLY_FILLED / FILLED, інший звіт → OPEN / CANCELED / EXPIRED / REJECTED, новий `exchange_order_id`), not found йому суперечить → `ExchangeStateMismatchError`, стан збережено. Зникнення ордера між двома lock — порушення інваріанта (`AccountStateError`).
- **Гонки.** Поки читання в дорозі, ордер може просунутися (fills): звіт застосовується до новішого стану за тими самими правилами, тож застарілий звіт — mismatch, а не відкат. Два одночасні reconcile можуть обидва прочитати біржу; перший застосовує (+1 revision), другий — no-op.
- **UNKNOWN → FILLED** через звіт можливий лише після застосування бракуючих fills (UNKNOWN не може мати повного виконання), і тоді його вже зробив сам fill; звіт лише підтверджує.

### 7.4 Порядок оновлень

- `cum_filled_qty` не може зменшуватися: оновлення з меншим значенням ігноруються як застарілі.
- Fill дедуплікується за `exec_id` (також унікальний ключ у БД).
- Fill може прийти раніше за OrderUpdate і навпаки. Портфель рахується з fills, статус ордера — з OrderUpdate. Розбіжність між `sum(fills.qty)` і `cum_filled_qty` після таймауту → перевірка через REST.

---

## 8. Життєвий цикл ордера (state machine)

Базові стани з `CLAUDE.md` плюс два додаткові: `UNKNOWN` (невідомий результат відправки, розділ 7 `CLAUDE.md`) і `CANCELING` (запит на скасування надіслано, але не підтверджено).

```text
                  ┌──────────► FAILED (нічого не надіслано / підтверджено відсутність)
                  │
NEW ──► SUBMITTING ──► OPEN ──► PARTIALLY_FILLED ──► FILLED
            │   │       │  │           │   │
            │   │       │  └──► CANCELING ◄┘
            │   │       │          │
            │   │       ▼          ▼
            │   │    CANCELED / EXPIRED
            │   └──► REJECTED
            └──────► UNKNOWN ──► (будь-який фактичний статус) | FAILED
```

Схема ілюстративна; джерело правди — таблиця нижче і `app/domain/order_state.py`
(`ALLOWED_TRANSITIONS`, тест `test_table_matches_contract_exactly`).

### Таблиця дозволених переходів

| Із | До |
|---|---|
| NEW | SUBMITTING, FAILED |
| SUBMITTING | OPEN, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED, FAILED, UNKNOWN |
| OPEN | PARTIALLY_FILLED, FILLED, CANCELING, CANCELED, EXPIRED, UNKNOWN |
| PARTIALLY_FILLED | PARTIALLY_FILLED, FILLED, CANCELING, CANCELED, EXPIRED, UNKNOWN |
| CANCELING | CANCELING, FILLED, CANCELED, EXPIRED, UNKNOWN |
| UNKNOWN | OPEN, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED, EXPIRED, FAILED |
| FILLED, CANCELED, REJECTED, EXPIRED, FAILED | — (фінальні) |

Правила:
- Переходи в той самий стан є тільки два: `PARTIALLY_FILLED → PARTIALLY_FILLED` і `CANCELING → CANCELING`. Вони несуть нові дані виконання і дозволені лише зі **строго більшим** кумулятивним `filled_qty`. Загального self-transition немає.
- Часткове виконання під час очікування скасування: ордер **залишається в `CANCELING`** з більшим `filled_qty` (переходу `CANCELING → PARTIALLY_FILLED` немає). `CANCELING → CANCELING` дозволений лише коли `old_filled_qty < new_filled_qty < qty`. Якщо кумулятивне виконання досягло `qty`, ордер фактично виконаний: тільки `CANCELING → FILLED`. Гонка «скасування проти виконання» завершується `CANCELING → FILLED` або `CANCELING → CANCELED`.
- `UNKNOWN` не фінальний. У нього можна потрапити з робочих станів (`SUBMITTING`, `OPEN`, `PARTIALLY_FILLED`, `CANCELING`), вихід — у фактичний статус за результатом перевірки (розділ 7.3). Сам резолвер — Phase 5.
- Перший звіт біржі не зобов'язаний проходити через `OPEN`: `SUBMITTING → PARTIALLY_FILLED / FILLED / CANCELED` дозволені.
- Для звичайних переходів `filled_qty` не зменшується (може лишитися тим самим). Час переходу не може бути раніше за `updated_at`. Кожен перехід збільшує `version` рівно на 1.
- Дозволена дуга не означає, що будь-які дані допустимі: інваріанти «статус ↔ виконання» перевіряються окремо. `NEW`, `SUBMITTING`, `OPEN`, `REJECTED`, `FAILED` — `filled_qty = 0`; `PARTIALLY_FILLED` — `0 < filled_qty < qty`; `FILLED` — `filled_qty = qty`; `CANCELING`, `UNKNOWN`, `CANCELED`, `EXPIRED` — `0 <= filled_qty < qty`.
- **Кожен активний (нетермінальний) `Order` має залишок `qty − filled_qty > 0`.** Достовірно відоме повне виконання — це `FILLED`, а не `UNKNOWN` / `CANCELING` / інший активний статус. `UNKNOWN` — консервативна активна експозиція (увесь ненульовий залишок) до reconciliation; підтверджене згодом повне виконання — перехід `UNKNOWN → FILLED`. Інваріант перевіряє конструктор `Order`, тож і `transition()` (через `dataclasses.replace`).
- Підтвердження біржі (ack), яке лише присвоює `exchange_order_id`, **не є переходом `OrderStatus`**: `SUBMITTING → SUBMITTING` немає. Реалізовано як доменну операцію `record_exchange_order_id(order, exchange_order_id, at=...)` (`app/domain/order_state.py`): зберігає `status`, збільшує `version` на 1, оновлює `updated_at`, не дозволяє змінити вже встановлений `exchange_order_id`; той самий id повторно — без змін.
- Недозволений перехід — це помилка стану, а не привід її проігнорувати: `BotEvent(error)` і запит статусу через REST.
- `SUBMITTING → CANCELED` потрібен, бо post-only ордер, що перетнув би стакан, біржа може скасувати одразу (як саме Bybit повідомляє про це — перевірити).
- `CANCELED` може мати `filled_qty > 0`. Портфель враховує частину, що виконалась.
- Кожен перехід записується в `order_events`.

---

## 9. Risk Manager

### 9.0 Межа і контракт V1 (затверджено; реалізація — Phase 6, не завершена)

- **Потік:** `Strategy → PlaceOrderIntent → instrument preflight → RiskManager → Execution`. Вхід Risk — доменний `PlaceOrderIntent` (exchange-neutral); Risk не імпортує реалізації бірж і не бачить `OrderRequest`. `CancelOrderIntent` через Risk не йде: скасування ризик лише зменшує й доступне завжди.
- **Лише approve / reject.** Risk не змінює intent: не зменшує й не збільшує кількість, не додає `reduce_only`, не змінює ціну чи сторону; немає REDUCED і `approved_qty`. Надмірний intent відхиляється.
- **Перевірки `InstrumentSpec`** (tick, qty step, min / max qty біржі, min notional) — окремий preflight-контракт, Risk їх не дублює.
- **Чисте детерміноване ядро:** `evaluate(intent, snapshot, policy) -> RiskDecision` без Clock, мережі, mutable-стану й логування; snapshot і policy — незмінні й подаються явно (Risk сам ні в біржу, ні в симулятор не ходить). Арифметика `Decimal` — у явних контекстах (знак / модуль — `copy_negate` / `copy_abs`), результат не залежить від глобального контексту.
- **Snapshot V1:** `snapshot_id`, `symbol`, `trading_state`, `position_qty` (знакова; `None` = невідомо, `0` = відомо FLAT), `open_orders` (`None` = невідомо, порожньо = відомо немає) — деталі потенційно активних ордерів **лише цього символу**, включно з `SUBMITTING` / `UNKNOWN` (повний залишок), для worst-case позиції; `account_open_order_count` (`None` = невідомо, `0` = відомо немає) — кількість активних ордерів **усього акаунта** (усі символи) для глобального `max_open_orders`. Ці значення не пов'язані рівністю (на інших символах можуть бути ордери); якщо обидва відомі, `account_open_order_count ≥ len(open_orders)`, інакше snapshot суперечливий. Поле обов'язкове, без значення за замовчуванням. **Немає** cash, equity, mark і часу / свіжості — вони з'являться разом із чесною семантикою свіжості оцінки.
- **Розклад intent** за знаковою позицією `q` і знаковим `d` (BUY +, SELL −), а не за стороною: `reducing = min(|d|, |q|)` при протилежних знаках, інакше 0; `increasing = |d| − reducing`. SELL при SHORT — збільшення. Розворот (LONG 5, SELL 8) має обидві частини: зменшення 5 і збільшення 3.
- **Ліміти V1:**
  - `max_order_qty`, `max_order_notional` (per-symbol) — fat-finger ліміти **нового ордера**: для звичайного intent застосовуються до **всього** `qty` і `limit price × qty`, а не лише до частини, що збільшує (LONG 5, SELL 100 при ліміті 10 → відмова). **Валідний reduce-only звільнений** від обох: виконання зобов'язане обмежити fill позицією, яку можна зменшити. MARKET без опорної ціни при увімкненому ліміті номіналу → відмова.
  - `max_position_qty` (per-symbol) — **worst-case** майбутня позиція: поточна + усі відкриті не-reduce-only ордери свого боку + новий не-reduce-only intent; протилежні відкриті ордери не нетуються (вважаються невиконаними), reduce-only ордери worst-case не збільшують. Відкриті ордери **враховуються обов'язково** (інакше два BUY 10 при ліміті 10 пройшли б обидва).
  - `max_open_orders` (глобальний, за `account_open_order_count`) — для **всіх** нових розміщень, включно з reduce-only: при `count ≥ max_open_orders` → відмова `MAX_OPEN_ORDERS` (при `count = max − 1` нове розміщення ще дозволене); якщо ліміт увімкнено, а лічильник невідомий → `UNKNOWN_OPEN_ORDERS`; при `max_open_orders = None` глобальний лічильник не потрібен.
  - Ліміти кількості — per-symbol (одиниці базового активу різні); символ без лімітів у політиці → відмова. `None` = ліміт явно вимкнено; значень за замовчуванням немає.
  - Поза V1: equity / min equity, cash, номінал позиції за mark, загальна експозиція, leverage / margin / ліквідація, daily loss / drawdown, свіжість даних.
- **TradingState:** RUNNING — усе за лімітами; REDUCE_ONLY — лише intents з `reduce_only=True` (звичайний «закриваючий» BUY / SELL відхиляється: позиція може змінитися до виконання й ордер розверне її); PAUSED — жодних розміщень; HALTED (активний kill switch) — жодних розміщень стратегії, включно з reduce-only; аварійне закриття — окремий майбутній механізм; скасування поза Risk доступне завжди. Персистентний латч і дії kill switch поки не реалізовані.
- **Reduce-only:** невідома позиція → відмова; FLAT / немає позиції → відмова; не той бік → відмова; валідний, але більший за позицію → дозволено (виконання обмежує fill і не допускає розвороту).
- **Fail closed:** невідомий потрібний стан → відмова (невідома позиція — будь-яке розміщення; невідомі відкриті ордери символу — коли потрібні для worst-case позиції, або невідомий лічильник акаунта — коли увімкнено `max_open_orders`; обидва випадки — одна причина `UNKNOWN_OPEN_ORDERS`); неможливий точний розрахунок → відмова.
- **Помилки:** відмова — очікуваний результат `RiskDecision` з машинними кодами `RiskReason`, не exception; exception лише для некоректного входу (типи, невідповідність symbol, некоректні snapshot / policy).
- **Аудит:** рішення містить `intent_id`, `snapshot_id`, `policy_id`, `reasons`, `exposure`; persistence / логування — пізніше.
- **Побудова snapshot (`app/risk/snapshots.py`, чиста межа):** `open_order_exposure(order)` перетворює активний локальний `Order` (NEW, SUBMITTING, OPEN, PARTIALLY_FILLED, CANCELING, UNKNOWN) на `OpenOrderExposure` з точним `remaining_qty = qty − filled_qty` (> 0) і скопійованими `side`, `price` (`None` для MARKET), `reduce_only`, `status`; термінальний ордер → `DomainValidationError`; позитивний залишок гарантує домен, `OpenOrderExposure` перевіряє його ще раз захисно. `build_risk_snapshot(...)` отримує вже зібраний локальний уніфікований перелік активних ордерів **одного символу** (`None` = невідомо), зберігає їхній порядок, відхиляє чужий символ, термінальний ордер і повтор `client_order_id`, а `account_open_order_count` передає як є (не через `len`). Вона не читає біржу чи реєстр, не бере lock, не генерує `snapshot_id`, не визначає `TradingState` і не рахує позицію з fills — це робить orchestration.
- **Конкурентність:** чисте ядро гонку двох рішень над одним snapshot не вирішує. Серіалізація й резервування — в orchestration / execution: оцінка, створення `Order(NEW)` і включення його у відкриті ордери наступного snapshot відбуваються послідовно під одним lock акаунта (див. 7.0). Стан акаунта (ордери + позиції + revision під одним lock) і `PlacementCoordinator` (`services`), що поєднує його з `evaluate`, реалізовано (7.0); власник `TradingState`, submit і persistence — ще ні.

### 9.1 Три рівні перевірок (цільова картина; V1 — див. 9.0)

1. **Pre-trade (кожен intent).** Відповідність tick/step/min_qty/min_notional — окремий instrument preflight перед Risk; у Risk: максимальна кількість на ордер; максимальна позиція з урахуванням активних **і** `SUBMITTING`/`UNKNOWN` ордерів; максимальний capital allocation на стратегію; максимальна кількість одночасних ордерів; `max_loss_per_trade` (для Grid — втрата на рівні worst-case, див. 9.4); максимальне плече; достатність вільної маржі.
2. **Portfolio (при кожній зміні стану).** Загальна експозиція, drawdown від піку equity, денний збиток, відстань до ліквідації.
3. **Системні circuit breakers.** Застарілі дані, розрив private stream, серія API-помилок, частка відхилених ордерів, невирішена розбіжність reconciliation.

### 9.2 TradingState

| Стан | Нові ордери, що збільшують позицію | Reduce-only | Скасування | Вихід |
|---|---|---|---|---|
| RUNNING | так | так | так | — |
| REDUCE_ONLY | ні | так | так | автоматично, коли умова зникла |
| PAUSED | ні | ні (крім kill switch) | так | автоматично після успішної синхронізації або вручну |
| HALTED | ні | тільки дії kill switch | так | **тільки вручну** |

Для кожного типу порушення в конфігурації явно задається цільовий стан. Приклад:

```yaml
risk:
  on_stale_data: REDUCE_ONLY
  on_daily_loss_limit: HALTED
  on_max_drawdown: HALTED
  on_reconciliation_mismatch: PAUSED
  on_liquidation_distance_breach: HALTED
```

### 9.3 Kill Switch

- Незалежний компонент. Має прямий доступ до `TradingClient` в обхід Strategy.
- Тригери: вручну (CLI-команда або файл-прапорець), критичні ліміти ризику, повторні критичні помилки.
- Дії (налаштовуються): (1) заблокувати нові ордери — завжди; (2) `cancel_all_orders` — налаштовується, за замовчуванням так; (3) закрити позиції reduce-only ринковими ордерами — налаштовується окремо; (4) записати причину; (5) сповістити.
- Стан kill switch персистентний: після рестарту бот залишається в `HALTED`, доки людина не зніме його вручну.

### 9.4 Оцінка ліквідації

- Власний оцінювач: ціна ліквідації для ізольованої позиції з урахуванням `maintenance_margin_rate` за tier ризик-ліміту, комісії на закриття та mark price.
- Для Grid головна перевірка — **worst case**: яка позиція накопичиться, якщо ціна пройде всі рівні до межі діапазону, і де тоді ліквідація. Якщо оцінка ліквідації при worst-case позиції ближча до межі діапазону, ніж налаштований буфер → попередження або блокування старту.
- У live значення `liquidation_price` від біржі має пріоритет, а власний оцінювач — для контролю і для backtest. Розбіжність понад поріг логується.
- **Застереження:** у режимі cross margin (особливо на Unified Trading Account) ліквідація залежить від усього акаунта, а не від однієї позиції. Власна оцінка там ненадійна. Це один з аргументів за isolated margin у v1 (відкрите питання).

---

## 10. Portfolio і відстеження позицій

- Позиція будується з **fills**: знакова кількість, середня ціна входу, realized PnL, комісії, funding. Облік за середньою ціною (не FIFO), бо perpetual-позиція на біржі має одну середню ціну входу. Точне правило Bybit для середньої при частковому зменшенні і перевороті позиції перевірити.
- Unrealized PnL рахується від mark price.
- Equity = wallet balance + unrealized PnL.
- Облік окремо: `gross_pnl`, `fees`, `funding`, `net_pnl`.
- Funding-платежі отримуються з біржі: через стрім виконань або історію транзакцій (тип записів і канал перевірити в документації).
- Локальна позиція періодично і після reconnect звіряється з `get_positions()`. Розбіжність більша за `qty_step` вважається розбіжністю reconciliation (розділ 13).
- Для Grid окремо ведеться **inventory сітки**: скільки позиції набрано незакритими рівнями. Зовні позиція одна, але стратегії потрібна розбивка по рівнях.

---

## 11. Persistence

- **SQLAlchemy 2.x async Core** (без ORM для доменних сутностей) + **Alembic**. SQLite (`aiosqlite`) для розробки й durable-локальних запусків; PostgreSQL (`asyncpg`) для testnet-довгих прогонів і production. Залежності ще не додано (див. 11.0).
- Ризик-релевантний стан акаунта зберігається **однією транзакційною межею** `AccountStateStore` (11.0), а не набором незалежних репозиторіїв. Журнали (`*_events`, snapshots, `strategy_state`) — окремі append-only записи; бізнес-логіки в persistence немає.
- API-секрети в БД не зберігаються.

### 11.0 Durable стан акаунта і recovery contract (затверджено; реалізації немає)

Мета: після будь-якого падіння процесу бот **ніколи не вважає exposure меншою за реально можливу**.

**Модель persistence (V1).** `prepare immutable next state → durable transaction commit → publish next state to RAM`. RAM ніколи не публікується раніше за успішний durable commit.
- Definite commit failure (точно не закомічено): RAM лишається старим; операція завершується помилкою.
- Uncertain commit outcome (результат commit невідомий): стан акаунта вважається **unusable / poisoned** — подальші мутації заборонені до reload / recovery з durable store; effective `TradingState` лишається PAUSED.

**Write-ahead (обов'язковий інваріант).** `durable SUBMITTING commit` — **до** того, як може початися `place_order`. Якщо durable store не підтвердив SUBMITTING, мережева відправка заборонена. Звідси доказово: `durable NEW після рестарту → відправка не починалася → recovery NEW → FAILED` (за умов: єдиний шлях до `place_order` — `OrderSubmitter`; один writer; `client_order_id` ніколи не перевикористовується).

**Порядок DB / RAM для мутації:** `account lock → прочитати поточний стан / revision → підготувати незмінний наступний стан → await durable commit, поки account lock утримується → при успіху опублікувати стан у RAM → unlock`. `await` commit БД під account lock дозволений і потрібний для серіалізації; `await` мережі під account lock як і раніше заборонений.

**Межа мережі й БД.** `DB SUBMITTING → network → DB outcome`. Розподіленої транзакції з біржею немає; розрив між БД і біржею закриває консервативний стан **UNKNOWN**.

**Атомарні транзакції** (кожен рядок — одна транзакція):

| Операція | Зміни |
|---|---|
| схвалена резервація | `PlacementRecord` + `Order(NEW)` + revision |
| маркер відправки | `Order(SUBMITTING)` + revision |
| fill | Fill / `exec_id` + Order + точний виконаний notional + позиція + revision |
| звіт біржі / результат відправки / ack | статус і метадані Order + revision |

Order і позицію від одного fill **не можна** зберігати окремими транзакціями.

**Revision.** Збережена revision акаунта — монотонно зростаюча версія закомічених змін локального ризик-релевантного стану; після рестарту не скидається. Commit — CAS за `expected_revision`. V1: один процес / один writer; writer token у кожній мутації зараз не проєктується.

**Один writer.** Обмеження V1: **один writer-процес на `account_scope_id`**; два writers — непідтримувана конфігурація й помилка розгортання. Майбутнє посилення (не частина першої реалізації): PostgreSQL advisory lock, SQLite process / file lock, lease / takeover policy.

**Архітектура.** Обрано `AccountStateStore` Protocol — одна транзакційна persistence-межа агрегату акаунта (ордери, placements, fills, позиції, revision). Не кілька незалежних репозиторіїв для мутацій Order / Fill / Position; не event sourcing. Стан акаунта не виконує SQL (dependency inversion).

**Реалізований контракт store (з execution ще не інтегровано).** Порт належить шару execution / стану акаунта; `app.persistence` містить лише адаптери й кодеки (dependency inversion):

```text
domain ← execution (порт: app/execution/persistence.py) ← services
                         ↑
        app.persistence adapters (memory.py; згодом SQL) + codecs.py
```

`app.execution` ніколи не імпортує `app.persistence` (перевіряє архітектурний тест для всього пакета); порт залежить лише від stdlib, domain і власних моделей execution.
- `AccountStateStore` (`app/execution/persistence.py`): `load(account_scope_id=...) -> PersistedAccountState | None` і `commit(AccountStateChange)`; без lease / writer token / SQL-об'єктів. Там само — persisted-моделі й помилки порту.
- `PersistedAccountState` — незмінний знімок (read-only mappings): revision, placements за `intent_id`, orders за `client_order_id`, fills за `exec_id`, `PersistedPosition` за символом, точний виконаний notional за `client_order_id`. Перевикористовує `PlacementRecord`, `Order`, `Fill` як є.
- `PersistedPosition(symbol, known, qty)` — явний маркер (unknown зберігається, а не кодується відсутністю). `PersistedOrderNotional` — точний скінченний Decimal ≥ 0.
- `AccountStateChange` — набір записів (placements, orders, fills, positions, notionals) з `expected_revision`; `new_revision` = `expected` або `expected + 1`; зміна з тією самою revision може записувати лише відхилені placements. Для акаунта без жодного commit `expected_revision = 0`.
- Правила commit: CAS за revision (`StoreConflictError`); ідентичний повторний запис — no-op («ідентичний» = рівний і однаково представлений, тож `Decimal("4")` і `Decimal("4.0")` — різні payload); та сама ідентичність з іншими даними — конфлікт (placements, fills); `Order` замінюється лише більшою `version` (та сама — no-op або конфлікт, менша — конфлікт); позиції й notional — проєкції, що замінюються в межах успішного CAS. Унікальні в межах scope: `intent_id`, `client_order_id` (у тому числі між placements), `exec_id`, `exchange_order_id`, символ позиції. Посилання результуючого стану: схвалений placement має ордер з тими самими умовами intent; fill належить наявному ордеру (символ, бік, сумісний `exchange_order_id`; без `client_order_id` — відмова); кожен ордер має рівно один notional, нульовий тоді й лише тоді, коли нічого не виконано. Store не відтворює state machine і нічого не обчислює.
- Атомарність: повна валідація, потім застосування цілком. Помилки: `StoreValidationError` (зміна некоректна), `StoreConflictError`, `StoreCommitError` (точно не закомічено), `StoreUncertainError` (результат невідомий, може бути застосовано повністю).
- `InMemoryAccountStateStore` (`app/persistence/memory.py`) — **еталонний адаптер / fake, не durable-сховище**: кілька scopes, внутрішній lock серіалізує commit, детермінована ін'єкція збоїв (`DEFINITE` — нічого не записано; `UNCERTAIN` — застосовано повністю, потім помилка); валідація й конфлікти мають пріоритет над ін'єкцією. CAS за revision **не замінює** серіалізацію на рівні акаунта: store не є механізмом координації кількох writers.
- Кодеки (`codecs.py`) використовує лише майбутній фізичний DB-адаптер; in-memory store тримає доменні Decimal / datetime як є.

**Decimal.** Ризик-критичні Decimal ніколи не зберігаються як float (зокрема SQLite `NUMERIC` / `REAL`). Логічне представлення — **канонічний lossless decimal-текст** для qty, price, `avg_fill_price`, виконаного notional, позиції й збережених Decimal-полів exposure. Кодек приймає лише exact finite `Decimal`, відхиляє NaN / Infinity і робить round-trip без залежності від глобального decimal-контексту (експонента й хвостові нулі зберігаються). «Канонічний» означає детерміноване lossless представлення: зберігаються цифри коефіцієнта, експонента (scale, хвостові нулі) і знак нуля (`decode(encode(x)).as_tuple() == x.as_tuple()`); `normalize()` не застосовується. Реалізовано в `app/persistence/codecs.py`: encoder пише scientific string з великою `E` через явний контекст (результат `str()` залежав би від глобального `capitals`); decoder приймає будь-яке строге скінченне ASCII-написання без пробілів, підкреслень і не-ASCII цифр (пошкодження сховища не «виправляється»). Час: encoder вимагає aware datetime з нульовим offset і нічого не конвертує, пише лише `YYYY-MM-DDTHH:MM:SS.ffffff+00:00`; decoder приймає лише цю форму. Помилка — `PersistenceCodecError` (`ValueError`).

**Час.** Збережені мітки — UTC aware instants; naive datetime заборонені. SQLite — канонічний UTC-текст; PostgreSQL — `timestamptz`. Біржові та локальні мітки — окремі поля.

**Позиція.** Durable-представлення **явно** розрізняє unknown / known flat / known long / known short; «немає рядка = unknown» не є єдиним механізмом. Логічно: `known: bool`, `qty: Decimal | None` з інваріантом `known=False → qty=None`, `known=True → qty` — скінченний Decimal (включно з 0). Після рестарту навіть збережене `known=True` **не** робить runtime-позицію актуальною: recovery тимчасово робить runtime-view невідомим до звірки з біржею.

**Durable ідентичність** (переживає рестарт): `intent_id`, `client_order_id`, `exec_id`, `exchange_order_id`. Концептуальні обмеження: `UNIQUE(account_scope_id, intent_id)`, `UNIQUE(account_scope_id, client_order_id)`, `UNIQUE(account_scope_id, exec_id)`, `UNIQUE(account_scope_id, exchange_order_id) WHERE exchange_order_id IS NOT NULL`. Повтор intent після рестарту не викликає нову оцінку Risk (зберігаються й відхилені placements); повтор `exec_id` не застосовує fill удруге.

**`client_order_id`.** Майбутній production-генератор — криптографічно стійкий випадковий ідентифікатор зі стабільним префіксом бота, у межах обмежень адаптера біржі; унікальність у БД — defense in depth. Простий збережений лічильник не є єдиним джерелом унікальності (стирання dev-БД призвело б до повтору id на біржі). Генератор ще не реалізовано.

**Аудит політики.** `policy_id` — content-addressed або однозначно пов'язаний з хешем канонічного вмісту політики. Збережений placement / `RiskDecision` містить ідентичність / хеш політики, `snapshot_id`, reasons, результат exposure і сам intent: історичне рішення лишається поясненим після зміни поточної конфігурації.

**TradingState.** Напрям: майбутній `SafetyController` володіє effective `TradingState` і поєднує durable / ручний safety latch з runtime-умовами (recovery, здоров'я store, reconciliation). Перша persistence-реалізація `SafetyController` не містить.

**Пропущені fills.** `MissingFillsError` не можна обходити синтезом fills з `OrderUpdate`. Повний startup recovery потребує можливості історії виконань (`get_executions` або еквівалент). Доки її немає: `missing fills виявлено → recovery неповний → PAUSED`. Це не блокує реалізацію persistence foundation; Bybit-ендпоінт історії тут не проєктується.

**Неоднозначність API позицій.** Відсутність символу в майбутньому `get_positions()` поки **не** трактується як flat: контракт адаптера має явно це гарантувати, перш ніж recovery зможе вважати відсутність запису відомим нулем.

**UNKNOWN + not found.** Grace-політика не визначена; лишається поточна семантика: `UNKNOWN + authoritative not-found → нерозв'язаний UNKNOWN`, доки окрема політика не доведе безпечне звільнення резервації.

**Довговічність SQLite.** Режим, що претендує на crash durability: `journal_mode=WAL`, `synchronous=FULL`. In-memory paper durability не обіцяє; persistent paper можна додати пізніше.

**Логічна схема** (без DDL; DEC — канонічний decimal-текст, TS — UTC):

| Сутність | Ключ | Головне |
|---|---|---|
| `accounts` | `account_scope_id` | revision (CAS), durable safety latch, мітки часу |
| `policies` | `policy_id` | хеш і канонічний вміст політики |
| `placements` | (`account_scope_id`, `intent_id`) | усі поля intent; рішення (approved, reasons, `snapshot_id`, `policy_id`, exposure DEC); `client_order_id` (NULL для відхилених), unique |
| `orders` | (`account_scope_id`, `client_order_id`) | усі поля `Order` (DEC, TS, `version`); виконаний notional DEC; `exchange_order_id` unique, якщо не NULL |
| `fills` | (`account_scope_id`, `exec_id`) | повний payload Fill; FK на ордер |
| `positions` | (`account_scope_id`, `symbol`) | `known` bool, `qty` DEC NULL з інваріантом known ↔ qty |

### Таблиці

Загальний перелік, включно з журналами. Для ризик-релевантного стану акаунта (placements, orders, fills, positions, accounts, policies) визначальні логічна схема й обмеження з 11.0 (наприклад, унікальність `exec_id` у межах `account_scope_id`).

| Таблиця | Зміст | Особливості |
|---|---|---|
| `orders` | Поточний стан ордерів | unique `client_order_id` |
| `order_events` | Кожен перехід статусу | append-only |
| `fills` | Виконання | unique `(exchange, exec_id)` — ідемпотентний insert |
| `positions_snapshots` | Знімки позицій (локальні і біржові) | append-only |
| `balance_snapshots` | Знімки балансу | append-only |
| `funding_payments` | Funding | unique за ключем біржі |
| `trades` | Завершені цикли (для Grid — пара buy→sell або sell→buy) | для метрик |
| `strategy_state` | Серіалізований стан стратегії (JSON + `schema_version`) | одна актуальна версія + історія |
| `grid_levels` | Стан рівнів сітки | FK на `strategy_state` |
| `bot_events` | Старт, стоп, reconnect, помилки (`category=error`), зміни TradingState, хеш конфігурації | append-only |
| `risk_events` | Відхилення, ліміти, kill switch | append-only |
| `reconciliation_events` | Розбіжності та дії | append-only |
| `kill_switch_state` | Поточний стан (латч) | один запис на бот |

### Відмова БД

Якщо запис у БД неможливий, бот не може гарантувати write-ahead для ордерів. Тому:
- помилка запису перед відправкою ордера → ордер **не відправляється**;
- невизначений результат commit → стан акаунта poisoned, мутації заборонені до reload / recovery, effective `PAUSED` (11.0);
- серія помилок БД → `PAUSED` + сповіщення (сповіщення працюють без БД).

---

## 12. State Recovery

Послідовність запуску (однакова для paper, testnet, live; у backtest спрощена):

```text
1. Завантажити і валідувати конфігурацію; записати config hash (без секретів).
2. Preflight (розділ 16.3 для live; спрощений для testnet/paper).
3. Перевірити kill_switch_state: якщо HALTED → лишатися в HALTED; інакше effective PAUSED до кінця recovery.
4. Завантажити durable-стан, відтворити його доменними конструкторами (перевірка інваріантів; невалідний запис → старт зупиняється) і класифікувати (11.0):
   NEW → FAILED (доведено write-ahead), SUBMITTING → UNKNOWN, UNKNOWN → UNKNOWN;
   OPEN / PARTIALLY_FILLED / CANCELING — потенційно застарілі, exposure зберігається, потрібна reconciliation;
   біржово-підтверджені термінальні стани не повертаються в активні через рестарт;
   runtime-позиції невідомі до звірки з біржею (збережене значення — лише «останнє локальне»).
5. Підключити private stream і БУФЕРИЗУВАТИ події (ще не застосовувати).
6. Отримати снапшот з біржі: open orders, positions, balances, fills з моменту останнього відомого.
7. Reconciliation (розділ 13). Результат: OK або розбіжність.
8. Застосувати пропущені fills → перебудувати портфель (потрібна можливість історії виконань;
   без неї пропущені fills → recovery неповний → PAUSED; fills ніколи не синтезуються з OrderUpdate).
9. Застосувати буфер подій зі стріму (з дедуплікацією).
10. Відновити стратегію: restore_state(snapshot) + звірка з фактичними ордерами.
11. Стратегія повертає «бажаний» набір ордерів; різниця з фактичним → intents через Risk.
12. TradingState = RUNNING тільки якщо кроки 6–10 пройшли без невирішених розбіжностей.
    Інакше PAUSED + сповіщення. Біржа недоступна або reconciliation неповна → PAUSED.
```

Відсутність символу у відповіді позицій біржі не означає flat, доки контракт адаптера цього явно не гарантує (11.0).

Стан стратегії в пам'яті, що лишився з минулого запуску, ніколи не використовується без звірки з біржею.

---

## 13. Exchange Reconciliation

### Коли запускається

- При старті (розділ 12).
- Після reconnect private stream.
- Періодично (інтервал у конфігурації).
- Після резолву `UNKNOWN`, який закінчився `FAILED`.
- Вручну (CLI).

### Що порівнюється і як реагуємо

| Ситуація | Автоматична дія | Стан |
|---|---|---|
| Локально OPEN, на біржі FILLED | Підтягнути fills, застосувати | RUNNING |
| Локально OPEN, на біржі CANCELED | Оновити статус; стратегія вирішує, що робити з рівнем | RUNNING |
| Локально активний, на біржі не знайдено ніде | `FAILED` + подія | RUNNING, якщо позиція сходиться; інакше PAUSED |
| На біржі наш ордер (з префіксом), локально немає | Імпортувати в стан, подія warning | PAUSED до ручного підтвердження (у v1) |
| На біржі сторонній ордер (без префікса) на нашому символі | Не чіпати, подія | PAUSED (субакаунт має бути виділеним) |
| Позиція не збігається і розбіжність пояснюється пропущеними fills | Застосувати fills | RUNNING |
| Позиція не збігається і розбіжність не пояснюється | Подія critical | PAUSED або HALTED (конфігурація) |
| Баланс відрізняється понад поріг (після врахування fees/funding) | Подія | PAUSED |
| Стан сітки не узгоджується з ордерами | Подія | PAUSED |

Кожна перевірка пише `ReconciliationEvent` з обома версіями стану (локальною і біржовою). Навіть коли все збігається, пишеться короткий запис `OK` для аудиту.

---

## 14. Grid Strategy (місце в архітектурі)

Детальний дизайн Grid — у фазі 7 ROADMAP. Тут фіксуємо архітектурні рішення.

Grid розділена щонайменше на дві частини: **геометрія сітки** (чисті детерміновані ціни рівнів між межами, без округлення до tick) і **прийняття рішень** (перетворення рівнів і подій на intents). Реалізована поки лише геометрія: `app/strategies/grid/levels.py` (`generate_grid_levels`). `levels` — загальна кількість рівнів разом з обома межами. Модуль `strategies` залежить тільки від `app.domain` і stdlib (перевіряє тест архітектури).

- Стратегія — синхронний клас з інтерфейсом:

```text
Strategy (protocol)
  strategy_id
  on_start(ctx) -> list[Intent]
  on_market(event, ctx) -> list[Intent]
  on_order_update(update, ctx) -> list[Intent]
  on_fill(fill, ctx) -> list[Intent]
  snapshot_state() -> StrategyStateSnapshot
  restore_state(snapshot, ctx) -> None
```

  `ctx` — read-only view: instrument spec, fee schedule, власні активні ордери, власна позиція, останній ринковий стан, `clock.now()`. Адаптера в `ctx` немає.

- Сітка = набір рівнів. Після fill buy на рівні `i` ставиться sell на рівні `i+1`; після fill sell на `i` — buy на `i-1`. Режим (LONG / SHORT / NEUTRAL) визначає **початковий стан** і прапорці `reduce_only`:
  - **LONG:** buy нижче ціни відкривають і нарощують long; sell вище закривають (reduce-only).
  - **SHORT:** дзеркально.
  - **NEUTRAL:** buy нижче і sell вище; нетто-позиція коливається навколо нуля (в one-way position mode).
- Pre-start аналіз (окремий чистий модуль, без стану): валідація параметрів проти `InstrumentSpec`, розрахунок прибутковості циклу (gross, fees, slippage, funding, net), положення ціни в діапазоні, worst-case позиція і ліквідація. Результат — звіт + вердикт OK / WARN / BLOCK.
- `out_of_range_policy` для нижньої і верхньої межі задається **окремо і явно** (без значення за замовчуванням): `STOP_TRADING`, `KEEP_ORDERS`, `CANCEL_GRID`, `REBUILD_GRID`, `WAIT_FOR_RETURN`, `EMERGENCY_EXIT`. `REBUILD_GRID` у v1 не рекомендується: він «доганяє» тренд і фіксує збиток на inventory.

---

## 15. Backtesting

### Схема

```text
HistoricalDataLoader ──► ReplayFeed (SimulatedClock) ──► Event Queue ──► TradingEngine
                                                                             │
                           Strategy / Risk / OrderManager / Portfolio  (ТІ САМІ)
                                                                             │
                                                            SimulatedExchange (fill model)
```

### Правила

- Той самий `TradingEngine`, Strategy, RiskManager, OrderManager, Portfolio. Відрізняються тільки фід, `SimulatedExchange`, `SimulatedClock` і тимчасова БД.
- **Look-ahead захист:**
  - ReplayFeed видає події строго в порядку часу; `SimulatedClock` монотонний (assert);
  - свічка доступна стратегії лише після її закриття (`close_time`);
  - `SimulatedExchange` виконує ордер тільки на даних, що з'явилися **після** моменту його розміщення (+ опційна латентність).
- **Модель виконання лімітних ордерів:**
  - консервативно: fill, коли ціна **пройшла крізь** рівень, а не лише торкнулась (touch-fill — окрема опція, завжди позначається у звіті);
  - на 1m свічках шлях ціни всередині бару невідомий. Якщо свічка перетинає кілька рівнів, використовується консервативне припущення: у межах одного бару **не** закриваються обидві сторони одного циклу, якщо це не випливає з порядку OHLC. Це обмеження явно вказується у звіті;
  - для фінальної валідації бажані історичні трейди (tick data) замість свічок (джерело — відкрите питання);
  - часткові fills: у v1 опція «обмеження частки обсягу бару»; без неї — повне виконання.
- **Витрати:** maker/taker fee з конфігурації (параметри акаунта), slippage для ринкових ордерів (половина спреду + bps), funding за історичними ставками в моменти funding за інтервалом інструменту, помножений на позицію за mark/close ціною.
- **Ліквідація:** за оцінювачем з розділу 9.4 на основі close/mark. Якщо ціна перетнула оцінку ліквідації — примусове закриття за ціною ліквідації з комісією, подія у звіті.
- **Survivorship bias:** набір символів задається явно; у звіті вказується, що символ жив увесь період.

### Результати

Обов'язкові метрики з `CLAUDE.md`: total return, net / gross PnL, fees, funding, max drawdown, Sharpe, Sortino, win rate, profit factor, кількість трейдів, середній трейд, найбільші прибуток і збиток, експозиція, equity curve.

Додатково для Grid:
- кількість завершених циклів;
- **inventory і нереалізований PnL на кінець періоду** (grid часто показує «прибуток» по циклах, який перекривається збитком на накопиченій позиції);
- максимальна позиція, мінімальна відстань до ліквідації, час поза діапазоном.

«Трейд» для Grid = завершений цикл buy→sell або sell→buy; незакриті рівні враховуються окремо.

Аналіз стійкості (фаза 8): sensitivity-sweep параметрів, розбивка на ринкові режими, out-of-sample період.

---

## 16. Paper, Testnet, Live

### 16.1 Paper trading

- Ринкові дані — **mainnet** public WS (реальні ціни й ліквідність).
- Виконання — той самий `SimulatedExchange`, що в backtest, але в real time: лімітний ордер виконується за потоком публічних трейдів (ціна пройшла крізь рівень).
- Стан `SimulatedExchange` персиститься, щоб paper переживав рестарт і на ньому можна було тренувати recovery.
- Мета — перевірити поведінку стратегії й ризику на живому ринку без грошей.

### 16.2 Testnet

- Bybit testnet: окремі base URL і **окремі** ключі.
- Ринкові дані — **testnet** public WS (ордери виконуються проти testnet-стакану, тому й ціни мають бути звідти).
- Ліквідність і ціни на testnet не відповідають mainnet. Testnet перевіряє **інтеграцію** (auth, формати, статуси, помилки, reconnect, recovery), а **не** прибутковість.
- Альтернатива — Bybit Demo Trading (mainnet-ціни, віртуальні кошти). Підтримку в API і обмеження треба перевірити; це відкрите питання.

### 16.3 Live

- Потрібні одночасно: `TRADING_MODE=live`, `LIVE_TRADING_ENABLED=true` **і** успішний live preflight.
- Live preflight (будь-яка помилка = старт заборонено):
  - ключі присутні й валідні; права: торгівля є, **виведення вимкнене**; IP whitelist, якщо API повідомляє про це;
  - біржа і mainnet endpoint відповідають режиму (mapping режим→URL зашитий у коді, а не в конфігу);
  - акаунт очікуваного типу; position mode і margin mode як у конфігурації;
  - символ торгується, `InstrumentSpec` отримано;
  - плече на біржі = плече в конфігурації;
  - доступний баланс ≥ capital allocation;
  - ризик-ліміти задані й валідні; worst-case позиція і ліквідація в межах;
  - kill switch не в HALTED;
  - reconciliation пройшов без розбіжностей.
- `capital_allocation` у конфігурації — жорстка межа: бот не використовує більше, навіть якщо на акаунті більше коштів.

### 16.4 Що спільне, а що відрізняється між режимами

| Компонент | Backtest | Paper | Testnet | Live |
|---|---|---|---|---|
| Domain models | спільні | спільні | спільні | спільні |
| Strategy | спільна | спільна | спільна | спільна |
| RiskManager + KillSwitch | спільні | спільні | спільні | спільні |
| OrderManager / ExecutionEngine | спільні | спільні | спільні | спільні |
| Portfolio | спільний | спільний | спільний | спільний |
| TradingEngine loop | спільний | спільний | спільний | спільний |
| Config schema | спільна | спільна | спільна | спільна |
| Logging | спільне | спільне | спільне | спільне |
| **Market data** | ReplayFeed (історія) | Bybit mainnet public WS | Bybit testnet public WS | Bybit mainnet public WS |
| **Trading adapter** | SimulatedExchange | SimulatedExchange (real time) | BybitAdapter (testnet) | BybitAdapter (mainnet) |
| **Private stream** | від SimulatedExchange | від SimulatedExchange | Bybit testnet private WS | Bybit mainnet private WS |
| **Clock** | SimulatedClock | SystemClock | SystemClock | SystemClock |
| **Persistence** | SQLite (тимчасова / in-memory) | SQLite | SQLite або PostgreSQL | PostgreSQL |
| **Reconciliation** | не потрібна (симулятор узгоджений) | проти SimulatedExchange | реальна | реальна |
| **Notifications** | вимкнені | опційно | увімкнені | увімкнені |
| **Preflight** | валідація конфігурації | + доступність даних | + ключі, права, акаунт | повний live preflight |

Уся різниця між режимами збирається в одному місці — `services/bootstrap`.

---

## 17. Конфігурація

- **Секрети — тільки з env** (`pydantic.SecretStr`): `BYBIT_API_KEY`, `BYBIT_API_SECRET`, `TELEGRAM_BOT_TOKEN`, `DATABASE_URL` (якщо з паролем).
- **Режим:** `TRADING_MODE` = `backtest | paper | testnet | live`, `LIVE_TRADING_ENABLED`.
- **Параметри — YAML:** `configs/development.yaml`, `paper.yaml`, `testnet.yaml`; `production.yaml` з'явиться у Phase 13. Зараз у YAML: профіль, біржа, символ і параметри Grid; ризик, пороги stale-даних, reconciliation і нотифікації додаються у фазах, які їх використовують. У YAML ніколи немає секретів і обмежень інструмента (tick, step, мінімуми — від біржі).
- **Завантаження YAML** (`app/config/loader.py`): похідний від `SafeLoader` loader без Python-тегів; дробові числа стають `Decimal` з тексту скаляра (без `float`), цілі — лише десятковий запис, дубльовані ключі й шістдесяткові числа — помилка. Помилки не містять вмісту файлу.
- **Завантаження конфігурації** (`app/config/bootstrap.py`, `load_config`): environment (+ явно переданий `.env`) → `EnvSettings` → безпечний шлях профілю (лише імена з `PROFILE_MODES`, файл після `resolve()` має лишатися в `configs/`, symlink назовні відхиляється) → YAML `AppConfig` → незмінний `LoadedConfig(env, app)`. Змінні оточення ОС мають пріоритет над `.env`. Завантаження конфігурації **не має runtime-побічних ефектів**: не налаштовує логування, не створює клієнтів, не відкриває БД, не змінює оточення й робочий каталог. Секрети в логування передає runtime bootstrap (`loaded.env.secret_values()`).
- **Профіль ↔ режим:** `development` — `backtest`, `paper`; `paper` — `paper`; `testnet` — `testnet`. `profile.name` у файлі має збігатися з `CONFIG_PROFILE`. Без профілю `production` режим `live` неможливо сконфігурувати.
- **Валідація:** Pydantic-схема з локальними перевірками (`lower_price > 0`, `upper_price > lower_price`, `levels ≥ 2`, `order_qty > 0`, `extra="forbid"`). Перевірки на кшталт `leverage ≤ risk.max_leverage` і обидві `out_of_range_policy` додаються разом із відповідними полями (Phase 6–7); `TRADING_MODE=live` можливий лише з профілем `production`.
- При старті в `bot_events` пишеться хеш конфігурації й сама конфігурація без секретів.
- Base URL бірж для кожного режиму визначені в коді, а не в YAML. Так неможливо випадково спрямувати live-ключі на неправильний endpoint або testnet-режим на mainnet.

---

## 18. Логування

- **structlog**, JSON у файл / stdout, людиночитний формат у dev.
- Обов'язкові поля контексту: `mode`, `exchange`, `symbol`, `strategy_id`, а також `intent_id`, `client_order_id`, `exec_id` там, де застосовно. Так можна простежити ланцюжок intent → order → fill.
- Логуються: сигнали/intents, рішення ризику, кожен перехід ордера, fills, зміни позиції, risk events, помилки біржі, reconnect, reconciliation, метрики продуктивності.
- **Захист секретів:** процесор structlog, що маскує ключі за іменем поля (`key`, `secret`, `token`, `signature`, `password`, `authorization`) і за значенням відомих секретів. HTTP-заголовки й тіла підписаних запитів не логуються повністю. Тест: у логах немає значення секрету, навіть коли він потрапив у виняток.

---

## 19. Моніторинг

v1 — мінімально, без зовнішньої інфраструктури:
- `HealthMonitor` агрегує: статус WS (public/private), `data_age`, затримку API, частку помилок API, частку відхилених ордерів, TradingState, кількість відкритих ордерів.
- Метрики портфеля: realized/unrealized PnL, drawdown, експозиція, використання маржі, відстань до ліквідації.
- Періодичний знімок метрик у БД + рядок у лог.
- CLI `status` читає останній стан.
- Пізніше, за потреби, — Prometheus endpoint / Grafana. Не в v1.

---

## 20. Нотифікації

- `Notifier` protocol + `TelegramNotifier`.
- `NotificationDispatcher`: обмежена черга (bounded), окрема задача, таймаут на відправку, throttle і дедуплікація однакових повідомлень. Якщо черга повна, повідомлення відкидаються з записом у лог. **Помилка нотифікацій ніколи не впливає на торговий цикл.**
- Події з `CLAUDE.md`: старт, стоп, fill, відкриття/закриття позиції, ризик-ліміт, kill switch, втрата з'єднання, розбіжність reconciliation, критична помилка.
- Fills у Grid можуть бути частими, тому для них налаштовується агрегація (зведення раз на N хвилин).
- У v1 Telegram — **тільки вихідний канал**. Команди з Telegram (включно з kill switch) не приймаються, бо це окрема поверхня атаки. Kill switch — через CLI / файл-прапорець на сервері.

---

## 21. Межі безпеки

| Межа | Правило |
|---|---|
| Секрети | Тільки env / secret manager; `SecretStr`; не в YAML, БД, логах, винятках, експорті конфігурації |
| Доступ до ключів | Ключі потрапляють лише в `exchanges/bybit` (auth). Решта коду отримує адаптер, а не ключі |
| Права ключа | Торгівля — так; виведення — **ні** (перевіряється в preflight); IP whitelist — якщо доступний |
| Акаунт | Виділений субакаунт з обмеженим балансом; основні кошти на іншому акаунті |
| Торгові виклики | Тільки `ExecutionEngine` і `KillSwitch` мають `TradingClient` |
| Режими | Mapping режим → endpoint у коді; live потребує двох прапорців і preflight |
| Git | `.env`, дампи БД і логи в `.gitignore`; у репозиторії тільки `.env.example` з порожніми значеннями |
| Вхідні канали керування | У v1 тільки CLI на сервері; жодних вхідних команд з месенджерів |

---

## 22. Що свідомо НЕ робимо у v1

- Кілька бірж одночасно, крос-біржовий арбітраж.
- Кілька інстансів стратегії та символів одночасно (моделі це дозволяють, оркестрація — ні).
- Hedge position mode (див. відкриті питання).
- Автоматичний `REBUILD_GRID`.
- Зовнішні брокери повідомлень, мікросервіси, Kubernetes.
- Вхідні команди з Telegram.
- ML / оптимізація параметрів у live.

---

## 23. Непідтверджені припущення про Bybit API

Нижче те, що потрібно **перевірити в актуальній документації Bybit v5** до реалізації відповідних фаз. Жоден з цих пунктів не вважається фактом:

1. `category=linear` для USDT perpetual.
2. `orderLinkId` як client order id: максимальна довжина, дозволені символи, чи біржа відхиляє дублікат, чи можна отримати ордер за `orderLinkId` після його закриття (і як довго).
3. Семантика статусів ордера (`New`, `PartiallyFilled`, `Filled`, `Cancelled`, `Rejected`, `PartiallyFilledCanceled`, `Deactivated` тощо) і їх mapping на доменні.
4. Поведінка post-only ордера, що перетинає стакан.
5. Private WS: канали `order`, `execution`, `position`, `wallet`; чи приходять funding-записи через `execution`.
6. Ping/heartbeat інтервал для WS; семантика snapshot/delta і `seq` у стакані.
7. Rate limits для кожної групи ендпоінтів і заголовки зі станом ліміту.
8. Ендпоінт з інформацією про API-ключ (права, IP whitelist).
9. Ендпоінт ставок комісій акаунта.
10. Unified Trading Account: доступність isolated margin, як біржа повертає `liqPrice` у cross/isolated.
11. `positionIdx` і one-way / hedge mode.
12. Testnet і Demo Trading: base URL, які ендпоінти підтримуються.
13. Джерело історичних даних (klines через REST, архіви трейдів) та глибина історії funding.
14. Інтервал funding для конкретного символу (може відрізнятися від 8 год).
15. Наявність disconnect-protection механізму (автоскасування ордерів при втраті з'єднання) для perpetual.

---

## 24. Відкриті питання (потрібне рішення власника)

1. Бібліотека для Bybit: pybit / ccxt / власний тонкий клієнт (рішення у фазі 2).
2. Margin mode: isolated (рекомендовано для v1) чи cross.
3. Position mode: one-way (рекомендовано) чи hedge.
4. Старт Long/Short Grid: з пласкої позиції (рекомендовано для v1) чи з початковою позицією під рівні по інший бік ціни (класичний варіант).
5. Дані для бектесту: 1m свічки (простіше) чи трейди (точніше для Grid).
6. Testnet чи Demo Trading як етап перед live (або обидва).
7. Kill switch за замовчуванням: закривати позиції чи лише скасовувати ордери.
8. Власник і напрямок залежностей KillSwitch. Зараз KillSwitch описаний у `risk/`, якому дозволено імпортувати тільки `domain` і `portfolio`, але KillSwitch потребує `TradingClient` з `exchanges/`. Follow-up: *Define the ownership and dependency direction for KillSwitch before implementing the risk/execution integration.*
