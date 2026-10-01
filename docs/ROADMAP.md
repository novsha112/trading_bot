# Roadmap розробки

> Статус: чернетка v0.1 (2026-09-30).
> Архітектура — `docs/ARCHITECTURE.md`. Правила — `CLAUDE.md`.
> Правило переходу: наступна критична фаза не починається, поки попередня не має тестів і не виконала критерії завершення. Успіх попередньої фази не означає автоматичного переходу до наступної.

---

## Зміни порядку відносно запропонованого і причини

| # | Зміна | Причина |
|---|---|---|
| 1 | **Bybit розділено на дві частини:** public market data у фазі 3, приватна торгівля (ордери, private WS) — у фазі 11 | Торговий код для реальної біржі не повинен з'являтися раніше, ніж є Execution Engine, Risk Manager і kill switch, які ним керують. Публічні дані потрібні рано для бектесту й paper |
| 2 | **Recovery & Reconciliation (фаза 10) перенесено перед Testnet (фаза 11)** | На testnet ми вже працюємо з реальними ордерами і рестартами. Без reconciliation перший же рестарт на testnet — неконтрольована ситуація. Базовий резолв `UNKNOWN` з'являється ще раніше, у фазі 5 |
| 3 | **Структуроване логування і маскування секретів — у фазах 0–1, а не 12** | Логування потрібне з першого рядка коду; фаза 12 додає тільки агрегацію метрик і сповіщення |
| 4 | **Ядро kill switch і TradingState — у фазі 6 (Risk)** | Kill switch не може з'явитися пізніше за першу стратегію |
| 5 | **Таблиця переходів ордера — у фазі 1 (domain), виконання — у фазі 5** | State machine — чиста доменна логіка, її тестують ізольовано |
| 6 | **`SimulatedExchange` створюється у фазі 8 і повторно використовується у фазі 9** | Одна модель виконання для backtest і paper — інакше результати не порівнювані |

Ці зміни узгоджуються з порядком у `CLAUDE.md` (розділ 36): пункт «Bybit adapter» розбито на дві частини, reconciliation у `CLAUDE.md` окремим етапом не виділено.

## Карта залежностей

```text
0 Bootstrap
└─ 1 Domain & Config
   ├─ 2 Exchange Abstraction (+ FakeExchange, рішення про бібліотеку)
   │  ├─ 3 Bybit Public Market Data
   │  └─ 5 Execution Engine ◄── 4 Persistence
   │     └─ 6 Risk Manager & Kill Switch
   │        └─ 7 Grid Strategy
   │           └─ 8 Backtesting (SimulatedExchange) ◄── 3 (історичні дані)
   │              └─ 9 Paper Trading ◄── 3 (live WS)
   │                 └─ 10 Recovery & Reconciliation
   │                    └─ 11 Bybit Private Adapter & Testnet
   │                       └─ 12 Monitoring & Notifications
   │                          └─ 13 Live Readiness
   │                             └─ 14 Live Trading (поетапно)
   └─ 4 Persistence
```

Фази 3 і 4 можна робити в будь-якому порядку після фази 2.

---

## Phase 0 — Project Bootstrap

**Мета.** Порожній, але робочий каркас проєкту з інструментами якості.

**Що реалізуємо.**
- Структуру каталогів з `CLAUDE.md`.
- `pyproject.toml`: Python 3.11+, залежності розробки (ruff, mypy, pytest, pytest-asyncio). Менеджер пакетів: uv (рекомендовано) або poetry — вибір власника.
- Конфігурацію ruff і mypy (strict для `app/domain`, `app/risk`, `app/execution`, `app/strategies`).
- `.gitignore` (`.env`, `*.db`, `logs/`, `data/`), `.env.example` з порожніми значеннями.
- Базове налаштування structlog з процесором маскування секретів.
- Тест правил залежностей між модулями (наприклад, `strategies` не імпортує `exchanges`).
- Опційно: pre-commit, CI (GitHub Actions: ruff + mypy + pytest).

**Файли.** `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`, `app/**/__init__.py`, `app/monitoring/logging.py` (або `app/config/logging.py`), `tests/unit/test_architecture_imports.py`, `tests/unit/test_log_redaction.py`.

**Тести.** Імпорти пакетів; тест правил залежностей; тест маскування секретів у логах (секрет у полі й у тексті винятку).

**Критерії завершення.** `ruff check`, `ruff format --check`, `mypy`, `pytest` проходять; `.env` ігнорується git; README описує, як запустити перевірки.

**Залежності.** Немає.

---

## Phase 1 — Domain Models & Configuration

**Мета.** Спільна мова системи і валідована конфігурація.

**Що реалізуємо.**
- Enums: `Side`, `OrderType`, `TimeInForce`, `OrderStatus`, `TradingMode`, `TradingState`, `GridMode`, `GridSpacing`, `OutOfRangePolicy`, `MarginMode`, `PositionMode`.
- Моделі з розділу 4 ARCHITECTURE (`InstrumentSpec`, `Ticker`, `Order`, `Fill`, `Position`, `Balance`, intents, events…).
- Хелпери округлення до `tick_size` / `qty_step` з явним напрямком (вниз / вгору), перевірка `min_qty` і `min_notional`.
- Таблиця дозволених переходів `OrderStatus` і функція `transition(order, new_status)` з помилкою на недозволений перехід.
- `Clock` protocol + `SystemClock` + `FixedClock` / `SimulatedClock` для тестів.
- Конфігурація: Pydantic Settings (env) + YAML loader; схеми `ExchangeConfig`, `RiskConfig`, `GridConfig`, `MarketDataConfig`, `NotificationConfig`; профілі `development/paper/testnet/production`.
- Крос-валідація конфігурації; `live` дозволений тільки з двома прапорцями й профілем `production`.
- Mapping режим → base URL біржі в коді (значення URL беруться з документації у фазі 3/11; поки що — порожні заглушки з помилкою на використання).

**Фактичний обсяг після review.** Моделі без поточного споживача відкладено (`events`, `RiskConfig`, `MarketDataConfig`, `NotificationConfig`, `out_of_range_policy`, `production.yaml` — у фазах, що їх використовують). Mapping режим → base URL перенесено в Exchange Adapter (Phase 3/11). Схема YAML у Phase 1: `profile`, `exchange` (`bybit`, symbol), `strategy.grid` (lower/upper price, levels, spacing, mode, order_qty).

**Файли.** `app/domain/{enums,instrument,market,orders,order_state,fills,positions,balances,intents,events,errors,clock,rounding}.py`; `app/config/{settings,schema,loader}.py`; `configs/{development,paper,testnet,production}.yaml`.

**Тести.**
- Округлення: граничні значення, `Decimal`-точність, від'ємні й нульові значення відхиляються.
- `min_notional` / `min_qty` перевірки.
- State machine: кожен дозволений перехід проходить, кожен недозволений падає (параметризований тест по всій матриці).
- Конфігурація: валідні профілі завантажуються; невалідні (lower ≥ upper, відсутня out-of-range policy, leverage > max) відхиляються; секрети не з'являються у `repr` / `model_dump`; live без другого прапорця відхиляється.

**Критерії завершення.** Усі моделі типізовані, mypy strict без винятків; 100% гілок state machine покрито; конфігурація без секретів експортується в JSON.

**Залежності.** Phase 0.

---

## Phase 2 — Exchange Abstraction

**Мета.** Інтерфейси біржі, нормалізовані помилки і тестовий дубль, на якому можна будувати execution без реальної біржі.

**Стан.** Розпочато: async-контракти `MarketDataClient` / `AccountClient` / `TradingClient`, DTO `OrderRequest` / `OrderRef` / `OrderAck` і помилки `NotSent` / `Rejected` / `AmbiguousResult` — у `app/exchanges/`. Публічний market-data адаптер Bybit (`app/exchanges/bybit/`, REST V5 через `httpx`, USDT linear perpetual) — додано: бібліотеку для Bybit фактично обрано як власний тонкий клієнт поверх `httpx` (ADR ще не оформлено). Приватний REST-транспорт Bybit (HMAC-підпис, детерміновані GET/POST, межа `Ambiguous` для мутуючих POST) — додано, без бізнес-ендпоінтів. Чистий mapping `OrderRequest` → тіло Bybit `POST /v5/order/create` (`order_mapping.py`, локальна валідація → `ExchangeRequestValidationError`) — додано, без виклику транспорту. `TradingClient` / акаунт на Bybit, retry, rate limiter, `FakeExchange`, contract suite — наступні кроки фази. Геометрія Grid (`app/strategies/grid/levels.py`) лишається раннім ізольованим компонентом; решта Grid Strategy чекає своїх залежностей (Phase 5–6).

**Що реалізуємо.**
- Протоколи `MarketDataClient`, `AccountClient`, `TradingClient` (стріми — разом зі споживачами); DTO межі `OrderRequest`, `OrderRef`, `OrderAck`.
- Ієрархія помилок: `ExchangeNotSentError`, `ExchangeRejectedError` (підклас `ExchangeAuthenticationError`), `ExchangeAmbiguousResultError` — три сиблінги за відомим ефектом запиту (ARCHITECTURE 5.2).
- Спільні утиліти: retry-політика (exponential backoff + jitter, окремо для read / cancel; для create — без retry), token-bucket rate limiter, timeout-обгортка.
- `FakeExchange` — in-memory реалізація зі скриптованими сценаріями: timeout після прийняття ордера, reject, часткові fills, дублікати й зміна порядку подій, розрив стріму.
- **Contract test suite** — набір тестів поведінки адаптера, параметризований реалізацією.
- **Spike вибору бібліотеки для Bybit:** перевірити за актуальною документацією pybit, ccxt і варіант власного клієнта (asyncio, WS, `orderLinkId`, помилки, rate limits). Рішення записати в `docs/adr/0001-bybit-client.md`.

**Файли.** `app/exchanges/{base,errors,retry,rate_limit}.py`; `app/exchanges/fake/{exchange,scenarios}.py`; `tests/integration/contract/test_trading_adapter_contract.py`; `docs/adr/0001-bybit-client.md`.

**Тести.** Retry не повторює create; backoff зростає і обмежений; rate limiter тримає ліміт під навантаженням (з fake clock); contract suite проти `FakeExchange`.

**Критерії завершення.** Протоколи стабільні; contract suite зелений; ADR щодо бібліотеки прийнятий власником.

**Залежності.** Phase 1.

---

## Phase 3 — Bybit Public Market Data

**Мета.** Надійні публічні ринкові дані Bybit: REST і WebSocket, плюс завантаження історії для бектесту.

**Що реалізуємо.**
- REST: інструменти (`InstrumentSpec`), тікер, klines, історія funding, risk limit tiers.
- Public WS: тікер, orderbook L1, публічні трейди; reconnect з backoff; heartbeat; ресубскрипція після reconnect; валідація послідовності стакану (якщо біржа дає `seq` / `update_id`).
- `MarketDataFeed`, `MarketDataCache`, `StaleDataMonitor` (`data_age`, `last_update`, `connection_status`).
- Скрипт завантаження історії (klines, funding; трейди — якщо обрано) у локальне сховище (формат — Parquet або CSV.gz, рішення у фазі).
- Перевірка всіх використаних ендпоінтів і полів у документації; невідповідності — у ARCHITECTURE розділ 23.

**Файли.** `app/exchanges/bybit/{public_client,public_ws,mapping,endpoints}.py`; `app/market_data/{feed,cache,stale_monitor,replay}.py`; `scripts/download_history.py`; `tests/fixtures/bybit/*.json` (записані реальні відповіді).

**Тести.**
- Unit: mapping з фікстур реальних відповідей у доменні моделі (включно з рядковими числами → `Decimal`).
- Unit: stale-детекція з fake clock; перехід станів з'єднання.
- Unit: розрив послідовності стакану → запит snapshot.
- Integration: WS reconnect проти локального фейкового WS-сервера (розрив, мовчання без pong, некоректне повідомлення).
- Integration (ручний, з маркером `live_public`): короткий реальний запит до публічних ендпоінтів Bybit.

**Критерії завершення.** Фід працює безперервно ≥ 24 години з логуванням reconnect-ів і без витоку пам'яті; stale-детектор спрацьовує при штучному розриві; історія за обраний період завантажена і перевірена на пропуски.

**Залежності.** Phase 2.

---

## Phase 4 — Persistence

**Мета.** Надійне збереження стану й журналу.

**Що реалізуємо.**
- SQLAlchemy 2.x async, Alembic; SQLite (`aiosqlite`) і PostgreSQL (`asyncpg`).
- Таблиці з розділу 11 ARCHITECTURE; унікальні ключі (`client_order_id`, `exec_id`).
- Репозиторії: orders (+ order_events), fills, positions/balances snapshots, strategy_state (JSON + `schema_version`), grid_levels, bot/risk/reconciliation events, kill_switch_state.
- Транзакційний запис «ордер + подія переходу».

**Файли.** `app/persistence/{db,tables,unit_of_work}.py`; `app/persistence/repositories/*.py`; `migrations/`; `alembic.ini`.

**Тести.**
- CRUD на SQLite (in-memory і файл).
- Повторний insert того самого `exec_id` не створює дубль.
- Міграції: upgrade/downgrade на порожній БД.
- Той самий набір на PostgreSQL (у Docker, маркер `postgres`) — опційно, але обов'язково до фази 11.
- Відмова БД: репозиторій піднімає типізовану помилку, а не «ковтає» її.

**Критерії завершення.** Схема покриває все з `CLAUDE.md` розділ 23; секретів у схемі немає; міграції відтворювані.

**Залежності.** Phase 1.

---

## Phase 5 — Execution Engine

**Мета.** Контрольований життєвий цикл ордерів без дублів і втрати стану.

**Що реалізуємо.**
- `ClientOrderIdGenerator` (префікс бота, унікальність, персистентний лічильник).
- `OrderManager` (синхронний): застосування `OrderUpdate`, відкидання застарілих оновлень, дедуплікація fills, контроль монотонності `cum_filled_qty`.
- `ExecutionEngine`: write-ahead submit (NEW → persist → SUBMITTING → persist → send), обробка ack / reject / unknown / not sent, підтвердження статусу через стрім або REST, cancel flow (`CANCELING`).
- Резолвер `UNKNOWN` (запит за `client_order_id` з backoff, вікно, `FAILED` після вичерпання).
- Мінімальний `TradingEngine` loop (черга подій, single writer), поки що без стратегії — керування через тестові intents.

**Файли.** `app/execution/{client_ids,order_manager,engine,unknown_resolver}.py`; `app/services/trading_engine.py`.

**Тести (failure-сценарії з `CLAUDE.md` розділ 29):**
- timeout після відправки → `UNKNOWN` → ордер знайдено → правильний статус;
- timeout → ордер не знайдено → `FAILED`, повторної відправки немає;
- повторний intent на той самий рівень під час `UNKNOWN` не створює другий ордер;
- reject → `REJECTED`, без retry;
- частковий fill → `PARTIALLY_FILLED`, потім cancel → `CANCELED` з `filled_qty > 0`;
- fill приходить до ack; оновлення приходять у зворотному порядку; дубль fill;
- cancel на вже виконаний ордер (race) → `FILLED`;
- рестарт з ордером у `SUBMITTING` у БД → при старті статус резолвиться, а не відправляється знову;
- помилка БД перед відправкою → ордер не відправлено.

**Критерії завершення.** Усі сценарії зелені проти `FakeExchange`; жоден шлях коду не викликає `place_order` двічі для одного intent; у кожного переходу є запис в `order_events`.

**Залежності.** Phase 2, Phase 4.

---

## Phase 6 — Risk Manager & Kill Switch

**Мета.** Обов'язковий шар, що може зупинити торгівлю незалежно від стратегії.

**Що реалізуємо.**
- Pre-trade перевірки (tick/step/min, max order qty, max position з урахуванням активних і `UNKNOWN` ордерів, capital allocation, max open orders, max leverage, вільна маржа, max loss per trade).
- Portfolio-перевірки: exposure, max drawdown, daily loss limit, відстань до ліквідації.
- Системні: stale data, розрив private stream, частота API-помилок, частка відхилених ордерів.
- `TradingState` (RUNNING / REDUCE_ONLY / PAUSED / HALTED) з явною таблицею «порушення → стан» з конфігурації.
- `KillSwitch`: тригери (CLI, файл-прапорець, ліміти), дії (блок, cancel all, опційне закриття позицій reduce-only), персистентний латч, запис причини, виклик нотифікатора (поки заглушка-протокол).
- Оцінювач ліквідації для isolated margin (за формулою з документації Bybit, з tiers).
- Мінімальний `Portfolio` (позиція з fills, PnL, fees, funding) — потрібен для перевірок.

**Файли.** `app/risk/{manager,checks,limits,trading_state,kill_switch,liquidation}.py`; `app/portfolio/{portfolio,position_tracker,pnl}.py`; `scripts/kill_switch.py` (CLI).

**Тести.** Кожен ліміт: нижче / на межі / вище; RiskManager ніколи не збільшує qty; ордер у `UNKNOWN` враховано в експозиції; daily loss скидається на межі доби (UTC, з fake clock); kill switch скасовує ордери через `FakeExchange` навіть коли стратегія «зависла»; латч переживає рестарт; розрахунок ліквідації на прикладах, порахованих вручну (long / short, різні tiers); PnL портфеля з комісіями й funding.

**Критерії завершення.** Будь-яке критичне порушення → нові ордери зупинені (перевірено тестом); kill switch працює без участі стратегії.

**Залежності.** Phase 5.

---

## Phase 7 — Grid Strategy

**Мета.** Grid Long / Short / Neutral з чесним розрахунком прибутковості й ризиків.

**Що реалізуємо.**
- Протокол `Strategy` і `StrategyContext`.
- Розрахунок рівнів: arithmetic і geometric; округлення до tick; перевірка, що після округлення рівні не злилися.
- Розмір ордера: фіксований qty або з capital allocation; перевірка min_qty / min_notional на **кожному** рівні.
- `GridProfitability`: gross per cycle, fees (maker/taker), slippage, spread, funding за очікуваний час утримання, net per cycle, breakeven spacing. Вердикт OK / WARN / BLOCK за налаштуваннями ризику.
- Аналіз стартової ціни: положення в діапазоні (нижня / центр / верхня третина) і попередження для несприятливого старту залежно від режиму.
- Worst-case аналіз: позиція при проходженні всіх рівнів до межі, маржа, оцінка ліквідації, відстань від межі діапазону.
- Логіка: початкове розміщення за режимом; на fill — парний ордер на сусідньому рівні; часткові fills (парний ордер на виконану частину або після повного виконання — рішення фіксується в коді й тестах).
- `out_of_range_policy` окремо для нижньої та верхньої межі; усі політики з `CLAUDE.md`.
- `snapshot_state` / `restore_state` + метод «бажаний набір ордерів» для відновлення.

**Стан.** Геометрію рівнів (`levels.py`, arithmetic / geometric, без округлення до tick) реалізовано раніше за порядком фаз: це чиста функція без залежностей від execution / risk. Логіка рішень, розмір ордера, прибутковість і out-of-range лишаються в цій фазі.

**Файли.** `app/strategies/base.py`; `app/strategies/grid/{config,levels,sizing,profitability,start_analysis,worst_case,strategy,out_of_range,state}.py`; `scripts/grid_preview.py` (друк аналізу сітки для конфігурації без торгівлі).

**Тести.** Рівні (обидва типи, граничні значення, злиття після округлення); прибутковість на прикладах, порахованих вручну; net < 0 → BLOCK; worst-case для long / short / neutral; реакція на fill у кожному режимі; частковий fill; кожна out-of-range політика; restore зі снапшоту дає ті самі бажані ордери; стратегія не імпортує нічого поза `domain` (тест архітектури).

**Критерії завершення.** `grid_preview` для реальних параметрів друкує повний звіт (gross / fees / funding / slippage / net, worst-case, ліквідація); логіка покрита тестами; жодних викликів біржі зі стратегії.

**Залежності.** Phase 6.

---

## Phase 8 — Backtesting Engine

**Мета.** Відтворюваний бектест тим самим кодом Strategy / Risk / Execution / Portfolio.

**Стан.** Розпочато фундамент симуляції (раніше за порядком фаз, як ізольований компонент без залежностей від execution / risk): детермінований `SimulatedExchange` (`app/exchanges/simulated.py`) — lifecycle ордера (place LIMIT GTC / POST_ONLY → OPEN, cancel, get, open orders), ідемпотентність за `client_order_id`, час з `Clock`, детерміновані повні fills LIMIT-ордерів за зовнішньою ціною виконання (`fill_crossed_limit_orders`, атомарний batch, комісія й роль ліквідності — `None`). Partial fills, market-ордери, IOC / FOK, reduce-only, комісії, slippage, funding, баланси й позиції — ще ні. Модуль стане пакетом `simulated/`, коли з'являться matching / fees тощо. Backtest і paper trading **не** реалізовані.

**Що реалізуємо.**
- `SimulatedExchange` (реалізує `TradingClient` + приватний стрім): matching лімітних і ринкових ордерів, post-only, reduce-only, комісії, slippage, funding, ліквідація, опційна латентність, опційні часткові fills.
- `ReplayFeed` + `SimulatedClock` з assert монотонності часу.
- `BacktestRunner`: збирання тих самих компонентів через `services/bootstrap` з режимом `backtest`.
- Метрики з `CLAUDE.md` розділ 18 + Grid-метрики (цикли, inventory, unrealized на кінець, max position, min liquidation distance, час поза діапазоном).
- Звіт: консольний + файл (JSON/CSV equity curve; HTML/PNG графік — опційно).
- Скрипт sensitivity-sweep параметрів і розбивка на періоди (in-sample / out-of-sample, різні ринкові режими).

**Файли.** `app/exchanges/simulated/{exchange,matching,fees,funding,liquidation,latency}.py`; `app/backtesting/{runner,data_loader,metrics,report,sweep}.py`; `scripts/run_backtest.py`.

**Тести.**
- Look-ahead: стратегія ніколи не отримує подію з часом > `clock.now()`; ордер не виконується на барі, де його розмістили, якщо шлях OHLC цього не дозволяє.
- Детермінованість: два прогони → ідентичний результат.
- Бухгалтерія: `net = gross − fees ± funding`, баланс кінця = початок + net (перевірка інваріанту).
- Ручні сценарії з відомим PnL (кілька барів, 2–3 рівні).
- Funding: знак для long / short при додатній і від'ємній ставці.
- Ліквідація спрацьовує при різкому русі.

**Критерії завершення.** Звіт з усіма метриками на завантаженій історії; sweep показує чутливість до параметрів; результати з різних режимів ринку й out-of-sample задокументовані в `docs/research/` без тверджень про гарантовану прибутковість. Рішення, чи переходити до paper, приймає власник на основі звіту.

**Залежності.** Phase 7, Phase 3 (історичні дані).

---

## Phase 9 — Paper Trading

**Мета.** Стратегія на живому ринку в реальному часі без реальних ордерів.

**Що реалізуємо.**
- Режим `paper` у bootstrap: mainnet public WS + `SimulatedExchange` у real time (виконання за потоком публічних трейдів) + `SystemClock` + SQLite.
- Запуск як довготривалого процесу, graceful shutdown (SIGINT/SIGTERM: зупинка нових ордерів, збереження стану).
- Порівняння: paper-результати проти бектесту на тому самому періоді (реплей записаного потоку).
- Відоме обмеження до фази 10: рестарт paper-процесу починає симуляцію наново.

**Файли.** `app/paper_trading/{runner,realtime_sim_bridge}.py`; `app/services/bootstrap.py` (режим paper); `scripts/run_paper.py`; `scripts/record_market_stream.py` (запис потоку для реплею).

**Тести.** Integration: реплей записаного WS-потоку через paper-зв'язку дає той самий результат, що backtest на тих самих даних; stale data під час paper → REDUCE_ONLY; розрив WS → reconnect без дублювання подій.

**Критерії завершення.** Безперервна робота щонайменше 2 тижні (рекомендовано; фінальний строк визначає власник) без необроблених винятків; поведінка сітки в логах відповідає очікуваній; розбіжність paper vs бектест на тих самих даних пояснена.

**Залежності.** Phase 8, Phase 3.

---

## Phase 10 — Recovery & Reconciliation

**Мета.** Бот коректно відновлюється після будь-якого рестарту і не торгує, якщо його стан не збігається з біржею.

**Що реалізуємо.**
- Послідовність запуску з розділу 12 ARCHITECTURE (буферизація private stream під час снапшоту).
- `Reconciler`: порівняння ордерів, позицій, балансу, fills; таблиця реакцій з розділу 13 ARCHITECTURE; `ReconciliationEvent`.
- Тригери: старт, reconnect private stream, періодичний, після `FAILED` з `UNKNOWN`, ручний (CLI).
- Персистентний стан `SimulatedExchange` для paper, щоб recovery перевірялося на paper до testnet.
- Відновлення стану Grid з `strategy_state` + звірка з фактичними ордерами.

**Файли.** `app/services/{recovery,reconciliation}.py`; `app/paper_trading/sim_state_store.py`; `scripts/reconcile.py`.

**Тести (матриця сценаріїв).** Пропущений fill під час простою; ордер скасований зовні; сторонній ордер на символі; наш ордер, якого немає локально; позиція не пояснюється fills → PAUSED; `UNKNOWN` у БД при старті; розрив private stream і пропущені події → resync; `kill -9` у момент між SUBMITTING і ack; kill switch у HALTED переживає рестарт.

**Критерії завершення.** На paper: серія примусових рестартів (включно з `kill -9`) у випадкові моменти не приводить ні до дублів, ні до невиявлених розбіжностей; усі розбіжності мають `ReconciliationEvent`.

**Залежності.** Phase 9.

---

## Phase 11 — Bybit Private Adapter & Testnet

**Мета.** Реальна торгова інтеграція з Bybit, перевірена на testnet.

**Що реалізуємо.**
- Auth (підпис запитів за документацією v5), private REST: баланс, позиції, open orders, get order, fills, fee rates, account config, API key info, set leverage, create / cancel / cancel all.
- Private WS: order, execution, position, wallet; auth, heartbeat, reconnect → тригер reconciliation.
- Mapping статусів і помилок Bybit на доменні (на основі перевірених кодів помилок).
- Preflight для testnet: ключі, права, тип акаунта, position/margin mode, плече, баланс.
- Contract test suite проти testnet (ручний запуск, маркер `testnet`).
- Postgres як основна БД для довгих прогонів.

**Файли.** `app/exchanges/bybit/{auth,private_client,private_ws,status_mapping,error_mapping}.py`; `app/services/preflight.py`; `scripts/run_testnet.py`.

**Тести.** Unit: підпис на тестовому векторі з документації (якщо є) або на зафіксованому прикладі; mapping з записаних відповідей; секрети не потрапляють у логи запитів. Testnet (ручні): повний contract suite; post-only, що перетинає стакан; скасування неіснуючого ордера; дублікат `orderLinkId`; частковий fill; втрата з'єднання під час submit (штучний timeout); kill switch drill.

**Критерії завершення.** Testnet-прогін щонайменше 1 тиждень (рекомендовано) з рестартами, reconnect-ами і kill switch drill; нуль дублів; усі розбіжності пояснені; всі непідтверджені припущення з розділу 23 ARCHITECTURE закриті або задокументовані.

**Залежності.** Phase 10.

---

## Phase 12 — Monitoring & Notifications

**Мета.** Власник бачить стан бота і дізнається про проблеми, не читаючи логи.

**Що реалізуємо.**
- `HealthMonitor`: WS status, data freshness, API latency / errors, rejection rate, open orders, TradingState.
- Метрики портфеля: realized / unrealized PnL, drawdown, exposure, margin usage, liquidation distance; періодичні снапшоти в БД.
- CLI `status`.
- `NotificationDispatcher` (bounded queue, timeout, throttle, dedupe) + `TelegramNotifier` (тільки вихідний канал); агрегація fills.

**Файли.** `app/monitoring/{health,metrics,status}.py`; `app/notifications/{base,dispatcher,telegram,formatting}.py`; `scripts/status.py`.

**Тести.** Недоступний Telegram / timeout / переповнена черга не впливають на торговий цикл (тест з повільним фейковим нотифікатором); throttle і dedupe; повідомлення не містять секретів; health переходить у degraded при stale data.

**Критерії завершення.** Усі події зі списку `CLAUDE.md` розділ 27 доходять у Telegram на testnet; вимкнення Telegram не зупиняє бота.

**Залежності.** Phase 11.

---

## Phase 13 — Live Trading Readiness

**Мета.** Усе, що має бути готове до першого реального ордера.

**Що реалізуємо.**
- Повний live preflight (розділ 16.3 ARCHITECTURE), включно з перевіркою, що виведення вимкнене.
- Профіль `production.yaml` з мінімальним capital allocation і консервативними лімітами.
- Deployment: Docker або systemd, автоперезапуск, синхронізація часу (NTP), PostgreSQL з бекапами, ротація логів.
- `docs/RUNBOOK.md`: запуск, зупинка, kill switch, дії при розбіжності, ротація ключів, відновлення з бекапу.
- `docs/GO_LIVE_CHECKLIST.md`.

**Файли.** `app/services/preflight.py` (live), `configs/production.yaml`, `docker/`, `docs/RUNBOOK.md`, `docs/GO_LIVE_CHECKLIST.md`.

**Тести.** Кожна умова preflight: окремий тест, що без неї live не стартує; live-режим з testnet-ключами / URL відхиляється; один прапорець без другого відхиляється.

**Критерії завершення.** Чекліст виконаний і підписаний власником; kill switch drill на testnet у production-конфігурації пройдено; ключі mainnet створено з мінімальними правами на виділеному субакаунті.

**Залежності.** Phase 12.

---

## Phase 14 — Live Trading (поетапно)

**Мета.** Обережний перехід на реальні гроші зі збільшенням капіталу тільки на основі даних.

**Етапи.**
1. **Мінімальний капітал** (сума, втрату якої власник готовий прийняти повністю), мінімальний розмір ордерів, кілька рівнів. Щоденний перегляд логів і reconciliation.
2. **Збільшення капіталу** — тільки якщо на етапі 1: нуль невирішених розбіжностей, нуль дублів, поведінка відповідає paper / testnet, фактичні комісії й funding збігаються з моделлю.
3. Кожна зміна параметрів стратегії повертається через бектест → paper перед live.

**Критерії зупинки (повернення на попередній етап).** Будь-яка невирішена розбіжність reconciliation, дубль ордера, спрацювання kill switch з невідомої причини, фактичні витрати суттєво гірші за модель.

**Тести.** Повний набір перед кожним релізом; реліз у live тільки з тегом у git і записом у `bot_events` (версія, хеш конфігурації).

**Залежності.** Phase 13.

---

## Що свідомо відкладено після v1

- Binance, OKX (нові реалізації протоколів з фази 2; contract suite вже є).
- Нові стратегії (DCA, Trend, Mean Reversion, Breakout) та модель `Signal` + position sizer.
- Кілька інстансів стратегії / символів одночасно.
- Prometheus / Grafana.
- Hedge mode.
