# Retirement Planner — Epics & Tickets (Flask/Python)

Conventions

* **ID format**: `EP-<epic#>-T<ticket#>`
* **Estimate**: S (≤0.5d), M (1–2d), L (3–5d)
* **Tests**: Each ticket includes concrete, automatable acceptance tests (pytest where applicable).
* **Deps**: Explicit dependencies to preserve parallelism.

---

## EP-1: Project Scaffolding & Infrastructure

**Goal**: Production-ready Python/Flask skeleton with testing, linting, type checks, and env management.

* **EP-1-T1**: Create repo scaffold (Flask app factory, blueprints, config) — **S**
  *AC*: `flask run` serves health endpoint `/healthz` returning `{status:"ok"}`.
  *Tests*: `test_healthz_ok()` asserts 200 JSON.
  *Deps*: None.

* **EP-1-T2**: Add tooling (pytest, coverage, black, isort, flake8/ruff, mypy) — **S**
  *AC*: `make test lint typecheck` run locally and in CI; coverage ≥80% on existing tests.
  *Tests*: CI passes with status badges.
  *Deps*: EP-1-T1.

* **EP-1-T3**: Config management via `.env` and Pydantic Settings — **S**
  *AC*: `APP_ENV`, `SECRET_KEY`, `DB_URL` injected; missing `SECRET_KEY` fails startup.
  *Tests*: unit test loads `.env.example`; missing key triggers exception.

* **EP-1-T4**: Makefile & pre-commit hooks — **S**
  *AC*: Pre-commit auto-runs format/lint; Makefile targets documented.
  *Tests*: Hook runs on sample commit (CI check).

* **EP-1-T5**: Dockerfile & dev docker-compose — **M**
  *AC*: `docker compose up` serves app + Postgres + Redis.
  *Tests*: curl `/healthz` inside container; DB migration runs.

---

## EP-2: Data Model & Persistence

**Goal**: Versioned scenario schema + DB with migrations.

* **EP-2-T1**: Define JSON schema v0.1 for Scenario — **M**
  *AC*: `schema/scenario_v0_1.json` validates sample; includes household, accounts, liabilities, incomes, expenses, policies, market\_model, strategy.
  *Tests*: jsonschema validation on fixtures passes.

* **EP-2-T2**: SQLAlchemy models (User, Scenario, Run, LedgerRow) — **M**
  *AC*: Alembic migration creates tables; relations and indexes present.
  *Tests*: CRUD tests; foreign key constraints enforced.

* **EP-2-T3**: Scenario versioning & immutable base+diff — **M**
  *AC*: Store base scenario + override patch; materialize full scenario view.
  *Tests*: patch apply yields expected materialized JSON.

* **EP-2-T4**: S3/local storage for exports & run artifacts — **S**
  *AC*: Save/retrieve CSV/PDF from storage backend.
  *Tests*: round‑trip file test using local MinIO/dev folder.

---

## EP-3: Deterministic Engine

**Goal**: Cashflow projection under fixed return & inflation with taxes stubbed.

* **EP-3-T1**: Time grid & unit system (annual; real vs nominal) — **S**
  *AC*: Switch toggles real/nominal outputs.
  *Tests*: Known inputs produce expected inflation-adjusted results.

* **EP-3-T2**: Mortgage amortization module — **M**
  *AC*: Given principal/rate/term/start, compute schedule, interest/principal splits, extra payments.
  *Tests*: Validate against closed-form payment and rounding rules.

* **EP-3-T3**: Baseline expenses & lumpy events — **S**
  *AC*: Supports categories + one‑offs by year.
  *Tests*: Cashflow ledger matches fixture.

* **EP-3-T4**: Account balance evolution (taxable/trad/Roth) — **M**
  *AC*: Contributions, withdrawals, growth at deterministic rate.
  *Tests*: Reproducible ledger; balances match formulas.

* **EP-3-T5**: Social Security stub (fixed input benefit) — **S**
  *AC*: Inject annual benefit starting at claim age.
  *Tests*: Benefit schedule aligns with claim parameters.

---

## EP-4: Monte Carlo Engine

**Goal**: Stochastic simulation with multivariate sampling and rebalancing.

* **EP-4-T1**: Random returns generator (normal/lognormal selectable) — **M**
  *AC*: Deterministic seeding; shape (assets × years × paths).
  *Tests*: Mean/vol near target within tolerance over large N.

* **EP-4-T2**: Correlated draws via covariance (Cholesky) — **M**
  *AC*: Sampled correlations within ±0.05 of target for 10k paths.
  *Tests*: Stats test for correlation matrix.

* **EP-4-T3**: Portfolio evolution w/ annual rebalance — **M**
  *AC*: Drift and rebalance logic; transaction/tax drag flags (no‑op for now).
  *Tests*: Rebalance restores target weights within epsilon.

* **EP-4-T4**: Withdrawal rules (4% real, fixed %, VPW scaffold) — **M**
  *AC*: Apply by path/year; track failures.
  *Tests*: Known toy cases (no volatility) reproduce deterministic results.

* **EP-4-T5**: Success metrics & percentiles — **S**
  *AC*: Report success rate, p5/p50/p95 terminal wealth, first failure year distribution.
  *Tests*: Fixtures verified.

---

## EP-5: Historical Bootstrapping Engine (V2-ready)

**Goal**: Resample realized returns to capture fat tails and regimes.

* **EP-5-T1**: Data loader for historical series (CSV) — **S**
  *AC*: Load monthly/annual asset class series; forward-fill/align.
  *Tests*: Schema validation; NA handling.

* **EP-5-T2**: Block bootstrap sampler (configurable block length) — **M**
  *AC*: Generates sequences preserving serial correlation.
  *Tests*: Autocorr of samples \~ source over lags.

* **EP-5-T3**: Integrate with portfolio evolution & metrics — **S**
  *AC*: Reuse EP‑4 logic; switch engine via config.
  *Tests*: End‑to‑end run produces metrics JSON.

---

## EP-6: Tax Engine (Federal + MA template)

**Goal**: Yearly tax calculation sufficient for planning.

* **EP-6-T1**: Federal brackets, standard vs itemized, filing statuses — **M**
  *AC*: Computes ordinary income tax from taxable income.
  *Tests*: IRS examples parity within \$1.

* **EP-6-T2**: Capital gains/dividends (qualified vs ordinary) — **M**
  *AC*: Tiered LTCG/NIIT threshold application.
  *Tests*: Threshold fixtures incl. MFJ/Single.

* **EP-6-T3**: Retirement account rules (deductibility, RMDs) — **M**
  *AC*: RMD table by age; penalty if missed.
  *Tests*: Known RMD factors.

* **EP-6-T4**: State tax plugin (implement MA) — **S**
  *AC*: MA flat rate + surtaxes as applicable.
  *Tests*: Sample returns validated.

* **EP-6-T5**: Withholding/est. payments & refund model — **S**
  *AC*: Compute net tax due/refund per year.
  *Tests*: Ledger reconciliation.

---

## EP-7: Withdrawal & Strategy Lab

**Goal**: Multiple spending rules + guardrails.

* **EP-7-T1**: Implement Fixed Real (4% rule) — **S**
  *AC*: Year‑1 % of initial, then inflate.
  *Tests*: Matches deterministic baseline.

* **EP-7-T2**: Fixed % of balance — **S**
  *AC*: Spend = s% × current balance.
  *Tests*: Toy case proofs.

* **EP-7-T3**: VPW (age‑based %) — **M**
  *AC*: Table-driven %; declines with horizon.
  *Tests*: Monotonicity checks.

* **EP-7-T4**: Guyton‑Klinger guardrails — **M**
  *AC*: COLA raise/cut with bands + decision rules.
  *Tests*: Scenario fixtures (trigger/not trigger cases).

* **EP-7-T5**: Tax‑aware account sequencing — **M**
  *AC*: Order: cash → taxable gains‑harvested → traditional up to bracket → Roth; configurable.
  *Tests*: Tax minimization vs naive.

---

## EP-8: Housing & Mortgage Module

**Goal**: Full mortgage lifecycle & what‑ifs.

* **EP-8-T1**: PMI & escrow modeling — **S**
  *AC*: PMI drops when LTV < threshold; escrow cashflows.
  *Tests*: LTV‑triggered PMI removal.

* **EP-8-T2**: Recast/refinance scenarios — **M**
  *AC*: New payment calc; closing cost amortization.
  *Tests*: Payment change correctness.

* **EP-8-T3**: Prepay vs invest analyzer — **M**
  *AC*: Compare NPV / success% across strategies.
  *Tests*: Sensitivity harness.

---

## EP-9: Kids & College Module

**Goal**: Child timeline, childcare, 529s, college costs.

* **EP-9-T1**: Child objects & timelines — **S**
  *AC*: Start/end of childcare & college years.
  *Tests*: Schedule generation.

* **EP-9-T2**: 529 account rules — **M**
  *AC*: Contributions, growth, qualified withdrawals, penalties.
  *Tests*: Penalty vs qualified cases.

* **EP-9-T3**: College cost curves & inflation — **M**
  *AC*: Public/private presets; custom inflator.
  *Tests*: Cost table generation.

---

## EP-10: Benefits Module (Social Security)

**Goal**: SSA benefit estimator & claiming strategies.

* **EP-10-T1**: PIA calculator from AIME inputs — **M**
  *AC*: Bend points, indexing, early/late factors.
  *Tests*: SSA sample parity within tolerance.

* **EP-10-T2**: Claiming strategy compare — **S**
  *AC*: 62 vs FRA vs 70; survivor options.
  *Tests*: Higher lifetime PV in expected cases.

---

## EP-11: Flask API & Services

**Goal**: Clean REST API for scenarios, runs, and results.

* **EP-11-T1**: Auth (session or token) + RBAC — **M**
  *AC*: Login, user roles; protected endpoints.
  *Tests*: Authz tests.

* **EP-11-T2**: CRUD endpoints for Scenario — **S**
  *AC*: POST/GET/PATCH/DELETE with jsonschema validation.
  *Tests*: API tests incl. invalid schema.

* **EP-11-T3**: Run orchestration endpoint — **M**
  *AC*: `/runs` triggers engine with config; returns run\_id; polling endpoint for status/results.
  *Tests*: E2E with small Monte Carlo run.

* **EP-11-T4**: Export endpoints (CSV ledger, PDF report stub) — **S**
  *AC*: Download artifacts by run\_id.
  *Tests*: File content checks.

---

## EP-12: Minimal UI (server-rendered or SPA shell)

**Goal**: Usable front end to toggle scenarios and compare results.

* **EP-12-T1**: Scenario builder form (Quick Start) — **M**
  *AC*: 10 key inputs; client validation; submit to API.
  *Tests*: Cypress/Playwright happy-path.

* **EP-12-T2**: Scenario list + clone/diff — **M**
  *AC*: Cards with KPIs; clone; textual diff of JSON.
  *Tests*: UI diff reflects changes.

* **EP-12-T3**: Results dashboard — **M**
  *AC*: Charts: net worth, success %, percentiles; table of KPIs.
  *Tests*: Chart renders with mocked data.

* **EP-12-T4**: Compare view (A/B) — **S**
  *AC*: Side‑by‑side KPIs & charts sync.
  *Tests*: Correct scenario labels & values.

---

## EP-13: Scenario Manager & Explainability

**Goal**: Named scenarios, deltas, and “why” panels.

* **EP-13-T1**: Scenario tagging & notes — **S**
  *AC*: Tags, markdown notes saved.
  *Tests*: CRUD tests.

* **EP-13-T2**: Explainability API (provenance of numbers) — **M**
  *AC*: Endpoint returns breakdown for any year: income sources, taxes, withdrawals.
  *Tests*: Snapshot test for a known year.

---

## EP-14: Reporting & Export

**Goal**: Professional PDF/CSV outputs.

* **EP-14-T1**: CSV ledger exporter — **S**
  *AC*: One row per year per account/cashflow; documented columns.
  *Tests*: Schema test; sample diff‑free.

* **EP-14-T2**: PDF report (WeasyPrint/ReportLab) — **M**
  *AC*: Title page, assumptions, charts, conclusions.
  *Tests*: Golden PDF checksum w/ deterministic seed.

---

## EP-15: Testing & Quality Harness

**Goal**: Confidence via property tests and golden datasets.

* **EP-15-T1**: Property tests for engines (hypothesis) — **M**
  *AC*: Invariants (no negative balances unless allowed, accounting identities).
  *Tests*: Hypothesis suite passes.

* **EP-15-T2**: Golden scenario fixtures — **S**
  *AC*: Two fixed scenarios with saved outputs; detect regressions.
  *Tests*: Snapshot comparisons.

* **EP-15-T3**: Performance budget tests — **S**
  *AC*: 1k‑path MC ≤ 2s on laptop baseline.
  *Tests*: Perf test gate in CI.

---

## EP-16: DevOps, CI/CD & Security

**Goal**: Ship safely and repeatably.

* **EP-16-T1**: GitHub Actions CI (test/lint/type/perf) — **S**
  *AC*: Matrix on py3.10–3.12; status checks required.
  *Tests*: Pipeline green.

* **EP-16-T2**: Containerized deploy (Render/Fly/Heroku) — **M**
  *AC*: One‑click deploy; secrets via env.
  *Tests*: Smoke test post‑deploy.

* **EP-16-T3**: Basic security hardening — **S**
  *AC*: Flask Talisman/helmet‑like headers, CSRF, rate limit, secure cookies.
  *Tests*: OWASP header checks; CSRF test.

* **EP-16-T4**: Error monitoring & logging — **S**
  *AC*: Structured logs, request IDs, Sentry integration.
  *Tests*: Error captured in mock Sentry.

---

## Milestones (Suggested Order)

1. EP‑1, EP‑2, EP‑3 (deterministic backbone)
2. EP‑4 (Monte Carlo)
3. EP‑11, EP‑12 (API + minimal UI)
4. EP‑6, EP‑7 (tax + strategies)
5. EP‑8, EP‑9, EP‑10 (domain modules)
6. EP‑13, EP‑14, EP‑15, EP‑16 (UX polish, outputs, quality, ops)

---

## Labels

* `backend`, `engine`, `tax`, `api`, `ui`, `infra`, `security`, `perf`, `reporting`, `college`, `mortgage`, `benefits`, `strategy`, `devops`
