# Retirement Planner App — System Design Blueprint

## 0) Why this app

A flexible, modular planner that works with sparse inputs (quick estimate) or rich, tax‑aware, multi‑account detail (power users). Users can compare strategies side‑by‑side (deterministic, Monte Carlo, historical bootstrap, regime models) and toggle real‑life components (mortgage, kids, college, etc.).

---

## 1) User personas & goals

* **Quick Planner**: 5‑minute estimate, minimal inputs, default assumptions.
* **Tinkerers/Power Users**: Deep scenario control, tax engines, account locations, glidepaths.
* **Advisors**: Client profiles, compliance audit trail, printable reports.

Primary goals:

1. Probability of success (portfolio longevity / goal coverage).
2. Spending sustainability & lifestyle guardrails.
3. Tax‑efficient funding & withdrawal sequencing.
4. Clear trade‑offs across scenarios.

---

## 2) Core concepts (data model primitives)

* **Person**: birth year, sex, longevity model, retirement age, benefits eligibility.
* **Household**: filing status, dependents, state, healthcare context.
* **Accounts**: taxable, traditional, Roth, HSA, 529, cash; holdings with asset class or tickers.
* **Liabilities**: mortgages (rate, term, amortization schedule), HELOC, student loans.
* **Incomes**: wages, RSUs, business, pensions, Social Security, rental, annuities.
* **Expenses**: baseline (by category), one‑offs, lumpy (childcare, college, weddings), healthcare/LTC.
* **Policies & Rules**: tax brackets (federal/state), SS formulas, RMD tables, contribution limits.
* **Market Models**: deterministic trend, Monte Carlo, bootstrap, regime‑switching, fat‑tail.
* **Strategies**: asset allocation/glidepath, rebalancing, withdrawal rules, Roth conversions, asset location.
* **Scenarios**: named bundles of assumptions & toggles; comparable side‑by‑side.

---

## 3) Input hierarchy (progressive disclosure)

Each input has:

* **Value Score** (utility/impact): High, Medium, Low.
* **Default Source**: conservative default, public table, or user estimate.
* **Sensitivity**: which outputs it most affects.

**High value**: Age/retirement date, savings rate, spending level, asset mix, tax state, mortgage terms, major child/college timelines, SS start ages.
**Medium**: Expense breakdown, pension details, employer match, RSU cadence, healthcare premiums, LTC assumptions, property tax trajectory.
**Low**: Minor expense categories, small side incomes, charitable timing.

UI: Start with a 10‑question **Quick Start**; then unlock **Advanced** tabs with optional depth.

---

## 4) Major modules

1. **Profile & Timeline Builder**

   * Life events: retire, kids, buy/sell house, sabbatical, college, inheritances, downsizing.
   * Visual timeline editor with drag‑drop events.

2. **Cash‑Flow Engine**

   * Annual (or monthly) ledger of incomes, taxes, contributions, expenses, loan payments.
   * Mortgage amortization; refi/recast; extra principal; home sale equity flow.

3. **Tax Engine**

   * Federal + state brackets; standard/itemized; AMT; NIIT; QBI; cap gains; wash sales.
   * Account types: Taxable/Traditional/Roth/HSA; contribution limits & catch‑ups.
   * RMDs; IRMAA checks for Medicare; ACA subsidies for pre‑65 retirements.
   * Roth conversion optimizer & bracket‑filling.

4. **Benefits Engine**

   * Social Security benefit calculator and claiming strategies (per person); pension reductions; annuitization options; QCDs from IRAs.

5. **Investment & Portfolio Engine**

   * Asset classes, expected returns, volatilities, correlations.
   * Glidepaths; rebalancing rules (threshold, time‑based, or none).
   * Asset location (which assets in which accounts) & tax drag modeling.

6. **Simulation Engine**

   * Deterministic path (real & nominal views).
   * Monte Carlo (multivariate normal/lognormal/t‑distribution, block bootstrap).
   * Historical bootstrapping (rolling windows or block resampling).
   * Regime‑switching / stress tests (stagflation 70s, dot‑com, GFC, 2022‑style inflation shock).
   * Sequence‑of‑returns risk analytics.

7. **Withdrawal & Spending Strategy Lab**

   * Rules: 4% + inflation, fixed % of balance, VPW, Guyton‑Klinger guardrails, floor‑and‑upside, RMD‑based, tax‑aware sequencing.
   * Dynamic COLA toggles and guardrails (raise/cut if success probability drifts).

8. **College & Kids Module**

   * Child timeline; childcare costs; K‑12 options; 529 funding/withdrawals; FAFSA/CSS heuristics; room & board vs tuition; 529→Roth eligibility checks.

9. **Housing Module**

   * Mortgage(s), HELOC, property taxes, maintenance %, insurance; move/downsizing; reverse mortgage explorer.

10. **Healthcare & LTC Module**

* Pre‑65 marketplace premiums + subsidies; Medicare Parts B/D; Medigap; IRMAA; LTC scenarios (self‑insure vs insurance).

11. **Scenario Manager**

* Named scenarios; clone; diff; tag; side‑by‑side comparison view; A/B/C deck.

12. **Reporting & Explainability**

* Success probability, safe spending bands, shortfall years, tax by source, where each \$ came from.
* Print‑ready PDF; audit trail of assumptions; narrative summary.

13. **Integrations** (optional)

* Aggregators (Plaid/Yodlee), SSA statement import, broker CSVs, tax table updates.

14. **Governance**

* Assumptions library; versioning; compliance notes; unit tests for calculators.

---

## 5) Strategy toggles & comparisons

* **Global toggles**: Healthcare model, college on/off, mortgage refi/recast/extra paydown, annuity purchase, home sale.
* **Portfolio toggles**: Static AA vs glidepath; include alternatives (REITs, TIPS, international); rebalance on/off.
* **Withdrawal toggles**: Method, guardrail thresholds, minimum floor, COLA logic.
* **Tax toggles**: Roth conversions years/amounts; itemize vs standard; capital‑gain harvesting; asset location.
* **Risk toggles**: Distribution family (normal/lognormal/t‑tail), bootstrap block length, regime sampler, fat‑tail stress.

**Compare View**: Grid of scenarios with KPIs: success %, median terminal wealth, 5th percentile terminal wealth, worst drawdown, first failure year, average tax paid, IRMAA hits, college fully funded Y/N, mortgage payoff year.

---

## 6) Outputs & visualizations (key charts)

* Net worth over time (real & nominal).
* Annual cash‑flow waterfall (income → taxes → spending → savings → residual).
* Account balances by type; tax‑deferred vs Roth vs taxable.
* Spending band & guardrails; failure‑year heatmap across simulations.
* Tax breakdown by source (ordinary, cap gains, NIIT, payroll, state);
* Roth conversion impact plot; IRMAA/ACA cliffs visualization.
* College funding coverage chart; mortgage amortization & prepayment savings.

---

## 7) Modeling details

### 7.1 Deterministic

* Real and nominal modes. Inflators: CPI paths; wage growth; college healthcare inflators.

### 7.2 Monte Carlo

* Multivariate sampling by asset class using covariance matrix; Cholesky or PCA.
* Distribution options: normal, lognormal, Student‑t (df configurable), skew‑t.
* Serial correlation option; block bootstrap for fat‑tail persistence (e.g., length 12–36 months).
* Rebalancing each step; transaction/tax drag optional.
* Success metric: solvency through horizon; also utility‑based metrics.

### 7.3 Historical Bootstrapping

* Rolling‑window sequences; block bootstrap to preserve regimes.
* Alternative: pick random starting years from history; splice with present yields for bonds.

### 7.4 Regime/Stress

* Hand‑crafted stress paths (1973–74 bear, 2000–02 + 2008 combo, 2022 inflation shock);
* Markov‑switching returns with calibrated transition matrix.

---

## 8) Withdrawal strategies (implementations)

* **Fixed real (4% rule)**: Year 1 = 4% of initial; then inflate.
* **Fixed % of balance**: spend rate s(t) \* balance(t).
* **VPW** (variable percentage withdrawal): age‑based percentage derived from remaining horizon & expected return.
* **Guardrails (Guyton‑Klinger)**: COLA raise/cut based on bands vs initial rule.
* **Floor‑and‑Upside**: Floor from safe assets/annuities + risky upside bucket.
* **Tax‑aware sequencing**: ordering by account type; bracket‑filling; cap‑gain harvesting.
* **RMD‑linked**: follow RMD %, optionally smoothed.

---

## 9) Taxes (key considerations)

* Filing status; brackets; standard vs itemized; SALT caps; AMT; NIIT 3.8%; QBI.
* Capital gains (short/long), basis tracking, lot selection; dividends (qualified vs ordinary).
* IRA/401k contributions & catch‑ups; Roth vs Traditional decision logic.
* **RMDs** and penalties; **IRMAA** thresholds; ACA subsidy cliffs.
* **Roth conversions**: bracket‑fill optimizer; state taxes; Medicare impact.
* QCDs from IRAs; charitable bunching; donor‑advised fund what‑ifs.

---

## 10) Kids & college

* Childcare cost curves; tax credits (CDCTC/CTC heuristics).
* 529 contributions/returns; state deductions; qualified expense modeling; room/board rules.
* College cost inflation; 4‑year vs 5‑year scenarios; public vs private; merit/need heuristics.

---

## 11) Housing & mortgages

* Amortization; PMI; escrow; property tax trajectories; maintenance % of home value.
* Refi/recast model; extra principal; prepay vs invest optimizer.
* Downsizing; reverse mortgage explorer; HELOC modeling.

---

## 12) Risk & longevity

* Longevity modeled via cohort life tables + optional frailty adjustment.
* Healthcare shocks & long‑term care scenarios (duration, cost distributions, insurance vs self‑insure).
* Job loss/sabbatical inserts.

---

## 13) Scenario architecture (tech)

* **Scenario = Base Profile + Set of Overrides** (immutable base + diff). Enables fast cloning and side‑by‑side.
* JSON‑serializable config with schema versioning and migration.
* Deterministic engine returns a timeseries; stochastic engines return a distribution object (percentiles, failure years, draws).
* Caching layer for repeated runs with identical seeds/assumptions.

---

## 14) Data schema sketch (JSON)

```json
{
  "household": {"state": "MA", "filing": "MFJ", "members": [{"name": "A", "birth_year": 1989, "retire_age": 60}, {"name": "B", "birth_year": 1990, "retire_age": 60}]},
  "accounts": [
    {"type": "taxable", "balance": 250000, "asset_mix": {"stocks": 0.7, "bonds": 0.3}},
    {"type": "traditional_ira", "balance": 400000, "asset_mix": {"stocks": 0.6, "bonds": 0.4}},
    {"type": "roth_ira", "balance": 150000, "asset_mix": {"stocks": 0.8, "bonds": 0.2}}
  ],
  "liabilities": [{"type": "mortgage", "principal": 600000, "rate": 0.055, "term_months": 360, "start_date": "2025-07-01"}],
  "incomes": [{"type": "salary", "gross": 180000, "growth": 0.03, "end_age": 60}],
  "expenses": {"baseline": 120000, "college": [{"child": "Kid1", "start_year": 2045, "annual": 45000, "years": 4}]},
  "policies": {"tax_state": "MA", "inflation": 0.02},
  "market_model": {"engine": "monte_carlo", "years": 40, "assumptions": {"stocks": {"mu": 0.07, "sigma": 0.18}, "bonds": {"mu": 0.03, "sigma": 0.06}, "corr": [[1.0, -0.2],[ -0.2, 1.0]]}},
  "strategy": {"withdrawal": {"type": "guardrails", "initial": 0.04, "bands": [0.2, -0.2]}, "glidepath": "to_40_60_by_age_70", "rebalance": {"rule": "annual"}}
}
```

---

## 15) UX flows

* **Home/Scenario Hub**: List cards with KPIs; create/clone; tag.
* **Quick Plan Wizard**: 8–12 inputs → instant result + confidence band.
* **Advanced Tabs**: Profile, Accounts, Taxes, Housing, Kids/College, Healthcare, Portfolio, Market, Strategies.
* **Scenario Compare**: sticky KPI header; synchronized timeline; diff table of assumptions.
* **What‑If Lab**: toggles with live re‑run; slider scrub for retirement age/spend rate; funnel of most impactful levers.
* **Explainability Pane**: “Why did taxes jump?” “Why IRMAA this year?”
* **Export/Share**: PDF report, CSV ledger, scenario JSON.

---

## 16) MVP vs. V2/V3

**MVP**

* Quick Plan + one Advanced tab per module (minimal fields).
* Deterministic + Monte Carlo (normal), basic tax for federal + one state.
* Accounts: taxable/trad/Roth; mortgage; Social Security; baseline expenses.
* Scenario compare (A/B), 4–6 charts, PDF export.

**V2**

* Historical bootstrap + block length control; guardrails; Roth conversion tool; IRMAA/ACA; college & childcare; property tax trajectories; asset location.

**V3**

* Regime‑switching; optimizer (maximize success / minimize lifetime tax subject to constraints); LTC module; reverse mortgage; integrations (Plaid/SSA uploads).

---

## 17) Engine interfaces (pseudocode)

```python
class MarketModel:
    def simulate(self, years, n_paths, seed, params) -> ReturnsCube:
        ...  # (assets x years x paths)

class CashflowEngine:
    def project(self, scenario: Scenario, returns: ReturnsPath) -> Ledger:
        ...  # taxes, contributions, withdrawals, balances

class Strategy:
    def withdraw(self, year, balances, need, rules, taxes) -> WithdrawalPlan:
        ...

class Evaluator:
    def evaluate(self, ledgers: list[Ledger]) -> Metrics:
        ...  # success %, p5/p50/p95, first failure year
```

---

## 18) Validation & testing

* Unit tests: tax calc vs IRS examples; SS calculator vs SSA estimates; mortgage amortization vs closed‑form.
* Golden datasets & regression tests; simulation determinism via seeds.
* Sensitivity harness: tornado chart for top drivers.

---

## 19) Security & privacy

* Local‑first option; encryption at rest; PII minimization; secrets vault; scoped tokens for aggregators; reproducible reports without raw PII.

---

## 20) Nice‑to‑have analytics

* Tornado charts: sensitivity to spend, retirement age, returns, inflation.
* Sequence‑risk heatmap: failure year vs return decile in first 5 years.
* Tax wedges chart: marginal vs average rates by year; bracket fill visualization.
* Cash bucket visual: floor‑and‑upside decomposition.

---

## 21) Feature checklist (for this project)

* [ ] Quick Start
* [ ] Scenario Manager (clone/diff)
* [ ] Deterministic & Monte Carlo engines
* [ ] Tax engine (federal + MA as template)
* [ ] Accounts & asset mixes
* [ ] Mortgage module
* [ ] Kids/college module
* [ ] SS benefits & claiming
* [ ] Withdrawal lab (4%/VPW/guardrails)
* [ ] Charts & PDF export
* [ ] Sensitivity analysis
* [ ] Historical bootstrap (V2)
* [ ] Roth conversions (V2)
* [ ] Healthcare/IRMAA/ACA (V2)
* [ ] Optimizer (V3)

---

## 22) Potential gaps / additions to consider

* Disability/term‑life insurance planning; survivor analysis.
* Business sale/windfall module; stock option/RSU modeling with vesting & 83(b) edge cases.
* Charitable planning (DAF, QCD, appreciated stock gifting) with tax interactions.
* Rental property pro‑forma & depreciation; passive loss carryovers.
* International moves (tax residency, totalization agreements) — optional.
* Behavioral guardrails: nudge alerts when success % < threshold; glidepath autopilot.

---

## 23) Next steps

1. Lock MVP scope (modules above) and finalize JSON schema v0.1.
2. Build deterministic engine + federal tax calc with unit tests.
3. Add 60/40 Monte Carlo with multivariate normal; charts + scenario compare.
4. Implement mortgage + SS + baseline expenses.
5. Ship Quick Plan → iterate into V2 features.


