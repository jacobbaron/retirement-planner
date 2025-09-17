# Simulation Orchestrator Implementation Plan

## Overview
Transform the current hardcoded `PortfolioEvolution.simulate()` into a flexible, composable simulation orchestrator that integrates all existing engines (income, expenses, social security, mortgages, withdrawals, etc.) through clean interfaces.

## Epic: EP-5 - Flexible Simulation Orchestrator

### Phase 1: Core Abstractions & Interfaces

---

#### **EP-5-T1**: Define Provider Protocol Interfaces
**Priority**: High  
**Estimate**: 2-3 days  
**Dependencies**: None  

**Acceptance Criteria**:
- [ ] Create `app/models/simulation/protocols.py` with clean Protocol interfaces:
  - `ReturnsProvider` - generates market returns
  - `IncomeProvider` - provides annual income by year/path
  - `ExpenseProvider` - provides annual expenses by year/path  
  - `LiabilityProvider` - provides liability payments (mortgages, loans)
  - `WithdrawalPolicy` - computes withdrawal amounts
  - `RebalancingStrategy` - handles portfolio rebalancing
  - `TaxCalculator` - computes annual taxes (optional)
- [ ] Each protocol has clear method signatures with type hints
- [ ] Include docstrings with parameter descriptions and return types
- [ ] Add validation for common edge cases (negative balances, invalid years)

**Tests**:
- [ ] Protocol conformance tests using mock implementations
- [ ] Type checking passes with mypy
- [ ] Edge case validation tests

**Files to Create**:
- `app/models/simulation/__init__.py`
- `app/models/simulation/protocols.py`

---

#### **EP-5-T2**: Create Simulation Configuration Model
**Priority**: High  
**Estimate**: 1-2 days  
**Dependencies**: EP-5-T1  

**Acceptance Criteria**:
- [ ] Create `SimulationConfig` Pydantic model in `app/models/simulation/config.py`
- [ ] Include all required configuration:
  - `unit_system: UnitSystem` (time grid, inflation)
  - `num_paths: int` (Monte Carlo paths)
  - `initial_portfolio_balance: float`
  - `target_asset_weights: Dict[str, float]`
  - Provider instances for all protocols
- [ ] Validation ensures providers are compatible with config
- [ ] Support optional providers (e.g., taxes can be None)
- [ ] Clear error messages for misconfiguration

**Tests**:
- [ ] Configuration validation tests
- [ ] Provider compatibility tests
- [ ] Invalid configuration rejection tests

**Files to Create**:
- `app/models/simulation/config.py`

---

#### **EP-5-T3**: Create Simulation Result Model
**Priority**: High  
**Estimate**: 1-2 days  
**Dependencies**: None  

**Acceptance Criteria**:
- [ ] Create `SimulationResult` Pydantic model in `app/models/simulation/result.py`
- [ ] Include comprehensive result data:
  - `portfolio_balances: NDArray` (years × paths)
  - `asset_balances: NDArray` (assets × years × paths)  
  - `withdrawals: NDArray` (years × paths)
  - `incomes: NDArray` (years × paths)
  - `expenses: NDArray` (years × paths)
  - `taxes: NDArray` (years × paths)
  - `rebalancing_events: NDArray[bool]` (years × paths)
  - `transaction_costs: NDArray` (years × paths)
- [ ] Optional detailed ledger for debugging/exports
- [ ] Helper methods for common queries (final balances, success rate, etc.)
- [ ] Serialization support for storage/export

**Tests**:
- [ ] Result model creation and validation
- [ ] Helper method accuracy tests
- [ ] Serialization round-trip tests

**Files to Create**:
- `app/models/simulation/result.py`

---

### Phase 2: Provider Adapters

---

#### **EP-5-T4**: Implement Returns Provider Adapter
**Priority**: High  
**Estimate**: 2 days  
**Dependencies**: EP-5-T1  

**Acceptance Criteria**:
- [ ] Create `RandomReturnsProviderAdapter` in `app/models/simulation/adapters/returns.py`
- [ ] Wraps existing `RandomReturnsGenerator` with `ReturnsProvider` interface
- [ ] Handles correlation matrices and multiple asset classes
- [ ] Supports both normal and lognormal distributions
- [ ] Maintains deterministic seeding for reproducibility
- [ ] No changes to existing `RandomReturnsGenerator` code

**Tests**:
- [ ] Adapter produces same results as direct `RandomReturnsGenerator` usage
- [ ] Protocol compliance tests
- [ ] Deterministic seeding verification
- [ ] Multiple asset class handling

**Files to Create**:
- `app/models/simulation/adapters/__init__.py`
- `app/models/simulation/adapters/returns.py`

---

#### **EP-5-T5**: Implement Income Provider Adapter
**Priority**: High  
**Estimate**: 3 days  
**Dependencies**: EP-5-T1  

**Acceptance Criteria**:
- [ ] Create `IncomeProviderAdapter` in `app/models/simulation/adapters/income.py`
- [ ] Integrates `IncomeEngine` and `SocialSecurityEngine`
- [ ] Handles all income types: salary, business, investment, retirement, SS
- [ ] Supports income timing constraints and growth rates
- [ ] Proper inflation adjustment using `UnitSystem`
- [ ] Path-specific income variations (if configured)

**Tests**:
- [ ] All income types properly calculated
- [ ] Timing constraints respected
- [ ] Inflation adjustment accuracy
- [ ] Integration with existing engine tests

**Files to Create**:
- `app/models/simulation/adapters/income.py`

---

#### **EP-5-T6**: Implement Expense Provider Adapter  
**Priority**: High  
**Estimate**: 2 days  
**Dependencies**: EP-5-T1  

**Acceptance Criteria**:
- [ ] Create `ExpenseProviderAdapter` in `app/models/simulation/adapters/expenses.py`
- [ ] Wraps existing `ExpenseEngine`
- [ ] Handles baseline expenses and lumpy events
- [ ] Supports expense categories: housing, transportation, healthcare
- [ ] Proper inflation adjustment
- [ ] Year-specific expense variations

**Tests**:
- [ ] All expense types calculated correctly
- [ ] Lumpy events occur in correct years
- [ ] Inflation adjustment accuracy
- [ ] Category-specific expense handling

**Files to Create**:
- `app/models/simulation/adapters/expenses.py`

---

#### **EP-5-T7**: Implement Liability Provider Adapter
**Priority**: Medium  
**Estimate**: 2-3 days  
**Dependencies**: EP-5-T1  

**Acceptance Criteria**:
- [ ] Create `LiabilityProviderAdapter` in `app/models/simulation/adapters/liabilities.py`
- [ ] Integrates `MortgageCalculator` for mortgage payments
- [ ] Handles multiple mortgages with different terms
- [ ] Supports other loan types (student, auto, credit card)
- [ ] Calculates principal/interest splits
- [ ] Handles extra payments and refinancing scenarios

**Tests**:
- [ ] Mortgage amortization accuracy
- [ ] Multiple mortgage handling
- [ ] Extra payment scenarios
- [ ] Refinancing calculations

**Files to Create**:
- `app/models/simulation/adapters/liabilities.py`

---

#### **EP-5-T8**: Implement Withdrawal Policy Adapters
**Priority**: High  
**Estimate**: 2 days  
**Dependencies**: EP-5-T1  

**Acceptance Criteria**:
- [ ] Create withdrawal policy adapters in `app/models/simulation/adapters/withdrawals.py`
- [ ] Unify existing withdrawal rules under `WithdrawalPolicy` interface:
  - `FixedRealWithdrawalPolicyAdapter`
  - `FixedPercentageWithdrawalPolicyAdapter` 
  - `VPWWithdrawalPolicyAdapter`
- [ ] Fix inflation overflow issue in existing withdrawal rules
- [ ] Consistent interface: `compute(year, path, need, portfolio_balance) -> float`
- [ ] Handle edge cases (depleted portfolio, negative needs)

**Tests**:
- [ ] All withdrawal strategies produce expected results
- [ ] Inflation overflow fix verified
- [ ] Edge case handling tests
- [ ] Policy switching tests

**Files to Create**:
- `app/models/simulation/adapters/withdrawals.py`

---

### Phase 3: Portfolio Engine Refactor

---

#### **EP-5-T9**: Extract Portfolio Engine Interface
**Priority**: High  
**Estimate**: 3-4 days  
**Dependencies**: EP-5-T1  

**Acceptance Criteria**:
- [ ] Create `PortfolioEngine` protocol in `app/models/simulation/protocols.py`
- [ ] Define clean interface methods:
  - `initialize(initial_balance, target_weights)`
  - `apply_cash_flow(amount, year, path)` (contributions/withdrawals)
  - `apply_returns(asset_returns_slice, year, path)`
  - `get_current_weights(year, path) -> NDArray`
  - `get_current_balance(year, path) -> float`
  - `check_rebalancing_needed(year, path) -> bool`
  - `apply_rebalancing(year, path) -> float` (returns transaction cost)
- [ ] Create `PortfolioEngineAdapter` that wraps existing `PortfolioEvolution`
- [ ] Maintain backward compatibility - existing `simulate()` method unchanged
- [ ] Handle multi-path state management cleanly

**Tests**:
- [ ] Adapter produces identical results to existing `PortfolioEvolution`
- [ ] All interface methods work correctly
- [ ] Multi-path state isolation
- [ ] Backward compatibility verified

**Files to Modify**:
- `app/models/simulation/protocols.py` (add PortfolioEngine)
- `app/models/portfolio_evolution.py` (add adapter class)

---

#### **EP-5-T10**: Implement Rebalancing Strategy Adapter
**Priority**: Medium  
**Estimate**: 1-2 days  
**Dependencies**: EP-5-T9  

**Acceptance Criteria**:
- [ ] Create `RebalancingStrategyAdapter` in `app/models/simulation/adapters/rebalancing.py`
- [ ] Extract rebalancing logic from `PortfolioEvolution`
- [ ] Support different rebalancing strategies:
  - Threshold-based (current implementation)
  - Time-based (annual, quarterly)
  - No rebalancing
- [ ] Configurable transaction costs
- [ ] Clean separation of concerns

**Tests**:
- [ ] Threshold-based rebalancing accuracy
- [ ] Transaction cost calculations
- [ ] Strategy switching tests
- [ ] No-rebalancing scenario

**Files to Create**:
- `app/models/simulation/adapters/rebalancing.py`

---

### Phase 4: Core Orchestrator

---

#### **EP-5-T11**: Implement Simulation Orchestrator
**Priority**: High  
**Estimate**: 5-6 days  
**Dependencies**: EP-5-T2, EP-5-T3, EP-5-T4, EP-5-T5, EP-5-T6, EP-5-T7, EP-5-T8, EP-5-T9  

**Acceptance Criteria**:
- [ ] Create `SimulationOrchestrator` class in `app/models/simulation/orchestrator.py`
- [ ] Implement main orchestration loop:
  - Initialize portfolio and providers
  - For each year and path:
    - Get income from `IncomeProvider`
    - Get expenses from `ExpenseProvider` 
    - Get liability payments from `LiabilityProvider`
    - Calculate cash need = expenses + liabilities - income
    - Compute withdrawal amount from `WithdrawalPolicy`
    - Apply taxes if `TaxCalculator` provided
    - Apply cash flows to portfolio
    - Apply market returns from `ReturnsProvider`
    - Check and apply rebalancing
    - Record all flows and balances
- [ ] Return comprehensive `SimulationResult`
- [ ] Handle edge cases (portfolio depletion, negative cash flows)
- [ ] Efficient memory management for large simulations
- [ ] Progress reporting for long-running simulations

**Tests**:
- [ ] End-to-end simulation produces expected results
- [ ] All cash flows properly integrated
- [ ] Edge case handling (portfolio depletion, etc.)
- [ ] Memory efficiency for large simulations
- [ ] Comparison with existing `PortfolioEvolution` results

**Files to Create**:
- `app/models/simulation/orchestrator.py`

---

#### **EP-5-T12**: Add Tax Calculator Stub and Interface
**Priority**: Low  
**Estimate**: 1 day  
**Dependencies**: EP-5-T1, EP-5-T11  

**Acceptance Criteria**:
- [ ] Create `TaxCalculator` protocol in protocols
- [ ] Implement `NoopTaxCalculator` (returns 0 taxes)
- [ ] Design interface for future tax engine integration
- [ ] Document tax calculation requirements for future implementation
- [ ] Integration point ready for tax engine when available

**Tests**:
- [ ] Noop calculator returns zero taxes
- [ ] Interface design validation
- [ ] Integration with orchestrator

**Files to Create**:
- `app/models/simulation/adapters/taxes.py`

---

### Phase 5: Integration & Factory Methods

---

#### **EP-5-T13**: Create Scenario to Configuration Factory
**Priority**: High  
**Estimate**: 3-4 days  
**Dependencies**: EP-5-T11, EP-5-T12  

**Acceptance Criteria**:
- [ ] Create `create_simulation_config_from_scenario()` in `app/models/simulation/factory.py`
- [ ] Convert `Scenario` model to `SimulationConfig`:
  - Extract time grid from scenario metadata
  - Create provider instances from scenario components
  - Set up proper asset allocation and rebalancing
  - Configure withdrawal strategy from scenario strategy
  - Handle all scenario edge cases and validations
- [ ] Support all existing scenario features
- [ ] Maintain data integrity during conversion
- [ ] Clear error messages for unsupported scenario features

**Tests**:
- [ ] All scenario types convert correctly
- [ ] Converted config produces equivalent results
- [ ] Edge cases handled properly
- [ ] Validation errors are clear

**Files to Create**:
- `app/models/simulation/factory.py`

---

#### **EP-5-T14**: Integration with Success Metrics Calculator
**Priority**: Medium  
**Estimate**: 2 days  
**Dependencies**: EP-5-T11, EP-5-T3  

**Acceptance Criteria**:
- [ ] Modify `SuccessMetricsCalculator` to accept `SimulationResult` directly
- [ ] Update success metrics calculation to use new result format
- [ ] Maintain backward compatibility with existing interfaces
- [ ] Add new metrics possible with richer simulation data:
  - Income replacement ratio
  - Expense coverage probability
  - Tax burden analysis
  - Cash flow sustainability

**Tests**:
- [ ] Success metrics match existing calculations
- [ ] New metrics are accurate
- [ ] Backward compatibility maintained
- [ ] Performance is acceptable

**Files to Modify**:
- `app/models/success_metrics.py`

---

### Phase 6: Performance & Optimization

---

#### **EP-5-T15**: Optimize Orchestrator Performance
**Priority**: Medium  
**Estimate**: 2-3 days  
**Dependencies**: EP-5-T11, EP-5-T14  

**Acceptance Criteria**:
- [ ] Profile orchestrator performance with large simulations (10K+ paths, 30+ years)
- [ ] Optimize bottlenecks:
  - Vectorize calculations where possible
  - Reduce memory allocations in inner loops
  - Optimize array operations
  - Consider parallel processing for independent paths
- [ ] Benchmark against existing `PortfolioEvolution.simulate()`
- [ ] Target: <20% performance regression for equivalent functionality
- [ ] Memory usage should scale linearly with paths/years

**Tests**:
- [ ] Performance benchmarks
- [ ] Memory usage tests
- [ ] Accuracy maintained after optimizations
- [ ] Large simulation stress tests

**Files to Modify**:
- `app/models/simulation/orchestrator.py`

---

#### **EP-5-T16**: Add Numerical Stability Improvements
**Priority**: Medium  
**Estimate**: 1-2 days  
**Dependencies**: EP-5-T8  

**Acceptance Criteria**:
- [ ] Fix inflation overflow in withdrawal calculations using `math.log1p()` and `math.expm1()`
- [ ] Add numerical stability checks for edge cases:
  - Very small portfolio balances
  - High inflation rates over long periods
  - Extreme market returns
- [ ] Implement graceful degradation for numerical edge cases
- [ ] Add warnings for potentially unstable calculations

**Tests**:
- [ ] High inflation scenarios work correctly
- [ ] Extreme market return scenarios
- [ ] Very small balance handling
- [ ] Numerical accuracy over long periods

**Files to Modify**:
- `app/models/simulation/adapters/withdrawals.py`
- `app/models/simulation/orchestrator.py`

---

### Phase 7: Testing & Documentation

---

#### **EP-5-T17**: Comprehensive Integration Tests
**Priority**: High  
**Estimate**: 3-4 days  
**Dependencies**: EP-5-T13, EP-5-T14  

**Acceptance Criteria**:
- [ ] Create end-to-end integration tests in `tests/integration/test_simulation_orchestrator.py`
- [ ] Test complete scenarios from `Scenario` model to `SimulationResult`
- [ ] Verify results match existing engine outputs where applicable
- [ ] Test all provider combinations
- [ ] Performance regression tests
- [ ] Edge case scenarios (portfolio depletion, extreme markets, etc.)
- [ ] Golden dataset tests for reproducibility

**Tests**:
- [ ] End-to-end scenario tests
- [ ] Provider combination tests
- [ ] Performance regression tests
- [ ] Golden dataset verification
- [ ] Edge case coverage

**Files to Create**:
- `tests/integration/test_simulation_orchestrator.py`
- `tests/fixtures/golden_scenarios.json`

---

#### **EP-5-T18**: Update Documentation and Examples
**Priority**: Medium  
**Estimate**: 2 days  
**Dependencies**: EP-5-T17  

**Acceptance Criteria**:
- [ ] Update README with new simulation architecture
- [ ] Create comprehensive examples in `examples/simulation_orchestrator.py`
- [ ] Document provider interfaces and how to extend them
- [ ] Migration guide from old `PortfolioEvolution.simulate()` usage
- [ ] Performance characteristics and best practices
- [ ] Troubleshooting guide for common issues

**Files to Create/Modify**:
- `README.md` (update simulation section)
- `examples/simulation_orchestrator.py`
- `docs/SIMULATION_ARCHITECTURE.md`
- `docs/MIGRATION_GUIDE.md`

---

## Migration Strategy

### Backward Compatibility
- All existing code continues to work unchanged
- `PortfolioEvolution.simulate()` method preserved
- Existing tests remain valid
- New orchestrator available as opt-in upgrade

### Rollout Plan
1. **Phase 1-2**: Core abstractions (no breaking changes)
2. **Phase 3-4**: Orchestrator implementation (parallel to existing code)
3. **Phase 5**: Integration layer (enables new functionality)
4. **Phase 6-7**: Optimization and documentation

### Risk Mitigation
- Extensive testing at each phase
- Performance benchmarking to prevent regressions
- Golden dataset tests to ensure numerical accuracy
- Gradual rollout with fallback to existing implementation

## Success Metrics
- [ ] All existing functionality preserved
- [ ] New orchestrator handles complex scenarios (income + expenses + liabilities + withdrawals + taxes)
- [ ] Performance within 20% of existing implementation
- [ ] Clean, extensible architecture for future enhancements
- [ ] Comprehensive test coverage (>90%)
- [ ] Clear documentation and migration path 