"""
Tests for simulation protocol interfaces.

This module tests protocol conformance using mock implementations
and validates edge case handling for all provider protocols.
"""

from typing import Any, Dict, Optional

import numpy as np
import pytest
from numpy.typing import NDArray

from app.models.time_grid import InflationAdjuster, TimeGrid, UnitSystem


class MockReturnsProvider:
    """Mock implementation of ReturnsProvider for testing."""

    def __init__(self, num_assets: int = 2):
        self.num_assets = num_assets
        self.asset_names = [f"asset_{i}" for i in range(num_assets)]
        self.target_weights = np.array([0.6, 0.4])[:num_assets]
        self.expected_returns = np.array([0.07, 0.03])[:num_assets]

    def generate_returns(
        self, years: int, num_paths: int, seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        if years <= 0 or num_paths <= 0:
            raise ValueError("years and num_paths must be positive")

        if seed is not None:
            np.random.seed(seed)

        # Generate simple random returns
        return np.random.normal(
            loc=self.expected_returns.reshape(-1, 1, 1),
            scale=0.15,
            size=(self.num_assets, years, num_paths),
        )

    def get_asset_names(self) -> list[str]:
        return self.asset_names

    def get_target_weights(self) -> NDArray[np.float64]:
        return self.target_weights

    def get_expected_returns(self) -> NDArray[np.float64]:
        return self.expected_returns


class MockIncomeProvider:
    """Mock implementation of IncomeProvider for testing."""

    def __init__(self, base_income: float = 50000):
        self.base_income = base_income

    def get_annual_income(self, year: int, path: int, unit_system: UnitSystem) -> float:
        if year < 0 or path < 0:
            raise ValueError("year and path must be non-negative")

        # Simple growth model
        return self.base_income * (1.02**year)

    def get_income_breakdown(
        self, year: int, path: int, unit_system: UnitSystem
    ) -> Dict[str, float]:
        total = self.get_annual_income(year, path, unit_system)
        return {
            "salary": total * 0.8,
            "bonus": total * 0.1,
            "investment": total * 0.1,
        }


class MockExpenseProvider:
    """Mock implementation of ExpenseProvider for testing."""

    def __init__(self, base_expenses: float = 40000):
        self.base_expenses = base_expenses

    def get_annual_expenses(
        self, year: int, path: int, unit_system: UnitSystem
    ) -> float:
        if year < 0 or path < 0:
            raise ValueError("year and path must be non-negative")

        # Inflate expenses from base year to current year
        return unit_system.inflation_adjuster.adjust_for_inflation(
            self.base_expenses,
            unit_system.inflation_adjuster.base_year,
            unit_system.inflation_adjuster.base_year + year,
        )

    def get_expense_breakdown(
        self, year: int, path: int, unit_system: UnitSystem
    ) -> Dict[str, float]:
        total = self.get_annual_expenses(year, path, unit_system)
        return {
            "housing": total * 0.3,
            "transportation": total * 0.2,
            "healthcare": total * 0.15,
            "other": total * 0.35,
        }


class MockLiabilityProvider:
    """Mock implementation of LiabilityProvider for testing."""

    def __init__(self, annual_payment: float = 12000):
        self.annual_payment = annual_payment
        self.remaining_years = 20

    def get_annual_payments(
        self, year: int, path: int, unit_system: UnitSystem
    ) -> float:
        if year < 0 or path < 0:
            raise ValueError("year and path must be non-negative")

        # Payment stops after remaining_years
        if year >= self.remaining_years:
            return 0.0
        return self.annual_payment

    def get_payment_breakdown(
        self, year: int, path: int, unit_system: UnitSystem
    ) -> Dict[str, Dict[str, float]]:
        payment = self.get_annual_payments(year, path, unit_system)
        if payment == 0:
            return {}

        return {
            "mortgage": {
                "principal": payment * 0.6,
                "interest": payment * 0.4,
                "total": payment,
            }
        }

    def get_remaining_balances(self, year: int, path: int) -> Dict[str, float]:
        if year >= self.remaining_years:
            return {}

        remaining_balance = self.annual_payment * (self.remaining_years - year) * 0.6
        return {"mortgage": remaining_balance}


class MockWithdrawalPolicy:
    """Mock implementation of WithdrawalPolicy for testing."""

    def __init__(self, withdrawal_rate: float = 0.04):
        self.withdrawal_rate = withdrawal_rate

    def compute_withdrawal(
        self,
        year: int,
        path: int,
        cash_need: float,
        portfolio_balance: float,
        unit_system: UnitSystem,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        if portfolio_balance < 0:
            raise ValueError("portfolio_balance cannot be negative")

        # Simple percentage-based withdrawal
        return max(0, portfolio_balance * self.withdrawal_rate)

    def get_strategy_name(self) -> str:
        return f"Fixed {self.withdrawal_rate:.1%} Withdrawal"


class MockRebalancingStrategy:
    """Mock implementation of RebalancingStrategy for testing."""

    def __init__(self, threshold: float = 0.05, transaction_cost_rate: float = 0.001):
        self.threshold = threshold
        self.transaction_cost_rate = transaction_cost_rate

    def should_rebalance(
        self,
        current_weights: NDArray[np.float64],
        target_weights: NDArray[np.float64],
        year: int,
        path: int,
    ) -> bool:
        # Rebalance if any weight deviates by more than threshold
        deviations = np.abs(current_weights - target_weights)
        return np.any(deviations > self.threshold)

    def compute_transaction_cost(
        self,
        current_balances: NDArray[np.float64],
        target_balances: NDArray[np.float64],
    ) -> float:
        # Cost based on amount traded
        trades = np.abs(target_balances - current_balances)
        return np.sum(trades) * self.transaction_cost_rate

    def get_strategy_name(self) -> str:
        return f"Threshold Rebalancing ({self.threshold:.1%})"


class MockTaxCalculator:
    """Mock implementation of TaxCalculator for testing."""

    def __init__(self, tax_rate: float = 0.22):
        self.tax_rate = tax_rate

    def compute_annual_taxes(
        self,
        year: int,
        path: int,
        ordinary_income: float,
        capital_gains: float,
        withdrawals_by_account_type: Dict[str, float],
        unit_system: UnitSystem,
    ) -> float:
        # Simple flat tax on ordinary income
        return ordinary_income * self.tax_rate

    def get_effective_tax_rate(
        self, year: int, path: int, total_income: float
    ) -> float:
        return self.tax_rate


class MockPortfolioEngine:
    """Mock implementation of PortfolioEngine for testing."""

    def __init__(self):
        self.initialized = False
        self.balances = None
        self.asset_balances = None
        self.target_weights = None
        self.asset_names = None

    def initialize(
        self,
        initial_balance: float,
        target_weights: NDArray[np.float64],
        asset_names: list[str],
        num_paths: int,
    ) -> None:
        self.initialized = True
        self.initial_balance = initial_balance
        self.target_weights = target_weights
        self.asset_names = asset_names
        self.num_paths = num_paths

        # Initialize with simple data structures
        self.balances = {}  # (year, path) -> balance
        self.asset_balances = {}  # (year, path) -> asset_balances array

        # Set initial balances
        for path in range(num_paths):
            self.balances[(0, path)] = initial_balance
            self.asset_balances[(0, path)] = initial_balance * target_weights

    def apply_cash_flow(self, amount: float, year: int, path: int) -> None:
        if not self.initialized:
            raise RuntimeError("Portfolio not initialized")

        # If this is a new year/path, copy from previous year
        if (year, path) not in self.balances:
            if year > 0 and (year - 1, path) in self.balances:
                self.balances[(year, path)] = self.balances[(year - 1, path)]
                self.asset_balances[(year, path)] = self.asset_balances[
                    (year - 1, path)
                ].copy()
            else:
                self.balances[(year, path)] = 0
                self.asset_balances[(year, path)] = np.zeros(len(self.target_weights))

        current_balance = self.balances.get((year, path), 0)
        self.balances[(year, path)] = current_balance + amount

        # Proportionally adjust asset balances
        if current_balance > 0:
            adjustment_factor = (current_balance + amount) / current_balance
            self.asset_balances[(year, path)] *= adjustment_factor

    def apply_returns(
        self, asset_returns: NDArray[np.float64], year: int, path: int
    ) -> None:
        if not self.initialized:
            raise RuntimeError("Portfolio not initialized")

        # Apply returns to each asset
        current_assets = self.asset_balances.get(
            (year, path), np.zeros(len(asset_returns))
        )
        new_assets = current_assets * (1 + asset_returns)
        self.asset_balances[(year, path)] = new_assets
        self.balances[(year, path)] = np.sum(new_assets)

    def get_current_balance(self, year: int, path: int) -> float:
        return self.balances.get((year, path), 0.0)

    def get_current_weights(self, year: int, path: int) -> NDArray[np.float64]:
        assets = self.asset_balances.get(
            (year, path), np.zeros(len(self.target_weights))
        )
        total = np.sum(assets)
        if total == 0:
            return np.zeros_like(assets)
        return assets / total

    def get_asset_balances(self, year: int, path: int) -> NDArray[np.float64]:
        return self.asset_balances.get((year, path), np.zeros(len(self.target_weights)))

    def apply_rebalancing(
        self,
        target_weights: NDArray[np.float64],
        transaction_cost: float,
        year: int,
        path: int,
    ) -> None:
        total_balance = self.get_current_balance(year, path) - transaction_cost
        new_assets = total_balance * target_weights
        self.asset_balances[(year, path)] = new_assets
        self.balances[(year, path)] = total_balance


# Test fixtures
@pytest.fixture
def unit_system():
    """Create a unit system for testing."""
    time_grid = TimeGrid(start_year=2024, end_year=2054, base_year=2024)
    inflation_adjuster = InflationAdjuster(inflation_rate=0.03, base_year=2024)
    return UnitSystem(
        time_grid=time_grid, inflation_adjuster=inflation_adjuster, display_mode="real"
    )


class TestReturnsProvider:
    """Test ReturnsProvider protocol conformance."""

    def test_mock_implementation_works(self):
        """Test that mock implementation satisfies protocol."""
        provider = MockReturnsProvider()

        # Test protocol methods
        returns = provider.generate_returns(5, 100, seed=42)
        assert returns.shape == (2, 5, 100)

        names = provider.get_asset_names()
        assert len(names) == 2
        assert names == ["asset_0", "asset_1"]

        weights = provider.get_target_weights()
        assert len(weights) == 2
        assert np.allclose(weights, [0.6, 0.4])

        expected = provider.get_expected_returns()
        assert len(expected) == 2
        assert np.allclose(expected, [0.07, 0.03])

    def test_returns_generation_deterministic(self):
        """Test that returns generation is deterministic with seed."""
        provider = MockReturnsProvider()

        returns1 = provider.generate_returns(5, 100, seed=42)
        returns2 = provider.generate_returns(5, 100, seed=42)

        assert np.allclose(returns1, returns2)

    def test_invalid_parameters_raise_errors(self):
        """Test that invalid parameters raise appropriate errors."""
        provider = MockReturnsProvider()

        with pytest.raises(ValueError, match="years and num_paths must be positive"):
            provider.generate_returns(0, 100)

        with pytest.raises(ValueError, match="years and num_paths must be positive"):
            provider.generate_returns(5, 0)


class TestIncomeProvider:
    """Test IncomeProvider protocol conformance."""

    def test_mock_implementation_works(self, unit_system):
        """Test that mock implementation satisfies protocol."""
        provider = MockIncomeProvider(base_income=50000)

        # Test annual income
        income = provider.get_annual_income(0, 0, unit_system)
        assert income == 50000

        # Test growth over time
        income_year5 = provider.get_annual_income(5, 0, unit_system)
        assert income_year5 > income

        # Test breakdown
        breakdown = provider.get_income_breakdown(0, 0, unit_system)
        assert isinstance(breakdown, dict)
        assert "salary" in breakdown
        assert sum(breakdown.values()) == pytest.approx(income)

    def test_invalid_parameters_raise_errors(self, unit_system):
        """Test that invalid parameters raise appropriate errors."""
        provider = MockIncomeProvider()

        with pytest.raises(ValueError, match="year and path must be non-negative"):
            provider.get_annual_income(-1, 0, unit_system)

        with pytest.raises(ValueError, match="year and path must be non-negative"):
            provider.get_annual_income(0, -1, unit_system)


class TestExpenseProvider:
    """Test ExpenseProvider protocol conformance."""

    def test_mock_implementation_works(self, unit_system):
        """Test that mock implementation satisfies protocol."""
        provider = MockExpenseProvider(base_expenses=40000)

        # Test annual expenses
        expenses = provider.get_annual_expenses(0, 0, unit_system)
        assert expenses >= 40000  # Should be inflated

        # Test breakdown
        breakdown = provider.get_expense_breakdown(0, 0, unit_system)
        assert isinstance(breakdown, dict)
        assert "housing" in breakdown
        assert sum(breakdown.values()) == pytest.approx(expenses)


class TestLiabilityProvider:
    """Test LiabilityProvider protocol conformance."""

    def test_mock_implementation_works(self, unit_system):
        """Test that mock implementation satisfies protocol."""
        provider = MockLiabilityProvider(annual_payment=12000)

        # Test payments during liability period
        payment = provider.get_annual_payments(0, 0, unit_system)
        assert payment == 12000

        # Test payments after liability period
        payment_late = provider.get_annual_payments(25, 0, unit_system)
        assert payment_late == 0

        # Test breakdown
        breakdown = provider.get_payment_breakdown(0, 0, unit_system)
        assert "mortgage" in breakdown
        assert breakdown["mortgage"]["total"] == payment

        # Test remaining balances
        balances = provider.get_remaining_balances(0, 0)
        assert "mortgage" in balances
        assert balances["mortgage"] > 0


class TestWithdrawalPolicy:
    """Test WithdrawalPolicy protocol conformance."""

    def test_mock_implementation_works(self, unit_system):
        """Test that mock implementation satisfies protocol."""
        policy = MockWithdrawalPolicy(withdrawal_rate=0.04)

        # Test withdrawal calculation
        withdrawal = policy.compute_withdrawal(
            year=0,
            path=0,
            cash_need=20000,
            portfolio_balance=500000,
            unit_system=unit_system,
        )
        assert withdrawal == 20000  # 4% of 50k

        # Test strategy name
        name = policy.get_strategy_name()
        assert "4.0%" in name

    def test_negative_portfolio_raises_error(self, unit_system):
        """Test that negative portfolio balance raises error."""
        policy = MockWithdrawalPolicy()

        with pytest.raises(ValueError, match="portfolio_balance cannot be negative"):
            policy.compute_withdrawal(
                year=0,
                path=0,
                cash_need=0,
                portfolio_balance=-1000,
                unit_system=unit_system,
            )


class TestRebalancingStrategy:
    """Test RebalancingStrategy protocol conformance."""

    def test_mock_implementation_works(self):
        """Test that mock implementation satisfies protocol."""
        strategy = MockRebalancingStrategy(threshold=0.05)

        # Test rebalancing decision
        current_weights = np.array([0.7, 0.3])
        target_weights = np.array([0.6, 0.4])

        should_rebalance = strategy.should_rebalance(
            current_weights, target_weights, year=0, path=0
        )
        assert should_rebalance  # 0.1 deviation > 0.05 threshold

        # Test transaction cost
        current_balances = np.array([70000, 30000])
        target_balances = np.array([60000, 40000])

        cost = strategy.compute_transaction_cost(current_balances, target_balances)
        assert cost > 0

        # Test strategy name
        name = strategy.get_strategy_name()
        assert "5.0%" in name


class TestTaxCalculator:
    """Test TaxCalculator protocol conformance."""

    def test_mock_implementation_works(self, unit_system):
        """Test that mock implementation satisfies protocol."""
        calculator = MockTaxCalculator(tax_rate=0.22)

        # Test tax calculation
        taxes = calculator.compute_annual_taxes(
            year=0,
            path=0,
            ordinary_income=50000,
            capital_gains=5000,
            withdrawals_by_account_type={"taxable": 10000},
            unit_system=unit_system,
        )
        assert taxes == 11000  # 22% of 50k

        # Test effective tax rate
        rate = calculator.get_effective_tax_rate(0, 0, 50000)
        assert rate == 0.22


class TestPortfolioEngine:
    """Test PortfolioEngine protocol conformance."""

    def test_mock_implementation_works(self):
        """Test that mock implementation satisfies protocol."""
        engine = MockPortfolioEngine()

        # Test initialization
        engine.initialize(
            initial_balance=100000,
            target_weights=np.array([0.6, 0.4]),
            asset_names=["stocks", "bonds"],
            num_paths=10,
        )

        # Test balance retrieval
        balance = engine.get_current_balance(0, 0)
        assert balance == 100000

        # Test weights
        weights = engine.get_current_weights(0, 0)
        assert np.allclose(weights, [0.6, 0.4])

        # Test asset balances
        assets = engine.get_asset_balances(0, 0)
        assert np.allclose(assets, [60000, 40000])

        # Test cash flow application
        engine.apply_cash_flow(-5000, 1, 0)  # Withdrawal
        balance_after = engine.get_current_balance(1, 0)
        assert balance_after == 95000

        # Test returns application
        returns = np.array([0.1, 0.05])  # 10% stocks, 5% bonds
        engine.apply_returns(returns, 1, 0)
        # Should apply to asset balances

        # Test rebalancing
        engine.apply_rebalancing(
            target_weights=np.array([0.7, 0.3]), transaction_cost=100, year=1, path=0
        )

    def test_uninitialized_engine_raises_error(self):
        """Test that using uninitialized engine raises error."""
        engine = MockPortfolioEngine()

        with pytest.raises(RuntimeError, match="Portfolio not initialized"):
            engine.apply_cash_flow(1000, 0, 0)

        with pytest.raises(RuntimeError, match="Portfolio not initialized"):
            engine.apply_returns(np.array([0.1, 0.05]), 0, 0)


# Integration tests
class TestProtocolIntegration:
    """Test that all protocols work together."""

    def test_all_protocols_can_be_instantiated(self, unit_system):
        """Test that all mock implementations can be created and used together."""
        # Create all providers
        returns_provider = MockReturnsProvider()
        income_provider = MockIncomeProvider()
        expense_provider = MockExpenseProvider()
        liability_provider = MockLiabilityProvider()
        withdrawal_policy = MockWithdrawalPolicy()
        rebalancing_strategy = MockRebalancingStrategy()
        tax_calculator = MockTaxCalculator()
        portfolio_engine = MockPortfolioEngine()

        # Test that they can all be used in sequence
        # (This would be the orchestrator's job)

        # Initialize portfolio
        portfolio_engine.initialize(
            initial_balance=500000,
            target_weights=returns_provider.get_target_weights(),
            asset_names=returns_provider.get_asset_names(),
            num_paths=1,
        )

        # Simulate one year
        year, path = 0, 0

        # Get cash flows
        income = income_provider.get_annual_income(year, path, unit_system)
        expenses = expense_provider.get_annual_expenses(year, path, unit_system)
        liabilities = liability_provider.get_annual_payments(year, path, unit_system)

        cash_need = expenses + liabilities - income
        portfolio_balance = portfolio_engine.get_current_balance(year, path)

        # Compute withdrawal
        withdrawal = withdrawal_policy.compute_withdrawal(
            year, path, cash_need, portfolio_balance, unit_system
        )

        # Apply cash flow
        portfolio_engine.apply_cash_flow(-withdrawal, year, path)

        # Apply returns
        returns = returns_provider.generate_returns(1, 1, seed=42)
        portfolio_engine.apply_returns(returns[:, 0, 0], year, path)

        # Check rebalancing
        current_weights = portfolio_engine.get_current_weights(year, path)
        target_weights = returns_provider.get_target_weights()

        if rebalancing_strategy.should_rebalance(
            current_weights, target_weights, year, path
        ):
            cost = rebalancing_strategy.compute_transaction_cost(
                portfolio_engine.get_asset_balances(year, path),
                portfolio_balance * target_weights,
            )
            portfolio_engine.apply_rebalancing(target_weights, cost, year, path)

        # Compute taxes
        taxes = tax_calculator.compute_annual_taxes(
            year, path, income, 0, {"taxable": withdrawal}, unit_system
        )

        # All operations should complete without error
        assert withdrawal >= 0
        assert taxes >= 0
        assert portfolio_engine.get_current_balance(year, path) >= 0
