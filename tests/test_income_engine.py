"""
Tests for income processing engine.

This module tests the income modeling functionality including salary growth,
bonuses, other income sources, inflation adjustment, timing constraints,
and tax treatment metadata.
"""

import pytest

from app.models.income_engine import (
    BusinessIncome,
    EmploymentIncome,
    IncomeCategory,
    IncomeChangeEvent,
    IncomeEngine,
    InvestmentIncome,
    RentalIncome,
    RetirementIncome,
    SelfEmploymentIncome,
    VariableIncome,
    calculate_income_present_value,
    create_income_engine_from_scenario,
    validate_income_timing,
)
from app.models.scenario import (
    Accounts,
    ExpectedReturns,
    Expenses,
    Household,
    Incomes,
    Liabilities,
    MarketModel,
    OtherIncome,
    Policies,
    Salary,
    Scenario,
    Strategy,
    Volatility,
)
from app.models.time_grid import InflationAdjuster, TimeGrid


class TestIncomeCategory:
    """Test the base IncomeCategory class."""

    def test_basic_income_category(self):
        """Test basic income category creation."""
        category = IncomeCategory(
            name="test_income", annual_amount=50000, inflation_adjusted=True
        )

        assert category.name == "test_income"
        assert category.annual_amount == 50000
        assert category.inflation_adjusted is True
        assert category.start_year is None
        assert category.end_year is None
        assert category.growth_rate == 0.0
        assert category.tax_treatment == "other_income"
        assert category.withholding_rate == 0.0

    def test_income_category_with_timing(self):
        """Test income category with start and end years."""
        category = IncomeCategory(
            name="contract_work",
            annual_amount=75000,
            start_year=2025,
            end_year=2030,
            growth_rate=0.05,
        )

        assert category.name == "contract_work"
        assert category.annual_amount == 75000
        assert category.start_year == 2025
        assert category.end_year == 2030
        assert category.growth_rate == 0.05

    def test_invalid_end_year(self):
        """Test validation of end year."""
        with pytest.raises(ValueError):
            IncomeCategory(
                name="invalid",
                annual_amount=50000,
                start_year=2030,
                end_year=2025,  # End before start
            )

    def test_inflation_adjustment(self):
        """Test inflation adjustment calculation."""
        category = IncomeCategory(
            name="test", annual_amount=50000, inflation_adjusted=True
        )
        inflation_adjuster = InflationAdjuster(base_year=2024, inflation_rate=0.025)

        # Same year should return original amount
        amount = category.get_inflation_adjusted_amount(2024, inflation_adjuster)
        assert amount == 50000

        # Future year should be higher
        amount = category.get_inflation_adjusted_amount(2025, inflation_adjuster)
        expected = 50000 * 1.025
        assert abs(amount - expected) < 0.01

    def test_growth_adjustment(self):
        """Test growth adjustment calculation."""
        category = IncomeCategory(name="test", annual_amount=50000, growth_rate=0.03)

        # Base year should return original amount
        amount = category.get_growth_adjusted_amount(2024, 2024)
        assert amount == 50000

        # Future year should be higher
        amount = category.get_growth_adjusted_amount(2025, 2024)
        expected = 50000 * 1.03
        assert abs(amount - expected) < 0.01


class TestEmploymentIncome:
    """Test the EmploymentIncome class."""

    def test_employment_income_creation(self):
        """Test employment income creation."""
        income = EmploymentIncome(
            name="primary_salary",
            annual_amount=80000,
            person="primary",
            employer="Tech Corp",
            job_title="Software Engineer",
            bonus_amount=10000,
        )

        assert income.name == "primary_salary"
        assert income.annual_amount == 80000
        assert income.person == "primary"
        assert income.employer == "Tech Corp"
        assert income.job_title == "Software Engineer"
        assert income.bonus_amount == 10000
        assert income.tax_treatment == "w2_employment"
        assert income.withholding_rate == 0.22

    def test_bonus_calculations(self):
        """Test bonus amount calculations."""
        income = EmploymentIncome(
            name="test",
            annual_amount=80000,
            person="primary",
            bonus_amount=10000,
            bonus_frequency="annual",
            bonus_start_year=2025,
        )

        # Before bonus starts
        bonus = income.get_bonus_amount(2024)
        assert bonus == 0.0

        # After bonus starts
        bonus = income.get_bonus_amount(2025)
        assert bonus == 10000

        bonus = income.get_bonus_amount(2026)
        assert bonus == 10000

    def test_one_time_bonus(self):
        """Test one-time bonus calculation."""
        income = EmploymentIncome(
            name="test",
            annual_amount=80000,
            person="primary",
            bonus_amount=50000,
            bonus_frequency="one_time",
            bonus_start_year=2025,
        )

        # Only in the specific year
        bonus = income.get_bonus_amount(2025)
        assert bonus == 50000

        # Not in other years
        bonus = income.get_bonus_amount(2026)
        assert bonus == 0.0

    def test_no_bonus(self):
        """Test income with no bonus."""
        income = EmploymentIncome(
            name="test", annual_amount=80000, person="primary", bonus_amount=0
        )

        bonus = income.get_bonus_amount(2025)
        assert bonus == 0.0


class TestSelfEmploymentIncome:
    """Test the SelfEmploymentIncome class."""

    def test_self_employment_income_creation(self):
        """Test self-employment income creation."""
        income = SelfEmploymentIncome(
            name="freelance_work",
            annual_amount=60000,
            person="primary",
            business_name="Freelance Consulting",
            se_tax_rate=0.1413,
        )

        assert income.name == "freelance_work"
        assert income.annual_amount == 60000
        assert income.person == "primary"
        assert income.business_name == "Freelance Consulting"
        assert income.se_tax_rate == 0.1413
        assert income.tax_treatment == "self_employment"
        assert income.withholding_rate == 0.30


class TestBusinessIncome:
    """Test the BusinessIncome class."""

    def test_business_income_creation(self):
        """Test business income creation."""
        income = BusinessIncome(
            name="small_business",
            annual_amount=100000,
            person="primary",
            business_name="My Business LLC",
            business_type="consulting",
            annual_deductions=20000,
            depreciation=5000,
        )

        assert income.name == "small_business"
        assert income.annual_amount == 100000
        assert income.person == "primary"
        assert income.business_name == "My Business LLC"
        assert income.business_type == "consulting"
        assert income.annual_deductions == 20000
        assert income.depreciation == 5000
        assert income.tax_treatment == "business_income"
        assert income.withholding_rate == 0.25

    def test_net_income_calculation(self):
        """Test net business income calculation."""
        income = BusinessIncome(
            name="test",
            annual_amount=100000,
            person="primary",
            annual_deductions=20000,
        )
        inflation_adjuster = InflationAdjuster(base_year=2024, inflation_rate=0.025)

        net_income = income.get_net_income(2024, inflation_adjuster)
        expected = 100000 - 20000  # No inflation adjustment in base year
        assert net_income == expected

        # Test with inflation-adjusted deductions
        net_income = income.get_net_income(2025, inflation_adjuster)
        # Both gross income and deductions are inflation-adjusted
        gross = 100000 * 1.025  # Inflation adjusted
        deductions = 20000 * 1.025  # Also inflation adjusted
        expected = max(0, gross - deductions)
        assert abs(net_income - expected) < 0.01


class TestInvestmentIncome:
    """Test the InvestmentIncome class."""

    def test_investment_income_creation(self):
        """Test investment income creation."""
        income = InvestmentIncome(
            name="dividend_income",
            annual_amount=5000,
            income_type="dividends",
            account_type="taxable",
            qualified_dividends=True,
        )

        assert income.name == "dividend_income"
        assert income.annual_amount == 5000
        assert income.income_type == "dividends"
        assert income.account_type == "taxable"
        assert income.qualified_dividends is True
        assert income.tax_treatment == "investment_income"
        assert income.withholding_rate == 0.15  # Lower for qualified dividends

    def test_non_qualified_dividends(self):
        """Test non-qualified dividend withholding."""
        income = InvestmentIncome(
            name="dividend_income",
            annual_amount=5000,
            income_type="dividends",
            qualified_dividends=False,
        )

        assert income.withholding_rate == 0.22  # Higher for non-qualified


class TestRentalIncome:
    """Test the RentalIncome class."""

    def test_rental_income_creation(self):
        """Test rental income creation."""
        income = RentalIncome(
            name="rental_property",
            annual_amount=24000,
            property_address="123 Main St",
            property_type="residential",
            annual_expenses=8000,
            annual_depreciation=4000,
            management_fee_rate=0.08,
        )

        assert income.name == "rental_property"
        assert income.annual_amount == 24000
        assert income.property_address == "123 Main St"
        assert income.property_type == "residential"
        assert income.annual_expenses == 8000
        assert income.annual_depreciation == 4000
        assert income.management_fee_rate == 0.08
        assert income.tax_treatment == "rental_income"
        assert income.withholding_rate == 0.0  # No withholding for rental

    def test_net_rental_income_calculation(self):
        """Test net rental income calculation."""
        income = RentalIncome(
            name="test",
            annual_amount=24000,
            annual_expenses=8000,
            management_fee_rate=0.08,
        )
        inflation_adjuster = InflationAdjuster(base_year=2024, inflation_rate=0.025)

        net_income = income.get_net_rental_income(2024, inflation_adjuster)
        management_fee = 24000 * 0.08
        expected = 24000 - management_fee - 8000
        assert net_income == expected

        # Test with inflation adjustment
        net_income = income.get_net_rental_income(2025, inflation_adjuster)
        gross_rent = 24000 * 1.025
        management_fee = gross_rent * 0.08
        expenses = 8000 * 1.025  # Expenses also inflation-adjusted
        expected = gross_rent - management_fee - expenses
        assert abs(net_income - expected) < 0.01


class TestRetirementIncome:
    """Test the RetirementIncome class."""

    def test_retirement_income_creation(self):
        """Test retirement income creation."""
        income = RetirementIncome(
            name="pension",
            annual_amount=30000,
            income_type="pension",
            person="primary",
            cola_rate=0.02,
            taxable_percentage=0.8,
        )

        assert income.name == "pension"
        assert income.annual_amount == 30000
        assert income.income_type == "pension"
        assert income.person == "primary"
        assert income.cola_rate == 0.02
        assert income.taxable_percentage == 0.8
        assert income.tax_treatment == "retirement_income"
        assert income.withholding_rate == 0.10

    def test_cola_adjustment(self):
        """Test COLA adjustment calculation."""
        income = RetirementIncome(
            name="test",
            annual_amount=30000,
            cola_rate=0.02,
            income_type="pension",
            person="primary",
        )

        # Base year
        amount = income.get_cola_adjusted_amount(2024, 2024)
        assert amount == 30000

        # Future year with COLA
        amount = income.get_cola_adjusted_amount(2025, 2024)
        expected = 30000 * 1.02
        assert abs(amount - expected) < 0.01


class TestVariableIncome:
    """Test the VariableIncome class."""

    def test_variable_income_creation(self):
        """Test variable income creation."""
        income = VariableIncome(
            name="commission_work",
            annual_amount=60000,
            person="primary",
            variability_type="commission",
            base_amount=40000,
            variable_amount=20000,
            variability_factor=1.2,
        )

        assert income.name == "commission_work"
        assert income.annual_amount == 60000
        assert income.person == "primary"
        assert income.variability_type == "commission"
        assert income.base_amount == 40000
        assert income.variable_amount == 20000
        assert income.variability_factor == 1.2
        assert income.tax_treatment == "other_income"
        assert income.withholding_rate == 0.25

    def test_variable_amount_calculation(self):
        """Test variable amount calculation."""
        income = VariableIncome(
            name="test",
            annual_amount=60000,
            person="primary",
            variability_type="commission",
            base_amount=40000,
            variable_amount=20000,
            variability_factor=1.5,
        )

        amount = income.get_variable_amount(2025)
        expected = 40000 + (20000 * 1.5)
        assert amount == expected


class TestIncomeChangeEvent:
    """Test the IncomeChangeEvent class."""

    def test_income_change_event_creation(self):
        """Test income change event creation."""
        event = IncomeChangeEvent(
            year=2025,
            income_source_name="primary_salary",
            change_type="raise",
            new_amount=90000,
            description="Annual raise and promotion",
        )

        assert event.year == 2025
        assert event.income_source_name == "primary_salary"
        assert event.change_type == "raise"
        assert event.new_amount == 90000
        assert event.description == "Annual raise and promotion"

    def test_income_gap_event(self):
        """Test income gap event."""
        event = IncomeChangeEvent(
            year=2026,
            income_source_name="primary_salary",
            change_type="income_gap",
            description="Sabbatical year",
        )

        assert event.change_type == "income_gap"
        assert event.new_amount is None


class TestIncomeEngine:
    """Test the IncomeEngine class."""

    @pytest.fixture
    def time_grid(self):
        """Create a time grid for testing."""
        return TimeGrid(
            start_year=2024,
            end_year=2060,
            base_year=2024,
            time_unit="annual",
        )

    @pytest.fixture
    def inflation_adjuster(self):
        """Create an inflation adjuster for testing."""
        return InflationAdjuster(base_year=2024, inflation_rate=0.025)

    @pytest.fixture
    def income_engine(self, time_grid, inflation_adjuster):
        """Create an income engine for testing."""
        return IncomeEngine(
            time_grid=time_grid,
            inflation_adjuster=inflation_adjuster,
        )

    def test_engine_creation(self, income_engine):
        """Test income engine creation."""
        assert len(income_engine.income_categories) == 0
        assert len(income_engine.income_change_events) == 0

    def test_add_income_category(self, income_engine):
        """Test adding income categories."""
        category = IncomeCategory(name="test_income", annual_amount=50000)
        income_engine.add_income_category(category)

        assert len(income_engine.income_categories) == 1
        assert income_engine.income_categories[0].name == "test_income"

    def test_add_income_change_event(self, income_engine):
        """Test adding income change events."""
        event = IncomeChangeEvent(
            year=2025,
            income_source_name="test_income",
            change_type="raise",
            new_amount=60000,
            description="Annual raise",
        )
        income_engine.add_income_change_event(event)

        assert len(income_engine.income_change_events) == 1
        assert income_engine.income_change_events[0].year == 2025

    def test_get_annual_income_no_categories(self, income_engine):
        """Test getting annual income with no categories."""
        income = income_engine.get_annual_income(2025)
        assert income == {}

    def test_get_annual_income_with_categories(self, income_engine):
        """Test getting annual income with categories."""
        category = IncomeCategory(
            name="salary", annual_amount=80000, start_year=2024, end_year=2030
        )
        income_engine.add_income_category(category)

        income = income_engine.get_annual_income(2025)
        assert "salary" in income
        assert income["salary"] > 80000  # Should include inflation/growth

    def test_get_annual_income_timing_constraints(self, income_engine):
        """Test income timing constraints."""
        category = IncomeCategory(
            name="contract", annual_amount=60000, start_year=2025, end_year=2027
        )
        income_engine.add_income_category(category)

        # Before start year
        income = income_engine.get_annual_income(2024)
        assert "contract" not in income

        # During active period
        income = income_engine.get_annual_income(2026)
        assert "contract" in income

        # After end year
        income = income_engine.get_annual_income(2028)
        assert "contract" not in income

    def test_get_total_annual_income(self, income_engine):
        """Test getting total annual income."""
        category1 = IncomeCategory(name="salary", annual_amount=80000)
        category2 = IncomeCategory(name="bonus", annual_amount=10000)
        income_engine.add_income_category(category1)
        income_engine.add_income_category(category2)

        total = income_engine.get_total_annual_income(2025)
        assert total > 90000  # Should include both categories with adjustments

    def test_get_income_by_tax_treatment(self, income_engine):
        """Test getting income grouped by tax treatment."""
        employment = EmploymentIncome(
            name="salary", annual_amount=80000, person="primary"
        )
        investment = InvestmentIncome(
            name="dividends", annual_amount=5000, income_type="dividends"
        )
        income_engine.add_income_category(employment)
        income_engine.add_income_category(investment)

        tax_groups = income_engine.get_income_by_tax_treatment(2025)
        assert "w2_employment" in tax_groups
        assert "investment_income" in tax_groups
        assert tax_groups["w2_employment"] > 80000
        assert tax_groups["investment_income"] > 5000

    def test_get_income_series(self, income_engine):
        """Test getting income series."""
        category = IncomeCategory(
            name="salary", annual_amount=80000, start_year=2024, end_year=2026
        )
        income_engine.add_income_category(category)

        series = income_engine.get_income_series(2024, 2026)
        assert "salary" in series
        assert len(series["salary"]) == 3  # 2024, 2025, 2026
        assert all(amount > 0 for amount in series["salary"])

    def test_get_total_income_series(self, income_engine):
        """Test getting total income series."""
        category1 = IncomeCategory(name="salary", annual_amount=80000)
        category2 = IncomeCategory(name="bonus", annual_amount=10000)
        income_engine.add_income_category(category1)
        income_engine.add_income_category(category2)

        total_series = income_engine.get_total_income_series(2024, 2025)
        assert len(total_series) == 2
        assert total_series[0] == 90000  # Base year: 80000 + 10000
        assert total_series[1] > 90000  # Future year with adjustments

    def test_employment_income_with_bonus(self, income_engine):
        """Test employment income with bonus calculation."""
        employment = EmploymentIncome(
            name="salary",
            annual_amount=80000,
            person="primary",
            bonus_amount=10000,
            bonus_frequency="annual",
            bonus_start_year=2025,
        )
        income_engine.add_income_category(employment)

        # Before bonus starts
        income = income_engine.get_annual_income(2024)
        assert income["salary"] == 80000  # Base salary in base year

        # After bonus starts
        income = income_engine.get_annual_income(2025)
        assert income["salary"] > 90000  # Base salary + bonus with adjustments

    def test_income_change_events(self, income_engine):
        """Test income change events."""
        category = IncomeCategory(name="salary", annual_amount=80000)
        income_engine.add_income_category(category)

        # Add a raise event
        event = IncomeChangeEvent(
            year=2025,
            income_source_name="salary",
            change_type="raise",
            new_amount=90000,
            description="Annual raise",
        )
        income_engine.add_income_change_event(event)

        # Before the change
        income = income_engine.get_annual_income(2024)
        assert income["salary"] == 80000  # Original amount in base year

        # After the change
        income = income_engine.get_annual_income(2025)
        assert (
            income["salary"] == 90000
        )  # New amount (no adjustments applied to changed amount)

    def test_income_gap_event(self, income_engine):
        """Test income gap event."""
        category = IncomeCategory(name="salary", annual_amount=80000)
        income_engine.add_income_category(category)

        # Add a gap event
        event = IncomeChangeEvent(
            year=2025,
            income_source_name="salary",
            change_type="income_gap",
            description="Sabbatical year",
        )
        income_engine.add_income_change_event(event)

        # During the gap
        income = income_engine.get_annual_income(2025)
        assert income["salary"] == 0.0

        # After the gap
        income = income_engine.get_annual_income(2026)
        assert income["salary"] > 80000  # Back to normal


class TestIncomeEngineIntegration:
    """Test income engine integration functions."""

    @pytest.fixture
    def sample_scenario(self):
        """Create a sample scenario for testing."""
        household = Household(
            primary_age=35,
            spouse_age=33,
            filing_status="married_filing_jointly",
            state="CA",
        )

        salary_income = [
            Salary(
                person="primary",
                annual_amount=80000,
                start_year=2024,
                end_year=2054,
                growth_rate=0.03,
                bonus=10000,
            ),
            Salary(
                person="spouse",
                annual_amount=60000,
                start_year=2024,
                end_year=2052,
                growth_rate=0.025,
                bonus=5000,
            ),
        ]

        other_income = [
            OtherIncome(
                name="rental_income",
                annual_amount=12000,
                start_year=2025,
                end_year=2040,
                growth_rate=0.02,
            ),
        ]

        incomes = Incomes(
            salary=salary_income,
            social_security=[],
            pension=[],
            other=other_income,
        )

        strategy = Strategy()

        market_model = MarketModel(
            expected_returns=ExpectedReturns(stocks=0.07, bonds=0.03, cash=0.01),
            volatility=Volatility(stocks=0.15, bonds=0.05, cash=0.01),
            inflation=0.025,
        )

        return Scenario(
            household=household,
            accounts=Accounts(),
            liabilities=Liabilities(),
            incomes=incomes,
            expenses=Expenses(),
            policies=Policies(),
            strategy=strategy,
            market_model=market_model,
        )

    def test_create_engine_from_scenario(self, sample_scenario):
        """Test creating engine from scenario."""
        time_grid = TimeGrid(start_year=2024, end_year=2060, base_year=2024)
        inflation_adjuster = InflationAdjuster(base_year=2024, inflation_rate=0.025)

        engine = create_income_engine_from_scenario(
            sample_scenario, time_grid, inflation_adjuster
        )

        assert len(engine.income_categories) == 3  # 2 salaries + 1 other income
        assert any(cat.name == "primary_salary" for cat in engine.income_categories)
        assert any(cat.name == "spouse_salary" for cat in engine.income_categories)
        assert any(cat.name == "rental_income" for cat in engine.income_categories)

    def test_validate_income_timing_valid(self):
        """Test validation with valid timing."""
        time_grid = TimeGrid(start_year=2024, end_year=2060, base_year=2024)

        categories = [
            IncomeCategory(
                name="salary",
                annual_amount=80000,
                start_year=2024,
                end_year=2054,
                growth_rate=0.03,
            ),
            IncomeCategory(
                name="bonus",
                annual_amount=10000,
                start_year=2025,
                end_year=2030,
                growth_rate=0.0,
            ),
        ]

        errors = validate_income_timing(categories, time_grid)
        assert len(errors) == 0

    def test_validate_income_timing_invalid(self):
        """Test validation with invalid timing."""
        time_grid = TimeGrid(start_year=2024, end_year=2060, base_year=2024)

        categories = [
            IncomeCategory(
                name="early_income",
                annual_amount=50000,
                start_year=2020,  # Before time grid
                end_year=2025,
            ),
            IncomeCategory(
                name="late_income",
                annual_amount=50000,
                start_year=2055,
                end_year=2070,  # After time grid
            ),
            IncomeCategory(
                name="excessive_growth",
                annual_amount=50000,
                growth_rate=0.8,  # 80% growth rate
            ),
            IncomeCategory(
                name="excessive_withholding",
                annual_amount=50000,
                withholding_rate=0.8,  # 80% withholding
            ),
        ]

        errors = validate_income_timing(categories, time_grid)
        assert len(errors) == 4
        assert "starts before time grid" in errors[0]
        assert "ends after time grid" in errors[1]
        assert "excessive growth rate" in errors[2]
        assert "excessive withholding rate" in errors[3]

    def test_calculate_income_present_value(self):
        """Test present value calculation."""
        income_series = [80000, 82000, 84000, 86000, 88000]
        present_value = calculate_income_present_value(income_series, 0.03, 2024)
        assert present_value > 0
        assert present_value < sum(income_series)  # Should be less due to discounting

    def test_calculate_income_present_value_empty(self):
        """Test present value calculation with empty series."""
        income_series = []
        present_value = calculate_income_present_value(income_series, 0.03, 2024)
        assert present_value == 0.0

    def test_calculate_income_present_value_zero_discount(self):
        """Test present value calculation with zero discount rate."""
        income_series = [80000, 82000, 84000]
        present_value = calculate_income_present_value(income_series, 0.0, 2024)
        assert present_value == sum(income_series)


class TestIncomeEngineEdgeCases:
    """Test edge cases and error conditions."""

    def test_engine_with_no_income(self):
        """Test engine with no income categories."""
        time_grid = TimeGrid(start_year=2024, end_year=2060, base_year=2024)
        inflation_adjuster = InflationAdjuster(base_year=2024, inflation_rate=0.025)

        engine = IncomeEngine(
            time_grid=time_grid,
            inflation_adjuster=inflation_adjuster,
        )

        # Should return empty results
        income = engine.get_annual_income(2025)
        assert income == {}

        total = engine.get_total_annual_income(2025)
        assert total == 0.0

        tax_groups = engine.get_income_by_tax_treatment(2025)
        assert tax_groups == {}

    def test_engine_with_high_inflation(self):
        """Test engine with high inflation rate."""
        time_grid = TimeGrid(start_year=2024, end_year=2060, base_year=2024)
        inflation_adjuster = InflationAdjuster(
            base_year=2024, inflation_rate=0.10
        )  # 10% inflation

        engine = IncomeEngine(
            time_grid=time_grid,
            inflation_adjuster=inflation_adjuster,
        )

        category = IncomeCategory(
            name="salary", annual_amount=80000, inflation_adjusted=True
        )
        engine.add_income_category(category)

        # Test high inflation impact
        income = engine.get_annual_income(2024)
        assert income["salary"] == 80000

        income = engine.get_annual_income(2025)
        expected = 80000 * 1.10  # 10% inflation
        assert abs(income["salary"] - expected) < 0.01

    def test_engine_with_negative_growth(self):
        """Test engine with negative growth rate."""
        time_grid = TimeGrid(start_year=2024, end_year=2060, base_year=2024)
        inflation_adjuster = InflationAdjuster(base_year=2024, inflation_rate=0.025)

        engine = IncomeEngine(
            time_grid=time_grid,
            inflation_adjuster=inflation_adjuster,
        )

        category = IncomeCategory(
            name="declining_income",
            annual_amount=80000,
            growth_rate=-0.02,  # -2% growth
        )
        engine.add_income_category(category)

        # Test declining income
        income = engine.get_annual_income(2024)
        assert income["declining_income"] == 80000

        income = engine.get_annual_income(2025)
        # With inflation (2.5%) and negative growth (-2%), net effect is 0.5% increase
        expected = 80000 * 1.025 * 0.98  # Inflation * growth
        assert abs(income["declining_income"] - expected) < 0.01

    def test_engine_validation_errors(self):
        """Test engine validation with invalid parameters."""
        time_grid = TimeGrid(start_year=2024, end_year=2060, base_year=2024)
        inflation_adjuster = InflationAdjuster(base_year=2024, inflation_rate=0.025)

        # Test negative annual amount
        with pytest.raises(ValueError):
            IncomeCategory(
                name="invalid", annual_amount=-1000  # Invalid negative amount
            )

        # Test invalid withholding rate
        with pytest.raises(ValueError):
            IncomeCategory(
                name="invalid", annual_amount=50000, withholding_rate=1.5  # Invalid > 1
            )

        # Test invalid tax treatment
        with pytest.raises(ValueError):
            IncomeCategory(
                name="invalid", annual_amount=50000, tax_treatment="invalid"  # type: ignore
            )
