"""
Tests for baseline expenses and lumpy events module.

This module tests the expense modeling functionality including regular expense
categories, one-time lumpy events, inflation adjustment, and cashflow integration.
"""

import pytest

from app.models.baseline_expenses import (
    ExpenseCategory,
    ExpenseEngine,
    HealthcareExpenseCategory,
    HousingExpenseCategory,
    LumpyEvent,
    TransportationExpenseCategory,
    calculate_expense_inflation_impact,
    create_expense_engine_from_scenario,
    validate_expense_timing,
)
from app.models.scenario import (
    Expenses,
    HealthcareExpenses,
    HousingExpenses,
    LumpyExpense,
    TransportationExpenses,
)
from app.models.time_grid import InflationAdjuster, TimeGrid


class TestExpenseCategory:
    """Test the base ExpenseCategory class."""

    def test_basic_expense_category(self):
        """Test basic expense category creation."""
        category = ExpenseCategory(
            name="food", annual_amount=12000, inflation_adjusted=True
        )

        assert category.name == "food"
        assert category.annual_amount == 12000
        assert category.inflation_adjusted is True
        assert category.start_year is None
        assert category.end_year is None

    def test_expense_category_with_timing(self):
        """Test expense category with start and end years."""
        category = ExpenseCategory(
            name="education", annual_amount=20000, start_year=2025, end_year=2030
        )

        assert category.start_year == 2025
        assert category.end_year == 2030

    def test_invalid_end_year(self):
        """Test that end year must be >= start year."""
        with pytest.raises(ValueError, match="End year must be >= start year"):
            ExpenseCategory(
                name="test", annual_amount=1000, start_year=2030, end_year=2025
            )


class TestHousingExpenseCategory:
    """Test the HousingExpenseCategory class."""

    def test_housing_category_creation(self):
        """Test housing expense category creation."""
        category = HousingExpenseCategory(
            name="housing",
            mortgage_payment=2000,
            property_tax=6000,
            home_insurance=1200,
            hoa_fees=200,
            maintenance=3000,
            utilities=400,
        )

        assert category.mortgage_payment == 2000
        assert category.property_tax == 6000
        assert category.home_insurance == 1200
        assert category.hoa_fees == 200
        assert category.maintenance == 3000
        assert category.utilities == 400

        # Check calculated annual amount
        expected_annual = (2000 * 12) + 6000 + 1200 + (200 * 12) + 3000 + (400 * 12)
        assert category.annual_amount == expected_annual

    def test_housing_category_with_explicit_annual(self):
        """Test housing category with explicit annual amount."""
        category = HousingExpenseCategory(
            name="housing",
            annual_amount=50000,
            mortgage_payment=2000,
            property_tax=6000,
        )

        assert category.annual_amount == 50000  # Should use explicit value


class TestTransportationExpenseCategory:
    """Test the TransportationExpenseCategory class."""

    def test_transportation_category_creation(self):
        """Test transportation expense category creation."""
        category = TransportationExpenseCategory(
            name="transportation",
            auto_payment=300,
            auto_insurance=1200,
            gas=200,
            maintenance=1500,
        )

        assert category.auto_payment == 300
        assert category.auto_insurance == 1200
        assert category.gas == 200
        assert category.maintenance == 1500

        # Check calculated annual amount
        expected_annual = (300 * 12) + 1200 + (200 * 12) + 1500
        assert category.annual_amount == expected_annual


class TestHealthcareExpenseCategory:
    """Test the HealthcareExpenseCategory class."""

    def test_healthcare_category_creation(self):
        """Test healthcare expense category creation."""
        category = HealthcareExpenseCategory(
            name="healthcare", insurance=500, out_of_pocket=2000, medicare=200
        )

        assert category.insurance == 500
        assert category.out_of_pocket == 2000
        assert category.medicare == 200

        # Check calculated annual amount
        expected_annual = (500 * 12) + 2000 + (200 * 12)
        assert category.annual_amount == expected_annual


class TestLumpyEvent:
    """Test the LumpyEvent class."""

    def test_lumpy_event_creation(self):
        """Test lumpy event creation."""
        event = LumpyEvent(name="New Car", amount=35000, year=2035, category="vehicle")

        assert event.name == "New Car"
        assert event.amount == 35000
        assert event.year == 2035
        assert event.category == "vehicle"
        assert event.inflation_adjusted is True

    def test_lumpy_event_inflation_adjustment(self):
        """Test lumpy event inflation adjustment."""
        inflation_adjuster = InflationAdjuster(inflation_rate=0.025, base_year=2024)
        event = LumpyEvent(
            name="Home Improvement",
            amount=20000,
            year=2030,
            category="home_improvement",
        )

        # Adjust from 2024 to 2030 (6 years)
        adjusted_amount = event.get_inflation_adjusted_amount(2030, inflation_adjuster)
        expected_amount = 20000 * (1.025**6)

        assert abs(adjusted_amount - expected_amount) < 0.01

    def test_lumpy_event_no_inflation(self):
        """Test lumpy event without inflation adjustment."""
        inflation_adjuster = InflationAdjuster(inflation_rate=0.025, base_year=2024)
        event = LumpyEvent(
            name="Fixed Cost",
            amount=10000,
            year=2030,
            category="other",
            inflation_adjusted=False,
        )

        adjusted_amount = event.get_inflation_adjusted_amount(2030, inflation_adjuster)
        assert adjusted_amount == 10000


class TestExpenseEngine:
    """Test the ExpenseEngine class."""

    @pytest.fixture
    def time_grid(self):
        """Create a test time grid."""
        return TimeGrid(start_year=2024, end_year=2030, base_year=2024)

    @pytest.fixture
    def inflation_adjuster(self):
        """Create a test inflation adjuster."""
        return InflationAdjuster(inflation_rate=0.025, base_year=2024)

    @pytest.fixture
    def expense_engine(self, time_grid, inflation_adjuster):
        """Create a test expense engine."""
        return ExpenseEngine(time_grid=time_grid, inflation_adjuster=inflation_adjuster)

    def test_expense_engine_creation(self, expense_engine):
        """Test expense engine creation."""
        assert len(expense_engine.expense_categories) == 0
        assert len(expense_engine.lumpy_events) == 0

    def test_add_expense_category(self, expense_engine):
        """Test adding expense categories."""
        category = ExpenseCategory(name="food", annual_amount=12000)

        expense_engine.add_expense_category(category)
        assert len(expense_engine.expense_categories) == 1
        assert expense_engine.expense_categories[0].name == "food"

    def test_add_lumpy_event(self, expense_engine):
        """Test adding lumpy events."""
        event = LumpyEvent(name="New Car", amount=35000, year=2027, category="vehicle")

        expense_engine.add_lumpy_event(event)
        assert len(expense_engine.lumpy_events) == 1
        assert expense_engine.lumpy_events[0].name == "New Car"

    def test_get_annual_expenses_no_expenses(self, expense_engine):
        """Test getting annual expenses with no expenses configured."""
        expenses = expense_engine.get_annual_expenses(2025)
        assert expenses == {}

    def test_get_annual_expenses_with_categories(self, expense_engine):
        """Test getting annual expenses with regular categories."""
        # Add expense categories
        expense_engine.add_expense_category(
            ExpenseCategory(name="food", annual_amount=12000)
        )
        expense_engine.add_expense_category(
            ExpenseCategory(name="entertainment", annual_amount=6000)
        )

        expenses = expense_engine.get_annual_expenses(2025)

        # Should have inflation-adjusted amounts
        expected_food = 12000 * (1.025**1)  # 1 year from base
        expected_entertainment = 6000 * (1.025**1)

        assert "food" in expenses
        assert "entertainment" in expenses
        assert abs(expenses["food"] - expected_food) < 0.01
        assert abs(expenses["entertainment"] - expected_entertainment) < 0.01

    def test_get_annual_expenses_with_lumpy_events(self, expense_engine):
        """Test getting annual expenses with lumpy events."""
        # Add lumpy event for 2026
        expense_engine.add_lumpy_event(
            LumpyEvent(name="New Car", amount=35000, year=2026, category="vehicle")
        )

        # Test year with lumpy event
        expenses_2026 = expense_engine.get_annual_expenses(2026)
        expected_car = 35000 * (1.025**2)  # 2 years from base

        assert "New Car" in expenses_2026
        assert abs(expenses_2026["New Car"] - expected_car) < 0.01

        # Test year without lumpy event
        expenses_2025 = expense_engine.get_annual_expenses(2025)
        assert "New Car" not in expenses_2025

    def test_get_total_annual_expenses(self, expense_engine):
        """Test getting total annual expenses."""
        # Add expense categories
        expense_engine.add_expense_category(
            ExpenseCategory(name="food", annual_amount=12000)
        )
        expense_engine.add_expense_category(
            ExpenseCategory(name="entertainment", annual_amount=6000)
        )

        # Add lumpy event
        expense_engine.add_lumpy_event(
            LumpyEvent(name="New Car", amount=35000, year=2025, category="vehicle")
        )

        total = expense_engine.get_total_annual_expenses(2025)

        expected_food = 12000 * 1.025
        expected_entertainment = 6000 * 1.025
        expected_car = 35000 * 1.025
        expected_total = expected_food + expected_entertainment + expected_car

        assert abs(total - expected_total) < 0.01

    def test_get_expense_series(self, expense_engine):
        """Test getting expense series for all years."""
        # Add expense category
        expense_engine.add_expense_category(
            ExpenseCategory(name="food", annual_amount=12000)
        )

        # Add lumpy event for 2026
        expense_engine.add_lumpy_event(
            LumpyEvent(name="New Car", amount=35000, year=2026, category="vehicle")
        )

        series = expense_engine.get_expense_series()

        assert "food" in series
        assert "New Car" in series

        # Check food series (should be present for all years)
        food_series = series["food"]
        assert len(food_series) == 7  # 2024-2030

        # Check New Car series (should only be present for 2026)
        car_series = series["New Car"]
        assert len(car_series) == 1
        assert car_series[0][0] == 2026  # Year

    def test_get_total_expense_series(self, expense_engine):
        """Test getting total expense series."""
        # Add expense category
        expense_engine.add_expense_category(
            ExpenseCategory(name="food", annual_amount=12000)
        )

        series = expense_engine.get_total_expense_series()

        assert len(series) == 7  # 2024-2030

        # Check that amounts increase with inflation
        for i in range(1, len(series)):
            assert series[i][1] > series[i - 1][1]  # Amount should increase

    def test_category_timing_constraints(self, expense_engine):
        """Test expense categories with timing constraints."""
        # Add category that starts in 2026
        expense_engine.add_expense_category(
            ExpenseCategory(
                name="education", annual_amount=20000, start_year=2026, end_year=2028
            )
        )

        # Should not be active in 2025
        expenses_2025 = expense_engine.get_annual_expenses(2025)
        assert "education" not in expenses_2025

        # Should be active in 2026
        expenses_2026 = expense_engine.get_annual_expenses(2026)
        assert "education" in expenses_2026

        # Should be active in 2028
        expenses_2028 = expense_engine.get_annual_expenses(2028)
        assert "education" in expenses_2028

        # Should not be active in 2029
        expenses_2029 = expense_engine.get_annual_expenses(2029)
        assert "education" not in expenses_2029


class TestCreateExpenseEngineFromScenario:
    """Test creating expense engine from scenario data."""

    @pytest.fixture
    def time_grid(self):
        """Create a test time grid."""
        return TimeGrid(start_year=2024, end_year=2030, base_year=2024)

    @pytest.fixture
    def inflation_adjuster(self):
        """Create a test inflation adjuster."""
        return InflationAdjuster(inflation_rate=0.025, base_year=2024)

    @pytest.fixture
    def sample_expenses(self):
        """Create sample expenses data."""
        return Expenses(
            housing=HousingExpenses(
                mortgage_payment=2000,
                property_tax=6000,
                home_insurance=1200,
                hoa_fees=200,
                maintenance=3000,
                utilities=400,
            ),
            transportation=TransportationExpenses(
                auto_payment=300, auto_insurance=1200, gas=200, maintenance=1500
            ),
            healthcare=HealthcareExpenses(
                insurance=500, out_of_pocket=2000, medicare=0
            ),
            food=1000,  # Monthly
            entertainment=500,  # Monthly
            travel=8000,  # Annual
            education=0,
            other=800,  # Monthly
            lumpy_expenses=[
                LumpyExpense(
                    name="New Roof",
                    amount=15000,
                    year=2026,
                    category="home_improvement",
                ),
                LumpyExpense(
                    name="New Car", amount=35000, year=2028, category="vehicle"
                ),
            ],
        )

    def test_create_engine_from_scenario(
        self, time_grid, inflation_adjuster, sample_expenses
    ):
        """Test creating expense engine from scenario."""
        engine = create_expense_engine_from_scenario(
            sample_expenses, time_grid, inflation_adjuster
        )

        # Should have created categories for all expense types
        category_names = [cat.name for cat in engine.expense_categories]
        expected_categories = [
            "housing",
            "transportation",
            "healthcare",
            "food",
            "entertainment",
            "travel",
            "other",
        ]

        for expected in expected_categories:
            assert expected in category_names

        # Should have created lumpy events
        assert len(engine.lumpy_events) == 2
        event_names = [event.name for event in engine.lumpy_events]
        assert "New Roof" in event_names
        assert "New Car" in event_names

    def test_housing_category_calculation(
        self, time_grid, inflation_adjuster, sample_expenses
    ):
        """Test that housing category calculates correctly."""
        engine = create_expense_engine_from_scenario(
            sample_expenses, time_grid, inflation_adjuster
        )

        housing_category = next(
            cat for cat in engine.expense_categories if cat.name == "housing"
        )

        # Check that it's a HousingExpenseCategory
        assert isinstance(housing_category, HousingExpenseCategory)

        # Check component values
        assert housing_category.mortgage_payment == 2000
        assert housing_category.property_tax == 6000
        assert housing_category.home_insurance == 1200
        assert housing_category.hoa_fees == 200
        assert housing_category.maintenance == 3000
        assert housing_category.utilities == 400

    def test_monthly_to_annual_conversion(
        self, time_grid, inflation_adjuster, sample_expenses
    ):
        """Test that monthly expenses are converted to annual."""
        engine = create_expense_engine_from_scenario(
            sample_expenses, time_grid, inflation_adjuster
        )

        food_category = next(
            cat for cat in engine.expense_categories if cat.name == "food"
        )
        assert food_category.annual_amount == 1000 * 12  # Monthly to annual

        entertainment_category = next(
            cat for cat in engine.expense_categories if cat.name == "entertainment"
        )
        assert entertainment_category.annual_amount == 500 * 12  # Monthly to annual

        travel_category = next(
            cat for cat in engine.expense_categories if cat.name == "travel"
        )
        assert travel_category.annual_amount == 8000  # Already annual

    def test_zero_expenses_omitted(self, time_grid, inflation_adjuster):
        """Test that zero expenses are omitted."""
        expenses = Expenses(
            housing=HousingExpenses(),  # All zeros
            transportation=TransportationExpenses(),  # All zeros
            healthcare=HealthcareExpenses(),  # All zeros
            food=0,
            entertainment=0,
            travel=0,
            education=0,
            other=0,
            lumpy_expenses=[],
        )

        engine = create_expense_engine_from_scenario(
            expenses, time_grid, inflation_adjuster
        )

        # Should have no expense categories since all are zero
        assert len(engine.expense_categories) == 0
        assert len(engine.lumpy_events) == 0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_calculate_expense_inflation_impact(self):
        """Test inflation impact calculation."""
        base_amount = 10000
        years = 10
        inflation_rate = 0.025

        future_value = calculate_expense_inflation_impact(
            base_amount, years, inflation_rate
        )
        expected_value = 10000 * (1.025**10)

        assert abs(future_value - expected_value) < 0.01

    def test_validate_expense_timing_valid(self):
        """Test valid expense timing validation."""
        time_grid = TimeGrid(start_year=2024, end_year=2030, base_year=2024)

        # Valid cases
        assert validate_expense_timing(2025, 2028, time_grid) is True
        assert validate_expense_timing(None, 2028, time_grid) is True
        assert validate_expense_timing(2025, None, time_grid) is True
        assert validate_expense_timing(None, None, time_grid) is True

    def test_validate_expense_timing_invalid_start(self):
        """Test invalid start year validation."""
        time_grid = TimeGrid(start_year=2024, end_year=2030, base_year=2024)

        with pytest.raises(
            ValueError, match="Start year 2023 is outside time grid range"
        ):
            validate_expense_timing(2023, 2028, time_grid)

        with pytest.raises(
            ValueError, match="Start year 2031 is outside time grid range"
        ):
            validate_expense_timing(2031, 2028, time_grid)

    def test_validate_expense_timing_invalid_end(self):
        """Test invalid end year validation."""
        time_grid = TimeGrid(start_year=2024, end_year=2030, base_year=2024)

        with pytest.raises(
            ValueError, match="End year 2023 is outside time grid range"
        ):
            validate_expense_timing(2025, 2023, time_grid)

        with pytest.raises(
            ValueError, match="End year 2031 is outside time grid range"
        ):
            validate_expense_timing(2025, 2031, time_grid)


class TestIntegrationWithTimeGrid:
    """Test integration with time grid and inflation systems."""

    def test_expense_engine_with_unit_system(self):
        """Test expense engine integration with unit system."""
        from app.models.time_grid import create_time_grid

        # Create unit system
        unit_system = create_time_grid(
            start_year=2024, end_year=2030, base_year=2024, inflation_rate=0.025
        )

        # Create expense engine
        engine = ExpenseEngine(
            time_grid=unit_system.time_grid,
            inflation_adjuster=unit_system.inflation_adjuster,
        )

        # Add expense category
        engine.add_expense_category(ExpenseCategory(name="food", annual_amount=12000))

        # Test that expenses are properly inflation-adjusted
        expenses_2025 = engine.get_annual_expenses(2025)
        expected_food_2025 = 12000 * 1.025  # 1 year of inflation

        assert abs(expenses_2025["food"] - expected_food_2025) < 0.01

        # Test with unit system display in nominal mode (should match expense engine)
        unit_system.set_display_mode("nominal")
        display_value = unit_system.get_display_value(12000, 2025)
        assert abs(display_value - expenses_2025["food"]) < 0.01

    def test_real_vs_nominal_display(self):
        """Test real vs nominal display modes."""
        from app.models.time_grid import create_time_grid

        unit_system = create_time_grid(
            start_year=2024, end_year=2030, base_year=2024, inflation_rate=0.025
        )

        engine = ExpenseEngine(
            time_grid=unit_system.time_grid,
            inflation_adjuster=unit_system.inflation_adjuster,
        )

        engine.add_expense_category(ExpenseCategory(name="food", annual_amount=12000))

        # Test real mode (default) - expense engine always applies inflation
        # The unit system handles real vs nominal display
        unit_system.set_display_mode("real")
        expenses_real = engine.get_annual_expenses(2025)
        expected_real = 12000 * 1.025  # Expense engine applies inflation
        assert abs(expenses_real["food"] - expected_real) < 0.01

        # Test nominal mode
        unit_system.set_display_mode("nominal")
        expenses_nominal = engine.get_annual_expenses(2025)
        expected_nominal = 12000 * 1.025
        assert abs(expenses_nominal["food"] - expected_nominal) < 0.01


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_inflation_rate(self):
        """Test behavior with zero inflation rate."""
        time_grid = TimeGrid(start_year=2024, end_year=2030, base_year=2024)
        inflation_adjuster = InflationAdjuster(inflation_rate=0.0, base_year=2024)

        engine = ExpenseEngine(
            time_grid=time_grid, inflation_adjuster=inflation_adjuster
        )

        engine.add_expense_category(ExpenseCategory(name="food", annual_amount=12000))

        # With zero inflation, amounts should remain constant
        expenses_2025 = engine.get_annual_expenses(2025)
        expenses_2030 = engine.get_annual_expenses(2030)

        assert expenses_2025["food"] == 12000
        assert expenses_2030["food"] == 12000

    def test_negative_expense_amounts(self):
        """Test that negative expense amounts are rejected."""
        with pytest.raises(ValueError):
            ExpenseCategory(name="negative", annual_amount=-1000)

        with pytest.raises(ValueError):
            LumpyEvent(name="negative", amount=-5000, year=2025, category="other")

    def test_invalid_year_ranges(self):
        """Test invalid year ranges."""
        with pytest.raises(ValueError):
            LumpyEvent(
                name="invalid", amount=1000, year=1800, category="other"  # Too early
            )

        with pytest.raises(ValueError):
            LumpyEvent(
                name="invalid", amount=1000, year=2200, category="other"  # Too late
            )

    def test_empty_expense_engine(self):
        """Test expense engine with no expenses."""
        time_grid = TimeGrid(start_year=2024, end_year=2030, base_year=2024)
        inflation_adjuster = InflationAdjuster(inflation_rate=0.025, base_year=2024)

        engine = ExpenseEngine(
            time_grid=time_grid, inflation_adjuster=inflation_adjuster
        )

        # Should return empty results
        assert engine.get_annual_expenses(2025) == {}
        assert engine.get_total_annual_expenses(2025) == 0
        assert engine.get_expense_series() == {}
        assert engine.get_total_expense_series() == [
            (year, 0) for year in range(2024, 2031)
        ]
