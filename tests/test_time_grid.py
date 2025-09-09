"""
Tests for time grid and unit system functionality.

This module tests the time grid, inflation adjustment, currency formatting,
and real vs nominal value calculations.
"""

from decimal import Decimal

import pytest

from app.models.time_grid import (
    CurrencyFormatter,
    InflationAdjuster,
    TimeGrid,
    UnitSystem,
    calculate_years_between,
    create_time_grid,
    get_current_year,
    validate_year_range,
)


class TestTimeGrid:
    """Test TimeGrid functionality."""

    def test_time_grid_creation(self):
        """Test basic time grid creation."""
        grid = TimeGrid(start_year=2024, end_year=2070, base_year=2024)

        assert grid.start_year == 2024
        assert grid.end_year == 2070
        assert grid.base_year == 2024
        assert len(grid) == 47

    def test_time_grid_validation(self):
        """Test time grid validation."""
        # Valid grid
        grid = TimeGrid(start_year=2024, end_year=2070, base_year=2024)
        assert grid.start_year == 2024

        # Invalid: end year before start year
        with pytest.raises(ValueError, match="End year must be >= start year"):
            TimeGrid(start_year=2070, end_year=2024, base_year=2024)

        # Invalid: base year outside range
        with pytest.raises(
            ValueError, match="Base year must be within the projection period"
        ):
            TimeGrid(start_year=2024, end_year=2070, base_year=2020)

    def test_get_years(self):
        """Test getting years list."""
        grid = TimeGrid(start_year=2024, end_year=2026, base_year=2024)
        years = grid.get_years()

        assert years == [2024, 2025, 2026]

    def test_get_year_index(self):
        """Test getting year index."""
        grid = TimeGrid(start_year=2024, end_year=2026, base_year=2024)

        assert grid.get_year_index(2024) == 0
        assert grid.get_year_index(2025) == 1
        assert grid.get_year_index(2026) == 2

        # Invalid year
        with pytest.raises(
            ValueError, match="Year 2023 is outside the time grid range"
        ):
            grid.get_year_index(2023)

    def test_get_years_from_base(self):
        """Test getting years from base year."""
        grid = TimeGrid(start_year=2024, end_year=2070, base_year=2025)

        assert grid.get_years_from_base(2025) == 0
        assert grid.get_years_from_base(2026) == 1
        assert grid.get_years_from_base(2024) == -1


class TestInflationAdjuster:
    """Test InflationAdjuster functionality."""

    def test_inflation_adjuster_creation(self):
        """Test basic inflation adjuster creation."""
        adjuster = InflationAdjuster(inflation_rate=0.025, base_year=2024)

        assert adjuster.inflation_rate == 0.025
        assert adjuster.base_year == 2024

    def test_adjust_for_inflation_forward(self):
        """Test inflation adjustment forward in time."""
        adjuster = InflationAdjuster(inflation_rate=0.025, base_year=2024)

        # $1000 in 2024 should be $1025 in 2025
        adjusted = adjuster.adjust_for_inflation(1000, 2024, 2025)
        assert abs(adjusted - 1025) < 0.01

        # $1000 in 2024 should be $1050.625 in 2026 (2 years)
        adjusted = adjuster.adjust_for_inflation(1000, 2024, 2026)
        expected = 1000 * (1.025**2)
        assert abs(adjusted - expected) < 0.01

    def test_adjust_for_inflation_backward(self):
        """Test inflation adjustment backward in time."""
        adjuster = InflationAdjuster(inflation_rate=0.025, base_year=2024)

        # $1025 in 2025 should be $1000 in 2024
        adjusted = adjuster.adjust_for_inflation(1025, 2025, 2024)
        assert abs(adjusted - 1000) < 0.01

    def test_adjust_for_inflation_same_year(self):
        """Test inflation adjustment for same year."""
        adjuster = InflationAdjuster(inflation_rate=0.025, base_year=2024)

        adjusted = adjuster.adjust_for_inflation(1000, 2024, 2024)
        assert adjusted == 1000

    def test_to_real_value(self):
        """Test conversion to real value."""
        adjuster = InflationAdjuster(inflation_rate=0.025, base_year=2024)

        # $1025 in 2025 should be $1000 in real (2024) dollars
        real_value = adjuster.to_real_value(1025, 2025)
        assert abs(real_value - 1000) < 0.01

    def test_to_nominal_value(self):
        """Test conversion to nominal value."""
        adjuster = InflationAdjuster(inflation_rate=0.025, base_year=2024)

        # $1000 in real (2024) dollars should be $1025 in 2025
        nominal_value = adjuster.to_nominal_value(1000, 2025)
        assert abs(nominal_value - 1025) < 0.01


class TestCurrencyFormatter:
    """Test CurrencyFormatter functionality."""

    def test_currency_formatter_creation(self):
        """Test basic currency formatter creation."""
        formatter = CurrencyFormatter()

        assert formatter.currency_symbol == "$"
        assert formatter.decimal_places == 2
        assert formatter.thousands_separator == ","
        assert formatter.show_currency_symbol is True

    def test_format_currency_basic(self):
        """Test basic currency formatting."""
        formatter = CurrencyFormatter()

        assert formatter.format_currency(1000) == "$1,000.00"
        assert formatter.format_currency(1234.56) == "$1,234.56"
        assert formatter.format_currency(0) == "$0.00"

    def test_format_currency_no_symbol(self):
        """Test currency formatting without symbol."""
        formatter = CurrencyFormatter()

        assert formatter.format_currency(1000, show_symbol=False) == "1,000.00"
        assert formatter.format_currency(1000, show_symbol=True) == "$1,000.00"

    def test_format_currency_custom_decimal_places(self):
        """Test currency formatting with custom decimal places."""
        formatter = CurrencyFormatter(decimal_places=0)

        assert formatter.format_currency(1234.56) == "$1,235"  # Rounded
        assert formatter.format_currency(1234.4) == "$1,234"  # Rounded down

    def test_format_percentage(self):
        """Test percentage formatting."""
        formatter = CurrencyFormatter()

        assert formatter.format_percentage(0.05) == "5.00%"
        assert formatter.format_percentage(0.025) == "2.50%"
        assert formatter.format_percentage(0.1, decimal_places=1) == "10.0%"


class TestUnitSystem:
    """Test UnitSystem functionality."""

    def test_unit_system_creation(self):
        """Test basic unit system creation."""
        time_grid = TimeGrid(start_year=2024, end_year=2070, base_year=2024)
        inflation_adjuster = InflationAdjuster(inflation_rate=0.025, base_year=2024)

        unit_system = UnitSystem(
            time_grid=time_grid, inflation_adjuster=inflation_adjuster
        )

        assert unit_system.display_mode == "real"
        assert unit_system.time_grid.start_year == 2024

    def test_toggle_display_mode(self):
        """Test toggling display mode."""
        unit_system = create_time_grid(2024, 2070, 2024, 0.025)

        assert unit_system.display_mode == "real"

        mode = unit_system.toggle_display_mode()
        assert mode == "nominal"
        assert unit_system.display_mode == "nominal"

        mode = unit_system.toggle_display_mode()
        assert mode == "real"
        assert unit_system.display_mode == "real"

    def test_set_display_mode(self):
        """Test setting display mode."""
        unit_system = create_time_grid(2024, 2070, 2024, 0.025)

        unit_system.set_display_mode("nominal")
        assert unit_system.display_mode == "nominal"

        unit_system.set_display_mode("real")
        assert unit_system.display_mode == "real"

        # Invalid mode
        with pytest.raises(
            ValueError, match="Display mode must be 'real' or 'nominal'"
        ):
            unit_system.set_display_mode("invalid")

    def test_get_display_value_real(self):
        """Test getting display value in real mode."""
        unit_system = create_time_grid(2024, 2070, 2024, 0.025)
        unit_system.set_display_mode("real")

        # In real mode, amount should be unchanged
        assert unit_system.get_display_value(1000, 2025) == 1000
        assert unit_system.get_display_value(1000, 2024) == 1000

    def test_get_display_value_nominal(self):
        """Test getting display value in nominal mode."""
        unit_system = create_time_grid(2024, 2070, 2024, 0.025)
        unit_system.set_display_mode("nominal")

        # In nominal mode, amount should be inflation-adjusted
        # $1000 in 2024 real dollars should be $1025 in 2025 nominal
        nominal_value = unit_system.get_display_value(1000, 2025)
        assert abs(nominal_value - 1025) < 0.01

    def test_format_display_value(self):
        """Test formatting display value."""
        unit_system = create_time_grid(2024, 2070, 2024, 0.025)

        # Real mode
        unit_system.set_display_mode("real")
        formatted = unit_system.format_display_value(1000, 2025)
        assert formatted == "$1,000.00"

        # Nominal mode
        unit_system.set_display_mode("nominal")
        formatted = unit_system.format_display_value(1000, 2025)
        assert formatted == "$1,025.00"

    def test_get_inflation_adjusted_series(self):
        """Test getting inflation-adjusted series."""
        unit_system = create_time_grid(2024, 2070, 2024, 0.025)
        unit_system.set_display_mode("nominal")

        amounts = [1000, 1000, 1000]
        years = [2024, 2025, 2026]

        adjusted_series = unit_system.get_inflation_adjusted_series(amounts, years)

        # Should be inflation-adjusted for each year
        assert abs(adjusted_series[0] - 1000) < 0.01  # 2024: no adjustment
        assert abs(adjusted_series[1] - 1025) < 0.01  # 2025: 2.5% inflation
        assert abs(adjusted_series[2] - 1050.625) < 0.01  # 2026: 5.0625% inflation

    def test_get_inflation_adjusted_series_mismatched_lengths(self):
        """Test error handling for mismatched lengths."""
        unit_system = create_time_grid(2024, 2070, 2024, 0.025)

        amounts = [1000, 1000]
        years = [2024, 2025, 2026]

        with pytest.raises(
            ValueError, match="Amounts and years lists must have the same length"
        ):
            unit_system.get_inflation_adjusted_series(amounts, years)


class TestCreateTimeGrid:
    """Test create_time_grid helper function."""

    def test_create_time_grid_defaults(self):
        """Test create_time_grid with defaults."""
        unit_system = create_time_grid(2024, 2070)

        assert unit_system.time_grid.start_year == 2024
        assert unit_system.time_grid.end_year == 2070
        assert unit_system.time_grid.base_year == 2024  # Defaults to start_year
        assert unit_system.inflation_adjuster.inflation_rate == 0.025  # Default 2.5%

    def test_create_time_grid_custom(self):
        """Test create_time_grid with custom parameters."""
        unit_system = create_time_grid(2024, 2070, 2025, 0.03)

        assert unit_system.time_grid.start_year == 2024
        assert unit_system.time_grid.end_year == 2070
        assert unit_system.time_grid.base_year == 2025
        assert unit_system.inflation_adjuster.inflation_rate == 0.03


class TestUtilityFunctions:
    """Test utility functions."""

    def test_validate_year_range(self):
        """Test year range validation."""
        # Valid years
        assert validate_year_range(2024) is True
        assert validate_year_range(1900) is True
        assert validate_year_range(2100) is True

        # Invalid years
        with pytest.raises(ValueError, match="Year 1899 must be between 1900 and 2100"):
            validate_year_range(1899)

        with pytest.raises(ValueError, match="Year 2101 must be between 1900 and 2100"):
            validate_year_range(2101)

    def test_validate_year_range_custom(self):
        """Test year range validation with custom range."""
        assert validate_year_range(2024, 2000, 2050) is True

        with pytest.raises(ValueError, match="Year 1999 must be between 2000 and 2050"):
            validate_year_range(1999, 2000, 2050)

    def test_calculate_years_between(self):
        """Test calculating years between two years."""
        assert calculate_years_between(2024, 2024) == 1  # Same year
        assert calculate_years_between(2024, 2025) == 2  # Two years
        assert calculate_years_between(2024, 2026) == 3  # Three years

    def test_get_current_year(self):
        """Test getting current year."""
        current_year = get_current_year()
        assert isinstance(current_year, int)
        assert 2020 <= current_year <= 2030  # Reasonable range for current year


class TestIntegration:
    """Integration tests for the complete time grid system."""

    def test_retirement_planning_scenario(self):
        """Test a realistic retirement planning scenario."""
        # Create a 30-year retirement planning period
        unit_system = create_time_grid(2024, 2054, 2024, 0.025)

        # Test real vs nominal calculations
        annual_withdrawal_real = 50000  # $50k in 2024 dollars

        # In real mode, withdrawal stays constant
        unit_system.set_display_mode("real")
        real_withdrawals = []
        for year in unit_system.time_grid.get_years()[:5]:  # First 5 years
            withdrawal = unit_system.get_display_value(annual_withdrawal_real, year)
            real_withdrawals.append(withdrawal)

        # All should be the same in real mode
        assert all(w == 50000 for w in real_withdrawals)

        # In nominal mode, withdrawal increases with inflation
        unit_system.set_display_mode("nominal")
        nominal_withdrawals = []
        for year in unit_system.time_grid.get_years()[:5]:  # First 5 years
            withdrawal = unit_system.get_display_value(annual_withdrawal_real, year)
            nominal_withdrawals.append(withdrawal)

        # Should increase each year
        assert nominal_withdrawals[0] == 50000  # 2024: no inflation
        assert abs(nominal_withdrawals[1] - 51250) < 1  # 2025: 2.5% inflation
        assert abs(nominal_withdrawals[2] - 52531.25) < 1  # 2026: 5.0625% inflation

    def test_currency_formatting_integration(self):
        """Test currency formatting integration."""
        unit_system = create_time_grid(2024, 2070, 2024, 0.025)

        # Test formatting in both modes
        amount = 1234567.89

        unit_system.set_display_mode("real")
        real_formatted = unit_system.format_display_value(amount, 2025)
        assert real_formatted == "$1,234,567.89"

        unit_system.set_display_mode("nominal")
        nominal_formatted = unit_system.format_display_value(amount, 2025)
        # Should be inflation-adjusted and formatted
        expected_nominal = amount * 1.025
        expected_formatted = f"${expected_nominal:,.2f}"
        assert nominal_formatted == expected_formatted
