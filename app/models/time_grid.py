"""
Time grid and unit system for retirement planning calculations.

This module provides utilities for handling time periods, inflation adjustments,
and real vs nominal value calculations for the retirement planning engine.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class TimeGrid(BaseModel):
    """Time grid for retirement planning projections."""

    start_year: int = Field(
        ..., ge=1900, le=2100, description="Start year for projections"
    )
    end_year: int = Field(..., ge=1900, le=2100, description="End year for projections")
    base_year: int = Field(
        ..., ge=1900, le=2100, description="Base year for inflation calculations"
    )

    @field_validator("end_year")
    @classmethod
    def validate_end_year(cls, v: int, info: ValidationInfo) -> int:
        if "start_year" in info.data and v < info.data["start_year"]:
            raise ValueError("End year must be >= start year")
        return v

    @field_validator("base_year")
    @classmethod
    def validate_base_year(cls, v: int, info: ValidationInfo) -> int:
        start_year = info.data.get("start_year", v)
        end_year = info.data.get("end_year", v)
        if not (start_year <= v <= end_year):
            raise ValueError("Base year must be within the projection period")
        return v

    def get_years(self) -> List[int]:
        """Get list of years in the time grid."""
        return list(range(self.start_year, self.end_year + 1))

    def get_year_index(self, year: int) -> int:
        """Get the index of a year in the time grid."""
        if not self.start_year <= year <= self.end_year:
            raise ValueError(f"Year {year} is outside the time grid range")
        return year - self.start_year

    def get_years_from_base(self, year: int) -> int:
        """Get the number of years from the base year."""
        return year - self.base_year

    def __len__(self) -> int:
        """Get the number of years in the time grid."""
        return self.end_year - self.start_year + 1


class InflationAdjuster(BaseModel):
    """Handles inflation adjustments for real vs nominal values."""

    inflation_rate: float = Field(
        ..., ge=0, le=1, description="Annual inflation rate (0-1)"
    )
    base_year: int = Field(
        ..., ge=1900, le=2100, description="Base year for calculations"
    )

    def adjust_for_inflation(
        self, amount: float, from_year: int, to_year: int
    ) -> float:
        """
        Adjust an amount for inflation between two years.

        Args:
            amount: The amount to adjust
            from_year: The year the amount is from
            to_year: The year to adjust to

        Returns:
            The inflation-adjusted amount
        """
        years_diff = to_year - from_year
        if years_diff == 0:
            return amount

        # Calculate inflation factor: (1 + inflation_rate) ^ years
        inflation_factor = (1 + self.inflation_rate) ** years_diff

        if years_diff > 0:
            # Forward in time: multiply by inflation factor (real to nominal)
            return amount * inflation_factor
        else:
            # Backward in time: multiply by inflation factor (nominal to real)
            # Note: inflation_factor is < 1 when years_diff < 0
            return amount * inflation_factor

    def to_real_value(self, nominal_amount: float, year: int) -> float:
        """Convert nominal value to real value (base year dollars)."""
        return self.adjust_for_inflation(nominal_amount, year, self.base_year)

    def to_nominal_value(self, real_amount: float, year: int) -> float:
        """Convert real value to nominal value (year dollars)."""
        return self.adjust_for_inflation(real_amount, self.base_year, year)


class CurrencyFormatter(BaseModel):
    """Formats currency and percentage values for display."""

    currency_symbol: str = Field(default="$", description="Currency symbol")
    decimal_places: int = Field(
        default=2, ge=0, le=10, description="Number of decimal places"
    )
    thousands_separator: str = Field(default=",", description="Thousands separator")
    show_currency_symbol: bool = Field(
        default=True, description="Whether to show currency symbol"
    )

    def format_currency(self, amount: float, show_symbol: Optional[bool] = None) -> str:
        """
        Format a currency amount for display.

        Args:
            amount: The amount to format
            show_symbol: Override the default symbol display setting

        Returns:
            Formatted currency string
        """
        show_symbol = (
            show_symbol if show_symbol is not None else self.show_currency_symbol
        )

        # Round to specified decimal places
        rounded = round(amount, self.decimal_places)

        # Format with thousands separator
        if self.decimal_places > 0:
            formatted = f"{rounded:,.{self.decimal_places}f}"
        else:
            formatted = f"{int(rounded):,}"

        # Add currency symbol if requested
        if show_symbol:
            return f"{self.currency_symbol}{formatted}"
        else:
            return formatted

    def format_percentage(self, rate: float, decimal_places: int = 2) -> str:
        """
        Format a percentage for display.

        Args:
            rate: The rate as a decimal (0.05 = 5%)
            decimal_places: Number of decimal places to show

        Returns:
            Formatted percentage string
        """
        percentage = rate * 100
        return f"{percentage:.{decimal_places}f}%"


class UnitSystem(BaseModel):
    """Handles real vs nominal value calculations and formatting."""

    time_grid: TimeGrid = Field(..., description="Time grid for calculations")
    inflation_adjuster: InflationAdjuster = Field(
        ..., description="Inflation adjustment utilities"
    )
    currency_formatter: CurrencyFormatter = Field(
        default_factory=CurrencyFormatter, description="Currency formatting utilities"
    )
    display_mode: Literal["real", "nominal"] = Field(
        default="real", description="Current display mode"
    )

    def toggle_display_mode(self) -> Literal["real", "nominal"]:
        """Toggle between real and nominal display modes."""
        self.display_mode = "nominal" if self.display_mode == "real" else "real"
        return self.display_mode

    def set_display_mode(self, mode: Literal["real", "nominal"]) -> None:
        """Set the display mode."""
        if mode not in ["real", "nominal"]:
            raise ValueError("Display mode must be 'real' or 'nominal'")
        self.display_mode = mode

    def get_display_value(self, amount: float, year: int) -> float:
        """
        Get the value in the current display mode.

        Args:
            amount: The amount (assumed to be in real/base year dollars)
            year: The year for the calculation

        Returns:
            The amount in the current display mode
        """
        if self.display_mode == "real":
            return amount
        else:
            return self.inflation_adjuster.to_nominal_value(amount, year)

    def format_display_value(
        self, amount: float, year: int, show_symbol: Optional[bool] = None
    ) -> str:
        """
        Format a value for display in the current mode.

        Args:
            amount: The amount (assumed to be in real/base year dollars)
            year: The year for the calculation
            show_symbol: Override currency symbol display

        Returns:
            Formatted string for display
        """
        display_amount = self.get_display_value(amount, year)
        return self.currency_formatter.format_currency(display_amount, show_symbol)

    def get_inflation_adjusted_series(
        self, amounts: List[float], years: List[int]
    ) -> List[float]:
        """
        Get inflation-adjusted series for a list of amounts and years.

        Args:
            amounts: List of amounts (assumed to be in real/base year dollars)
            years: List of corresponding years

        Returns:
            List of amounts in the current display mode
        """
        if len(amounts) != len(years):
            raise ValueError("Amounts and years lists must have the same length")

        return [
            self.get_display_value(amount, year) for amount, year in zip(amounts, years)
        ]


def create_time_grid(
    start_year: int,
    end_year: int,
    base_year: Optional[int] = None,
    inflation_rate: float = 0.025,
) -> UnitSystem:
    """
    Create a complete unit system with time grid and inflation adjustment.

    Args:
        start_year: Start year for projections
        end_year: End year for projections
        base_year: Base year for inflation calculations (defaults to start_year)
        inflation_rate: Annual inflation rate (default 2.5%)

    Returns:
        Complete UnitSystem instance
    """
    if base_year is None:
        base_year = start_year

    time_grid = TimeGrid(start_year=start_year, end_year=end_year, base_year=base_year)

    inflation_adjuster = InflationAdjuster(
        inflation_rate=inflation_rate, base_year=base_year
    )

    return UnitSystem(time_grid=time_grid, inflation_adjuster=inflation_adjuster)


def validate_year_range(year: int, min_year: int = 1900, max_year: int = 2100) -> bool:
    """
    Validate that a year is within acceptable range.

    Args:
        year: Year to validate
        min_year: Minimum acceptable year
        max_year: Maximum acceptable year

    Returns:
        True if year is valid

    Raises:
        ValueError: If year is outside acceptable range
    """
    if not min_year <= year <= max_year:
        raise ValueError(f"Year {year} must be between {min_year} and {max_year}")
    return True


def calculate_years_between(start_year: int, end_year: int) -> int:
    """
    Calculate the number of years between two years (inclusive).

    Args:
        start_year: Starting year
        end_year: Ending year

    Returns:
        Number of years between (inclusive)
    """
    return end_year - start_year + 1


def get_current_year() -> int:
    """Get the current year."""
    return datetime.now().year
