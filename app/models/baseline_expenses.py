"""
Baseline expenses and lumpy events module for retirement planning.

This module handles expense modeling including regular expense categories
and one-time lumpy events, with inflation adjustment and cashflow integration.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, ValidationInfo

from .time_grid import TimeGrid, InflationAdjuster
from .scenario import Expenses


class ExpenseCategory(BaseModel):
    """Base class for expense categories."""

    name: str = Field(..., min_length=1, description="Category name")
    annual_amount: float = Field(
        ..., ge=0, description="Annual amount in base year dollars"
    )
    inflation_adjusted: bool = Field(
        default=True, description="Whether to apply inflation"
    )
    start_year: Optional[int] = Field(
        default=None, ge=1900, le=2100, description="Start year (None = all years)"
    )
    end_year: Optional[int] = Field(
        default=None, ge=1900, le=2100, description="End year (None = all years)"
    )

    @field_validator("end_year")
    @classmethod
    def validate_end_year(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        if v is not None and "start_year" in info.data:
            start_year = info.data["start_year"]
            if start_year is not None and v < start_year:
                raise ValueError("End year must be >= start year")
        return v


class HousingExpenseCategory(ExpenseCategory):
    """Housing expense category with detailed breakdown."""

    mortgage_payment: float = Field(
        default=0, ge=0, description="Monthly mortgage payment"
    )
    property_tax: float = Field(default=0, ge=0, description="Annual property tax")
    home_insurance: float = Field(default=0, ge=0, description="Annual home insurance")
    hoa_fees: float = Field(default=0, ge=0, description="Monthly HOA fees")
    maintenance: float = Field(default=0, ge=0, description="Annual maintenance costs")
    utilities: float = Field(default=0, ge=0, description="Monthly utilities")

    def __init__(self, **data: Any) -> None:
        # Calculate total annual amount from components
        if "annual_amount" not in data:
            mortgage_annual = data.get("mortgage_payment", 0) * 12
            hoa_annual = data.get("hoa_fees", 0) * 12
            utilities_annual = data.get("utilities", 0) * 12
            other_annual = (
                data.get("property_tax", 0)
                + data.get("home_insurance", 0)
                + data.get("maintenance", 0)
            )
            data["annual_amount"] = (
                mortgage_annual + hoa_annual + utilities_annual + other_annual
            )
        super().__init__(**data)


class TransportationExpenseCategory(ExpenseCategory):
    """Transportation expense category."""

    auto_payment: float = Field(default=0, ge=0, description="Monthly auto payment")
    auto_insurance: float = Field(default=0, ge=0, description="Annual auto insurance")
    gas: float = Field(default=0, ge=0, description="Monthly gas expenses")
    maintenance: float = Field(default=0, ge=0, description="Annual maintenance costs")

    def __init__(self, **data: Any) -> None:
        # Calculate total annual amount from components
        if "annual_amount" not in data:
            auto_annual = data.get("auto_payment", 0) * 12
            gas_annual = data.get("gas", 0) * 12
            other_annual = data.get("auto_insurance", 0) + data.get("maintenance", 0)
            data["annual_amount"] = auto_annual + gas_annual + other_annual
        super().__init__(**data)


class HealthcareExpenseCategory(ExpenseCategory):
    """Healthcare expense category."""

    insurance: float = Field(default=0, ge=0, description="Monthly insurance premium")
    out_of_pocket: float = Field(
        default=0, ge=0, description="Annual out-of-pocket costs"
    )
    medicare: float = Field(default=0, ge=0, description="Monthly Medicare premium")

    def __init__(self, **data: Any) -> None:
        # Calculate total annual amount from components
        if "annual_amount" not in data:
            insurance_annual = data.get("insurance", 0) * 12
            medicare_annual = data.get("medicare", 0) * 12
            other_annual = data.get("out_of_pocket", 0)
            data["annual_amount"] = insurance_annual + medicare_annual + other_annual
        super().__init__(**data)


class LumpyEvent(BaseModel):
    """One-time or irregular expense event."""

    name: str = Field(..., min_length=1, description="Event name")
    amount: float = Field(..., ge=0, description="Event amount in base year dollars")
    year: int = Field(..., ge=1900, le=2100, description="Year event occurs")
    category: Literal[
        "home_improvement", "vehicle", "medical", "education", "other"
    ] = Field(default="other", description="Event category")
    inflation_adjusted: bool = Field(
        default=True, description="Whether to apply inflation"
    )

    def get_inflation_adjusted_amount(
        self, year: int, inflation_adjuster: InflationAdjuster
    ) -> float:
        """Get the inflation-adjusted amount for the given year."""
        if not self.inflation_adjusted:
            return self.amount

        # Adjust from base year to target year
        return inflation_adjuster.adjust_for_inflation(
            self.amount, inflation_adjuster.base_year, year
        )


class ExpenseEngine(BaseModel):
    """Engine for processing baseline expenses and lumpy events."""

    time_grid: TimeGrid = Field(..., description="Time grid for calculations")
    inflation_adjuster: InflationAdjuster = Field(
        ..., description="Inflation adjustment utilities"
    )
    expense_categories: List[ExpenseCategory] = Field(
        default_factory=list, description="Regular expense categories"
    )
    lumpy_events: List[LumpyEvent] = Field(
        default_factory=list, description="One-time expense events"
    )

    def add_expense_category(self, category: ExpenseCategory) -> None:
        """Add an expense category."""
        self.expense_categories.append(category)

    def add_lumpy_event(self, event: LumpyEvent) -> None:
        """Add a lumpy event."""
        self.lumpy_events.append(event)

    def get_annual_expenses(self, year: int) -> Dict[str, float]:
        """
        Get all expenses for a given year.

        Args:
            year: The year to calculate expenses for

        Returns:
            Dictionary mapping expense names to amounts
        """
        expenses = {}

        # Add regular expense categories
        for category in self.expense_categories:
            if self._is_category_active(category, year):
                amount = category.annual_amount
                if category.inflation_adjusted:
                    amount = self.inflation_adjuster.adjust_for_inflation(
                        amount, self.inflation_adjuster.base_year, year
                    )
                expenses[category.name] = amount

        # Add lumpy events for this year
        for event in self.lumpy_events:
            if event.year == year:
                amount = event.get_inflation_adjusted_amount(
                    year, self.inflation_adjuster
                )
                expenses[event.name] = amount

        return expenses

    def get_total_annual_expenses(self, year: int) -> float:
        """Get total annual expenses for a given year."""
        expenses = self.get_annual_expenses(year)
        return sum(expenses.values())

    def get_expense_series(self) -> Dict[str, List[Tuple[int, float]]]:
        """
        Get expense series for all years in the time grid.

        Returns:
            Dictionary mapping expense names to lists of (year, amount) tuples
        """
        series: Dict[str, List[Tuple[int, float]]] = {}

        for year in self.time_grid.get_years():
            annual_expenses = self.get_annual_expenses(year)
            for name, amount in annual_expenses.items():
                if name not in series:
                    series[name] = []
                series[name].append((year, amount))

        return series

    def get_total_expense_series(self) -> List[Tuple[int, float]]:
        """Get total expense series for all years."""
        return [
            (year, self.get_total_annual_expenses(year))
            for year in self.time_grid.get_years()
        ]

    def _is_category_active(self, category: ExpenseCategory, year: int) -> bool:
        """Check if a category is active in the given year."""
        if category.start_year is not None and year < category.start_year:
            return False
        if category.end_year is not None and year > category.end_year:
            return False
        return True


def create_expense_engine_from_scenario(
    expenses: Expenses, time_grid: TimeGrid, inflation_adjuster: InflationAdjuster
) -> ExpenseEngine:
    """
    Create an expense engine from scenario expenses data.

    Args:
        expenses: Expenses data from scenario
        time_grid: Time grid for calculations
        inflation_adjuster: Inflation adjustment utilities

    Returns:
        Configured ExpenseEngine instance
    """
    engine = ExpenseEngine(time_grid=time_grid, inflation_adjuster=inflation_adjuster)

    # Add housing expenses
    housing_annual = (
        expenses.housing.mortgage_payment * 12
        + expenses.housing.property_tax
        + expenses.housing.home_insurance
        + expenses.housing.hoa_fees
        + expenses.housing.maintenance
        + expenses.housing.utilities * 12
    )
    if expenses.housing and housing_annual > 0:
        housing_category = HousingExpenseCategory(
            name="housing",
            mortgage_payment=expenses.housing.mortgage_payment,
            property_tax=expenses.housing.property_tax,
            home_insurance=expenses.housing.home_insurance,
            hoa_fees=expenses.housing.hoa_fees,
            maintenance=expenses.housing.maintenance,
            utilities=expenses.housing.utilities,
        )
        engine.add_expense_category(housing_category)

    # Add transportation expenses
    transportation_annual = (
        expenses.transportation.auto_payment * 12
        + expenses.transportation.auto_insurance
        + expenses.transportation.gas * 12
        + expenses.transportation.maintenance
    )
    if expenses.transportation and transportation_annual > 0:
        transportation_category = TransportationExpenseCategory(
            name="transportation",
            auto_payment=expenses.transportation.auto_payment,
            auto_insurance=expenses.transportation.auto_insurance,
            gas=expenses.transportation.gas,
            maintenance=expenses.transportation.maintenance,
        )
        engine.add_expense_category(transportation_category)

    # Add healthcare expenses
    healthcare_annual = (
        expenses.healthcare.insurance * 12
        + expenses.healthcare.out_of_pocket
        + expenses.healthcare.medicare * 12
    )
    if expenses.healthcare and healthcare_annual > 0:
        healthcare_category = HealthcareExpenseCategory(
            name="healthcare",
            insurance=expenses.healthcare.insurance,
            out_of_pocket=expenses.healthcare.out_of_pocket,
            medicare=expenses.healthcare.medicare,
        )
        engine.add_expense_category(healthcare_category)

    # Add simple expense categories
    if expenses.food > 0:
        engine.add_expense_category(
            ExpenseCategory(
                name="food",
                annual_amount=expenses.food * 12,  # Convert monthly to annual
            )
        )

    if expenses.entertainment > 0:
        engine.add_expense_category(
            ExpenseCategory(
                name="entertainment",
                annual_amount=expenses.entertainment * 12,  # Convert monthly to annual
            )
        )

    if expenses.travel > 0:
        engine.add_expense_category(
            ExpenseCategory(
                name="travel", annual_amount=expenses.travel  # Already annual
            )
        )

    if expenses.education > 0:
        engine.add_expense_category(
            ExpenseCategory(
                name="education", annual_amount=expenses.education  # Already annual
            )
        )

    if expenses.other > 0:
        engine.add_expense_category(
            ExpenseCategory(
                name="other",
                annual_amount=expenses.other * 12,  # Convert monthly to annual
            )
        )

    # Add lumpy events
    for lumpy_expense in expenses.lumpy_expenses:
        event = LumpyEvent(
            name=lumpy_expense.name,
            amount=lumpy_expense.amount,
            year=lumpy_expense.year,
            category=lumpy_expense.category,
        )
        engine.add_lumpy_event(event)

    return engine


def calculate_expense_inflation_impact(
    base_amount: float, years: int, inflation_rate: float
) -> float:
    """
    Calculate the impact of inflation on an expense over time.

    Args:
        base_amount: Base amount in current dollars
        years: Number of years to project
        inflation_rate: Annual inflation rate

    Returns:
        Future value after inflation
    """
    return base_amount * ((1 + inflation_rate) ** years)


def validate_expense_timing(
    start_year: Optional[int], end_year: Optional[int], time_grid: TimeGrid
) -> bool:
    """
    Validate that expense timing is within the time grid.

    Args:
        start_year: Start year for expense
        end_year: End year for expense
        time_grid: Time grid to validate against

    Returns:
        True if timing is valid

    Raises:
        ValueError: If timing is invalid
    """
    if start_year is not None:
        if start_year < time_grid.start_year or start_year > time_grid.end_year:
            raise ValueError(f"Start year {start_year} is outside time grid range")

    if end_year is not None:
        if end_year < time_grid.start_year or end_year > time_grid.end_year:
            raise ValueError(f"End year {end_year} is outside time grid range")

    return True
