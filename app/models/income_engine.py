"""
Income processing engine for retirement planning.

This module handles income modeling including salary growth, bonuses, and other income
sources with inflation adjustment, timing constraints, and tax treatment metadata.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from .time_grid import InflationAdjuster, TimeGrid

if TYPE_CHECKING:
    from .scenario import Scenario


class IncomeCategory(BaseModel):
    """Base class for income categories."""

    name: str = Field(..., min_length=1, description="Income source name")
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
    growth_rate: float = Field(
        default=0.0, description="Annual growth rate (real, not nominal)"
    )
    tax_treatment: Literal[
        "w2_employment",
        "self_employment",
        "business_income",
        "investment_income",
        "rental_income",
        "retirement_income",
        "other_income",
    ] = Field(default="other_income", description="Tax treatment category")
    withholding_rate: float = Field(
        default=0.0, ge=0, le=1, description="Default withholding rate"
    )

    @field_validator("end_year")
    @classmethod
    def validate_end_year(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        if v is not None and "start_year" in info.data:
            start_year = info.data["start_year"]
            if start_year is not None and v < start_year:
                raise ValueError("End year must be >= start year")
        return v

    def get_inflation_adjusted_amount(
        self, year: int, inflation_adjuster: InflationAdjuster
    ) -> float:
        """Get inflation-adjusted amount for a specific year."""
        if not self.inflation_adjusted:
            return self.annual_amount

        return inflation_adjuster.adjust_for_inflation(
            self.annual_amount, inflation_adjuster.base_year, year
        )

    def get_growth_adjusted_amount(self, year: int, base_year: int) -> float:
        """Get growth-adjusted amount for a specific year."""
        if self.growth_rate == 0.0:
            return self.annual_amount

        years_from_base = year - base_year
        growth_factor = (1 + self.growth_rate) ** years_from_base
        return self.annual_amount * growth_factor


class EmploymentIncome(IncomeCategory):
    """Employment income (W-2 wages) with standard withholding."""

    person: Literal["primary", "spouse"] = Field(
        ..., description="Which household member"
    )
    employer: str = Field(default="", description="Employer name")
    job_title: str = Field(default="", description="Job title")
    bonus_amount: float = Field(default=0, ge=0, description="Annual bonus amount")
    bonus_frequency: Literal["annual", "one_time", "recurring"] = Field(
        default="annual", description="Bonus payment frequency"
    )
    bonus_start_year: Optional[int] = Field(
        default=None, ge=1900, le=2100, description="First year bonus is paid"
    )

    def __init__(self, **data: Any) -> None:
        # Set default tax treatment for employment income
        if "tax_treatment" not in data:
            data["tax_treatment"] = "w2_employment"
        if "withholding_rate" not in data:
            data["withholding_rate"] = 0.22  # Default federal withholding
        super().__init__(**data)

    def get_bonus_amount(self, year: int) -> float:
        """Get bonus amount for a specific year."""
        if self.bonus_amount == 0:
            return 0.0

        if self.bonus_frequency == "one_time":
            if self.bonus_start_year and year == self.bonus_start_year:
                return self.bonus_amount
            return 0.0
        elif self.bonus_frequency == "annual":
            if self.bonus_start_year and year >= self.bonus_start_year:
                return self.bonus_amount
            return 0.0
        elif self.bonus_frequency == "recurring":
            # Recurring bonuses every few years (simplified)
            if self.bonus_start_year and year >= self.bonus_start_year:
                return self.bonus_amount
            return 0.0


class SelfEmploymentIncome(IncomeCategory):
    """Self-employment income (1099) with SE tax implications."""

    person: Literal["primary", "spouse"] = Field(
        ..., description="Which household member"
    )
    business_name: str = Field(default="", description="Business name")
    se_tax_rate: float = Field(
        default=0.1413, ge=0, le=1, description="Self-employment tax rate"
    )
    quarterly_estimated_tax: bool = Field(
        default=True, description="Whether to pay quarterly estimated taxes"
    )

    def __init__(self, **data: Any) -> None:
        # Set default tax treatment for self-employment income
        if "tax_treatment" not in data:
            data["tax_treatment"] = "self_employment"
        if "withholding_rate" not in data:
            data["withholding_rate"] = 0.30  # Higher withholding for SE tax
        super().__init__(**data)


class BusinessIncome(IncomeCategory):
    """Business income (Schedule C) with deduction tracking."""

    person: Literal["primary", "spouse"] = Field(
        ..., description="Which household member"
    )
    business_name: str = Field(default="", description="Business name")
    business_type: str = Field(default="", description="Type of business")
    annual_deductions: float = Field(
        default=0, ge=0, description="Annual business deductions"
    )
    depreciation: float = Field(default=0, ge=0, description="Annual depreciation")

    def __init__(self, **data: Any) -> None:
        # Set default tax treatment for business income
        if "tax_treatment" not in data:
            data["tax_treatment"] = "business_income"
        if "withholding_rate" not in data:
            data["withholding_rate"] = 0.25  # Estimated tax rate
        super().__init__(**data)

    def get_net_income(self, year: int, inflation_adjuster: InflationAdjuster) -> float:
        """Get net business income after deductions."""
        gross_income = self.get_inflation_adjusted_amount(year, inflation_adjuster)
        # Deductions should be calculated separately, not using the same method as income
        if self.inflation_adjusted:
            deductions = inflation_adjuster.adjust_for_inflation(
                self.annual_deductions, inflation_adjuster.base_year, year
            )
        else:
            deductions = self.annual_deductions
        return max(0, gross_income - deductions)


class InvestmentIncome(IncomeCategory):
    """Investment income (dividends, interest, capital gains)."""

    income_type: Literal["dividends", "interest", "capital_gains", "distributions"] = (
        Field(..., description="Type of investment income")
    )
    account_type: Literal["taxable", "tax_deferred", "tax_free"] = Field(
        default="taxable", description="Account type for tax treatment"
    )
    qualified_dividends: bool = Field(
        default=False, description="Whether dividends are qualified"
    )

    def __init__(self, **data: Any) -> None:
        # Set default tax treatment for investment income
        if "tax_treatment" not in data:
            data["tax_treatment"] = "investment_income"
        if "withholding_rate" not in data:
            # Lower withholding for investment income
            data["withholding_rate"] = (
                0.15 if data.get("qualified_dividends", False) else 0.22
            )
        super().__init__(**data)


class RentalIncome(IncomeCategory):
    """Rental income with depreciation considerations."""

    property_address: str = Field(default="", description="Property address")
    property_type: Literal["residential", "commercial", "vacation"] = Field(
        default="residential", description="Type of rental property"
    )
    annual_expenses: float = Field(
        default=0, ge=0, description="Annual rental expenses"
    )
    annual_depreciation: float = Field(
        default=0, ge=0, description="Annual depreciation deduction"
    )
    management_fee_rate: float = Field(
        default=0.08, ge=0, le=1, description="Property management fee rate"
    )

    def __init__(self, **data: Any) -> None:
        # Set default tax treatment for rental income
        if "tax_treatment" not in data:
            data["tax_treatment"] = "rental_income"
        if "withholding_rate" not in data:
            data["withholding_rate"] = 0.0  # No withholding for rental income
        super().__init__(**data)

    def get_net_rental_income(
        self, year: int, inflation_adjuster: InflationAdjuster
    ) -> float:
        """Get net rental income after expenses and depreciation."""
        gross_rent = self.get_inflation_adjusted_amount(year, inflation_adjuster)
        management_fee = gross_rent * self.management_fee_rate
        # Expenses should be calculated separately, not using the same method as income
        if self.inflation_adjusted:
            expenses = inflation_adjuster.adjust_for_inflation(
                self.annual_expenses, inflation_adjuster.base_year, year
            )
        else:
            expenses = self.annual_expenses
        total_expenses = management_fee + expenses
        return max(0, gross_rent - total_expenses)


class RetirementIncome(IncomeCategory):
    """Retirement income (pension, annuity, RMDs)."""

    income_type: Literal["pension", "annuity", "rmd", "social_security"] = Field(
        ..., description="Type of retirement income"
    )
    person: Literal["primary", "spouse"] = Field(
        ..., description="Which household member"
    )
    cola_rate: float = Field(
        default=0.0, ge=0, le=0.1, description="Cost of living adjustment rate"
    )
    taxable_percentage: float = Field(
        default=1.0, ge=0, le=1, description="Percentage of income that is taxable"
    )

    def __init__(self, **data: Any) -> None:
        # Set default tax treatment for retirement income
        if "tax_treatment" not in data:
            data["tax_treatment"] = "retirement_income"
        if "withholding_rate" not in data:
            data["withholding_rate"] = 0.10  # Default withholding for retirement income
        super().__init__(**data)

    def get_cola_adjusted_amount(self, year: int, base_year: int) -> float:
        """Get COLA-adjusted amount for a specific year."""
        if self.cola_rate == 0.0:
            return self.annual_amount

        years_from_base = year - base_year
        cola_factor = (1 + self.cola_rate) ** years_from_base
        return self.annual_amount * cola_factor


class VariableIncome(IncomeCategory):
    """Variable income (commission, seasonal, freelance)."""

    person: Literal["primary", "spouse"] = Field(
        ..., description="Which household member"
    )
    variability_type: Literal["commission", "seasonal", "freelance", "irregular"] = (
        Field(..., description="Type of income variability")
    )
    base_amount: float = Field(default=0, ge=0, description="Base guaranteed amount")
    variable_amount: float = Field(
        default=0, ge=0, description="Variable component amount"
    )
    variability_factor: float = Field(
        default=1.0, ge=0, description="Multiplier for variable component"
    )

    def __init__(self, **data: Any) -> None:
        # Set default tax treatment for variable income
        if "tax_treatment" not in data:
            data["tax_treatment"] = "other_income"
        if "withholding_rate" not in data:
            data["withholding_rate"] = 0.25  # Higher withholding for variable income
        super().__init__(**data)

    def get_variable_amount(self, year: int) -> float:
        """Get variable income amount for a specific year."""
        # Simplified variability - in real implementation, this could be more complex
        base = self.base_amount
        variable = self.variable_amount * self.variability_factor
        return base + variable


class IncomeChangeEvent(BaseModel):
    """An event that changes income (raise, job change, gap, etc.)."""

    year: int = Field(..., ge=1900, le=2100, description="Year of the change")
    income_source_name: str = Field(..., description="Name of the income source")
    change_type: Literal[
        "raise",
        "job_change",
        "income_gap",
        "reduction",
        "bonus_change",
        "retirement",
    ] = Field(..., description="Type of income change")
    new_amount: Optional[float] = Field(
        None, ge=0, description="New annual amount (if applicable)"
    )
    new_growth_rate: Optional[float] = Field(
        None, description="New growth rate (if applicable)"
    )
    description: str = Field(..., description="Description of the change")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional change metadata"
    )


class IncomeEngine(BaseModel):
    """Engine for processing all income sources over time."""

    time_grid: TimeGrid = Field(..., description="Time grid for calculations")
    inflation_adjuster: InflationAdjuster = Field(
        ..., description="Inflation adjustment utilities"
    )
    income_categories: List[IncomeCategory] = Field(
        default_factory=list, description="All income categories"
    )
    income_change_events: List[IncomeChangeEvent] = Field(
        default_factory=list, description="Income change events over time"
    )

    def add_income_category(self, category: IncomeCategory) -> None:
        """Add an income category."""
        self.income_categories.append(category)

    def add_income_change_event(self, event: IncomeChangeEvent) -> None:
        """Add an income change event."""
        self.income_change_events.append(event)

    def get_annual_income(self, year: int) -> Dict[str, float]:
        """
        Get all income for a given year.

        Args:
            year: The year to calculate income for

        Returns:
            Dictionary mapping income source names to amounts
        """
        income = {}

        # Add regular income categories
        for category in self.income_categories:
            if self._is_category_active(category, year):
                amount = self._calculate_category_income(category, year)
                if amount > 0:
                    income[category.name] = amount

        # Apply income change events for this year
        for event in self.income_change_events:
            if event.year == year and event.income_source_name in income:
                income[event.income_source_name] = self._apply_income_change(
                    income[event.income_source_name], event
                )

        return income

    def get_total_annual_income(self, year: int) -> float:
        """Get total annual income for a given year."""
        annual_income = self.get_annual_income(year)
        return sum(annual_income.values())

    def get_income_by_tax_treatment(self, year: int) -> Dict[str, float]:
        """Get income grouped by tax treatment for a given year."""
        annual_income = self.get_annual_income(year)
        tax_groups: Dict[str, float] = {}

        for category in self.income_categories:
            if category.name in annual_income:
                tax_treatment = category.tax_treatment
                if tax_treatment not in tax_groups:
                    tax_groups[tax_treatment] = 0.0
                tax_groups[tax_treatment] += annual_income[category.name]

        return tax_groups

    def get_income_series(
        self, start_year: Optional[int] = None, end_year: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Get income series for all years in the time grid.

        Args:
            start_year: First year to include (default: time_grid.start_year)
            end_year: Last year to include (default: time_grid.end_year)

        Returns:
            Dictionary mapping income source names to lists of annual amounts
        """
        if start_year is None:
            start_year = self.time_grid.start_year
        if end_year is None:
            end_year = self.time_grid.end_year

        income_series = {}
        years = list(range(start_year, end_year + 1))

        for category in self.income_categories:
            series = []
            for year in years:
                if self._is_category_active(category, year):
                    amount = self._calculate_category_income(category, year)
                    series.append(amount)
                else:
                    series.append(0.0)
            income_series[category.name] = series

        return income_series

    def get_total_income_series(
        self, start_year: Optional[int] = None, end_year: Optional[int] = None
    ) -> List[float]:
        """Get total income series for all years."""
        income_series = self.get_income_series(start_year, end_year)
        years = list(
            range(
                start_year or self.time_grid.start_year,
                (end_year or self.time_grid.end_year) + 1,
            )
        )

        total_series = []
        for i, year in enumerate(years):
            total = sum(series[i] for series in income_series.values())
            total_series.append(total)

        return total_series

    def _is_category_active(self, category: IncomeCategory, year: int) -> bool:
        """Check if a category is active in the given year."""
        if category.start_year is not None and year < category.start_year:
            return False
        if category.end_year is not None and year > category.end_year:
            return False
        return True

    def _calculate_category_income(self, category: IncomeCategory, year: int) -> float:
        """Calculate income for a specific category and year."""
        # Handle special cases for different income types first
        if isinstance(category, BusinessIncome):
            return category.get_net_income(year, self.inflation_adjuster)
        elif isinstance(category, RentalIncome):
            return category.get_net_rental_income(year, self.inflation_adjuster)
        elif isinstance(category, RetirementIncome):
            return category.get_cola_adjusted_amount(
                year, self.inflation_adjuster.base_year
            )
        elif isinstance(category, VariableIncome):
            return category.get_variable_amount(year)

        # For regular income categories, apply both inflation and growth
        base_amount = category.annual_amount

        # Apply inflation if enabled
        if category.inflation_adjusted:
            base_amount = self.inflation_adjuster.adjust_for_inflation(
                base_amount, self.inflation_adjuster.base_year, year
            )

        # Apply growth
        if category.growth_rate != 0.0:
            years_from_base = year - self.inflation_adjuster.base_year
            growth_factor = (1 + category.growth_rate) ** years_from_base
            base_amount = base_amount * growth_factor

        # Add bonus for employment income
        if isinstance(category, EmploymentIncome):
            bonus = category.get_bonus_amount(year)
            base_amount += bonus

        return base_amount

    def _apply_income_change(
        self, current_amount: float, event: IncomeChangeEvent
    ) -> float:
        """Apply an income change event to the current amount."""
        if event.change_type == "raise" and event.new_amount is not None:
            return event.new_amount
        elif event.change_type == "reduction" and event.new_amount is not None:
            return event.new_amount
        elif event.change_type == "income_gap":
            return 0.0
        elif event.change_type == "job_change" and event.new_amount is not None:
            return event.new_amount
        elif event.change_type == "retirement":
            return 0.0

        return current_amount


def create_income_engine_from_scenario(
    scenario: "Scenario", time_grid: TimeGrid, inflation_adjuster: InflationAdjuster
) -> IncomeEngine:
    """
    Create an income engine from a scenario.

    Args:
        scenario: The retirement planning scenario
        time_grid: Time grid for calculations
        inflation_adjuster: Inflation adjustment utilities

    Returns:
        Configured income engine
    """
    engine = IncomeEngine(
        time_grid=time_grid,
        inflation_adjuster=inflation_adjuster,
    )

    # Convert salary income to EmploymentIncome
    for salary in scenario.incomes.salary:
        employment_income = EmploymentIncome(
            name=f"{salary.person}_salary",
            annual_amount=salary.annual_amount,
            start_year=salary.start_year,
            end_year=salary.end_year,
            growth_rate=salary.growth_rate,
            person=salary.person,
            bonus_amount=salary.bonus,
            bonus_frequency="annual",
        )
        engine.add_income_category(employment_income)

    # Convert other income to appropriate categories
    for other_income in scenario.incomes.other:
        income_category = IncomeCategory(
            name=other_income.name,
            annual_amount=other_income.annual_amount,
            start_year=other_income.start_year,
            end_year=other_income.end_year,
            growth_rate=other_income.growth_rate,
            tax_treatment="other_income",
        )
        engine.add_income_category(income_category)

    return engine


def validate_income_timing(
    income_categories: List[IncomeCategory], time_grid: TimeGrid
) -> List[str]:
    """
    Validate income timing and parameters.

    Args:
        income_categories: List of income categories
        time_grid: Time grid for validation

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    for category in income_categories:
        # Check timing constraints
        if (
            category.start_year is not None
            and category.start_year < time_grid.start_year
        ):
            errors.append(f"{category.name} starts before time grid begins")

        if category.end_year is not None and category.end_year > time_grid.end_year:
            errors.append(f"{category.name} ends after time grid ends")

        # Check growth rate is reasonable
        if abs(category.growth_rate) > 0.5:  # 50% growth rate seems excessive
            errors.append(
                f"{category.name} has excessive growth rate: {category.growth_rate}"
            )

        # Check withholding rate is reasonable
        if category.withholding_rate > 0.5:  # 50% withholding seems excessive
            errors.append(
                f"{category.name} has excessive withholding rate: {category.withholding_rate}"
            )

    return errors


def calculate_income_present_value(
    income_series: List[float], discount_rate: float, start_year: int
) -> float:
    """
    Calculate the present value of an income stream.

    Args:
        income_series: List of annual income amounts
        discount_rate: Annual discount rate
        start_year: First year of the income stream

    Returns:
        Present value of the income stream
    """
    if not income_series:
        return 0.0

    present_value = 0.0
    for i, amount in enumerate(income_series):
        year = start_year + i
        years_from_start = year - start_year
        discount_factor = (1 + discount_rate) ** years_from_start
        present_value += amount / discount_factor

    return present_value
