"""
Social Security benefit calculation engine for retirement planning.

This module implements Social Security benefit calculations as a stub (simplified version)
for the retirement planning engine, providing fixed annual benefit amounts with
claim timing, benefit types, and COLA adjustments.
"""

from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from .time_grid import TimeGrid, InflationAdjuster
from .scenario import SocialSecurity, SocialSecurityStrategy, Household

if TYPE_CHECKING:
    from .scenario import Scenario


class SocialSecurityBenefit(BaseModel):
    """A Social Security benefit payment for a specific year."""

    year: int = Field(..., description="Year of the benefit payment")
    person: Literal["primary", "spouse"] = Field(..., description="Benefit recipient")
    benefit_type: Literal["retirement", "spousal", "survivor"] = Field(
        ..., description="Type of benefit"
    )
    annual_amount: float = Field(..., ge=0, description="Annual benefit amount")
    cola_adjustment: float = Field(
        default=0.0, description="COLA adjustment applied this year"
    )
    description: str = Field(..., description="Human-readable description")


class SocialSecurityEngine(BaseModel):
    """Engine for calculating Social Security benefits over time."""

    time_grid: TimeGrid = Field(..., description="Time grid for calculations")
    inflation_adjuster: InflationAdjuster = Field(
        ..., description="Inflation adjustment utilities"
    )
    household: Household = Field(..., description="Household composition")
    social_security_benefits: List[SocialSecurity] = Field(
        ..., description="Social Security benefit definitions"
    )
    strategy: SocialSecurityStrategy = Field(
        ..., description="Social Security claiming strategy"
    )
    cola_rate: float = Field(
        default=0.02, ge=0, le=0.1, description="Annual COLA rate"
    )

    def get_benefits_for_year(self, year: int) -> List[SocialSecurityBenefit]:
        """
        Get all Social Security benefits for a given year.

        Args:
            year: The year to calculate benefits for

        Returns:
            List of Social Security benefits for the year
        """
        benefits = []

        for benefit_def in self.social_security_benefits:
            if self._is_benefit_active(benefit_def, year):
                benefit = self._calculate_benefit(benefit_def, year)
                if benefit:
                    benefits.append(benefit)

        return benefits

    def get_total_annual_benefits(self, year: int) -> float:
        """
        Get total annual Social Security benefits for a given year.

        Args:
            year: The year to calculate benefits for

        Returns:
            Total annual Social Security benefits
        """
        benefits = self.get_benefits_for_year(year)
        return sum(benefit.annual_amount for benefit in benefits)

    def get_benefits_by_person(self, year: int) -> Dict[str, float]:
        """
        Get Social Security benefits by person for a given year.

        Args:
            year: The year to calculate benefits for

        Returns:
            Dictionary mapping person to total annual benefits
        """
        benefits = self.get_benefits_for_year(year)
        result = {"primary": 0.0, "spouse": 0.0}

        for benefit in benefits:
            result[benefit.person] += benefit.annual_amount

        return result

    def _is_benefit_active(self, benefit_def: SocialSecurity, year: int) -> bool:
        """Check if a benefit is active in the given year."""
        return year >= benefit_def.claim_year

    def _calculate_benefit(
        self, benefit_def: SocialSecurity, year: int
    ) -> Optional[SocialSecurityBenefit]:
        """
        Calculate a Social Security benefit for a specific year.

        Args:
            benefit_def: The benefit definition
            year: The year to calculate for

        Returns:
            Calculated benefit or None if not applicable
        """
        if not self._is_benefit_active(benefit_def, year):
            return None

        # Check if person is alive (simplified - assume both live to 100)
        person_age = self._get_person_age(benefit_def.person, year)
        if person_age > 100:
            return None

        # Calculate COLA-adjusted benefit amount
        years_since_claim = year - benefit_def.claim_year
        cola_adjustment = (1 + self.cola_rate) ** years_since_claim
        adjusted_amount = benefit_def.annual_benefit * cola_adjustment

        # Create benefit record
        benefit = SocialSecurityBenefit(
            year=year,
            person=benefit_def.person,
            benefit_type=benefit_def.benefit_type,
            annual_amount=adjusted_amount,
            cola_adjustment=cola_adjustment - 1.0,  # Show the adjustment factor
            description=f"{benefit_def.benefit_type.title()} benefit for {benefit_def.person}",
        )

        return benefit

    def _get_person_age(self, person: Literal["primary", "spouse"], year: int) -> int:
        """Get the age of a person in a given year."""
        if person == "primary":
            # Primary person's age in the base year (2024) is household.primary_age
            # So their birth year is 2024 - household.primary_age
            birth_year = 2024 - self.household.primary_age
        elif person == "spouse" and self.household.spouse_age is not None:
            # Spouse's age in the base year (2024) is household.spouse_age
            # So their birth year is 2024 - household.spouse_age
            birth_year = 2024 - self.household.spouse_age
        else:
            # Spouse doesn't exist
            return 0

        return year - birth_year

    def get_benefit_summary(self, start_year: int, end_year: int) -> Dict[str, Any]:
        """
        Get a summary of Social Security benefits over a range of years.

        Args:
            start_year: First year to include
            end_year: Last year to include

        Returns:
            Summary statistics and benefit details
        """
        total_benefits = 0.0
        benefit_years = 0
        max_annual_benefit = 0.0
        benefit_details = []

        for year in range(start_year, end_year + 1):
            annual_total = self.get_total_annual_benefits(year)
            if annual_total > 0:
                total_benefits += annual_total
                benefit_years += 1
                max_annual_benefit = max(max_annual_benefit, annual_total)

                benefits = self.get_benefits_for_year(year)
                benefit_details.extend(benefits)

        return {
            "total_benefits": total_benefits,
            "benefit_years": benefit_years,
            "average_annual_benefit": total_benefits / benefit_years if benefit_years > 0 else 0.0,
            "max_annual_benefit": max_annual_benefit,
            "benefit_details": benefit_details,
        }


def create_social_security_engine_from_scenario(
    scenario: "Scenario", time_grid: TimeGrid, inflation_adjuster: InflationAdjuster
) -> SocialSecurityEngine:
    """
    Create a Social Security engine from a scenario.

    Args:
        scenario: The retirement planning scenario
        time_grid: Time grid for calculations
        inflation_adjuster: Inflation adjustment utilities

    Returns:
        Configured Social Security engine
    """
    return SocialSecurityEngine(
        time_grid=time_grid,
        inflation_adjuster=inflation_adjuster,
        household=scenario.household,
        social_security_benefits=scenario.incomes.social_security,
        strategy=scenario.strategy.social_security_strategy,
        cola_rate=scenario.market_model.inflation,  # Use market inflation as COLA
    )


def validate_social_security_timing(
    social_security_benefits: List[SocialSecurity], household: Household
) -> List[str]:
    """
    Validate Social Security benefit timing and parameters.

    Args:
        social_security_benefits: List of Social Security benefit definitions
        household: Household composition

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    for benefit in social_security_benefits:
        # Check if person exists
        if benefit.person == "spouse" and household.spouse_age is None:
            errors.append("Spouse benefit defined but no spouse in household")

        # Check claim year is reasonable
        if benefit.person == "primary":
            # Primary person's birth year is 2024 - household.primary_age
            birth_year = 2024 - household.primary_age
            person_age_at_claim = benefit.claim_year - birth_year
        else:
            if household.spouse_age is None:
                continue
            # Spouse's birth year is 2024 - household.spouse_age
            birth_year = 2024 - household.spouse_age
            person_age_at_claim = benefit.claim_year - birth_year

        if person_age_at_claim < 62:
            errors.append(
                f"{benefit.person} claim age {person_age_at_claim} is below minimum 62"
            )
        elif person_age_at_claim > 70:
            errors.append(
                f"{benefit.person} claim age {person_age_at_claim} is above maximum 70"
            )

        # Check benefit amount is reasonable
        if benefit.annual_benefit <= 0:
            errors.append(f"{benefit.person} benefit amount must be positive")

    return errors


def calculate_social_security_present_value(
    benefits: List[SocialSecurityBenefit], discount_rate: float
) -> float:
    """
    Calculate the present value of Social Security benefits.

    Args:
        benefits: List of Social Security benefits
        discount_rate: Annual discount rate

    Returns:
        Present value of all benefits
    """
    if not benefits:
        return 0.0

    base_year = min(benefit.year for benefit in benefits)
    present_value = 0.0

    for benefit in benefits:
        years_from_base = benefit.year - base_year
        discount_factor = (1 + discount_rate) ** years_from_base
        present_value += benefit.annual_amount / discount_factor

    return present_value
