"""
Mortgage amortization calculations for retirement planning.

This module provides comprehensive mortgage amortization calculations including
payment schedules, interest/principal splits, extra payments, and refinancing scenarios.
"""

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from .scenario import Mortgage


class PaymentBreakdown(BaseModel):
    """Breakdown of a single mortgage payment."""

    payment_number: int = Field(..., ge=1, description="Payment number (1-based)")
    year: int = Field(..., ge=1900, le=2100, description="Year of payment")
    month: int = Field(..., ge=1, le=12, description="Month of payment")
    beginning_balance: float = Field(
        ..., ge=0, description="Balance at beginning of period"
    )
    payment_amount: float = Field(..., ge=0, description="Total payment amount")
    principal_payment: float = Field(
        ..., ge=0, description="Principal portion of payment"
    )
    interest_payment: float = Field(
        ..., ge=0, description="Interest portion of payment"
    )
    extra_payment: float = Field(default=0, ge=0, description="Extra principal payment")
    ending_balance: float = Field(..., ge=0, description="Balance at end of period")
    cumulative_interest: float = Field(
        ..., ge=0, description="Cumulative interest paid"
    )
    cumulative_principal: float = Field(
        ..., ge=0, description="Cumulative principal paid"
    )


class AmortizationSchedule(BaseModel):
    """Complete amortization schedule for a mortgage."""

    mortgage: Mortgage = Field(..., description="Mortgage parameters")
    payments: List[PaymentBreakdown] = Field(
        ..., description="List of payment breakdowns"
    )
    total_payments: int = Field(..., ge=1, description="Total number of payments")
    total_interest: float = Field(
        ..., ge=0, description="Total interest paid over life of loan"
    )
    total_principal: float = Field(
        ..., ge=0, description="Total principal paid over life of loan"
    )
    payoff_year: int = Field(..., ge=1900, le=2100, description="Year loan is paid off")
    payoff_month: int = Field(..., ge=1, le=12, description="Month loan is paid off")


class RefinancingScenario(BaseModel):
    """Parameters for a mortgage refinancing scenario."""

    new_rate: float = Field(..., ge=0, le=1, description="New interest rate (0-1)")
    new_term_years: int = Field(..., ge=1, le=50, description="New term in years")
    refinance_year: int = Field(..., ge=1900, le=2100, description="Year to refinance")
    refinance_month: int = Field(..., ge=1, le=12, description="Month to refinance")
    closing_costs: float = Field(
        default=0, ge=0, description="Closing costs for refinance"
    )
    cash_out_amount: float = Field(
        default=0, ge=0, description="Cash taken out in refinance"
    )

    @field_validator("refinance_month")
    @classmethod
    def validate_month(cls, v: int) -> int:
        if not 1 <= v <= 12:
            raise ValueError("Month must be between 1 and 12")
        return v


class MortgageCalculator:
    """Calculator for mortgage amortization and related calculations."""

    @staticmethod
    def calculate_monthly_payment(
        principal: float, annual_rate: float, term_years: int
    ) -> float:
        """
        Calculate the monthly mortgage payment using the standard formula.

        Args:
            principal: Loan principal amount
            annual_rate: Annual interest rate (as decimal, e.g., 0.055 for 5.5%)
            term_years: Loan term in years

        Returns:
            Monthly payment amount
        """
        if principal <= 0:
            return 0.0
        if annual_rate <= 0:
            return principal / (term_years * 12)

        monthly_rate = annual_rate / 12
        num_payments = term_years * 12

        # Standard mortgage payment formula
        if monthly_rate == 0:
            payment = principal / num_payments
        else:
            payment = (
                principal
                * (monthly_rate * (1 + monthly_rate) ** num_payments)
                / ((1 + monthly_rate) ** num_payments - 1)
            )

        # Round to nearest cent
        return round(payment, 2)

    @staticmethod
    def calculate_interest_payment(balance: float, annual_rate: float) -> float:
        """
        Calculate the interest portion of a monthly payment.

        Args:
            balance: Current loan balance
            annual_rate: Annual interest rate (as decimal)

        Returns:
            Interest payment amount
        """
        monthly_rate = annual_rate / 12
        return round(balance * monthly_rate, 2)

    @staticmethod
    def calculate_principal_payment(
        total_payment: float, interest_payment: float, extra_payment: float = 0
    ) -> float:
        """
        Calculate the principal portion of a payment.

        Args:
            total_payment: Total monthly payment
            interest_payment: Interest portion of payment
            extra_payment: Extra principal payment

        Returns:
            Total principal payment (regular + extra)
        """
        regular_principal = max(0, total_payment - interest_payment)
        return round(regular_principal + extra_payment, 2)

    @staticmethod
    def generate_amortization_schedule(
        mortgage: Mortgage,
        start_year: Optional[int] = None,
        start_month: Optional[int] = None,
    ) -> AmortizationSchedule:
        """
        Generate a complete amortization schedule for a mortgage.

        Args:
            mortgage: Mortgage parameters
            start_year: Start year for calculations (defaults to mortgage start_year)
            start_month: Start month for calculations (defaults to 1)

        Returns:
            Complete amortization schedule
        """
        if start_year is None:
            start_year = mortgage.start_year
        if start_month is None:
            start_month = 1

        # Calculate monthly payment
        monthly_payment = MortgageCalculator.calculate_monthly_payment(
            mortgage.current_balance, mortgage.interest_rate, mortgage.term_years
        )

        payments = []
        balance = mortgage.current_balance
        cumulative_interest = 0.0
        cumulative_principal = 0.0
        payment_number = 1

        current_year = start_year
        current_month = start_month

        # Generate payments until loan is paid off
        while balance > 0.01 and payment_number <= mortgage.term_years * 12:
            # Calculate interest payment
            interest_payment = MortgageCalculator.calculate_interest_payment(
                balance, mortgage.interest_rate
            )

            # Calculate principal payment
            principal_payment = MortgageCalculator.calculate_principal_payment(
                monthly_payment, interest_payment, mortgage.extra_payment
            )

            # Ensure we don't overpay
            if principal_payment > balance:
                principal_payment = balance
                monthly_payment = interest_payment + principal_payment

            # Calculate ending balance
            ending_balance = max(0, balance - principal_payment)

            # Update cumulative totals
            cumulative_interest += interest_payment
            cumulative_principal += principal_payment

            # Create payment breakdown
            payment = PaymentBreakdown(
                payment_number=payment_number,
                year=current_year,
                month=current_month,
                beginning_balance=balance,
                payment_amount=monthly_payment,
                principal_payment=principal_payment,
                interest_payment=interest_payment,
                extra_payment=mortgage.extra_payment,
                ending_balance=ending_balance,
                cumulative_interest=cumulative_interest,
                cumulative_principal=cumulative_principal,
            )

            payments.append(payment)

            # Update for next iteration
            balance = ending_balance
            payment_number += 1

            # Advance month/year
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

        # Determine payoff date
        if payments:
            last_payment = payments[-1]
            payoff_year = last_payment.year
            payoff_month = last_payment.month
        else:
            payoff_year = start_year
            payoff_month = start_month

        return AmortizationSchedule(
            mortgage=mortgage,
            payments=payments,
            total_payments=len(payments),
            total_interest=cumulative_interest,
            total_principal=cumulative_principal,
            payoff_year=payoff_year,
            payoff_month=payoff_month,
        )

    @staticmethod
    def calculate_refinancing_benefit(
        original_mortgage: Mortgage,
        refinancing: RefinancingScenario,
        current_balance: Optional[float] = None,
    ) -> Tuple[AmortizationSchedule, AmortizationSchedule, dict]:
        """
        Calculate the benefit of refinancing a mortgage.

        Args:
            original_mortgage: Current mortgage parameters
            refinancing: Refinancing scenario parameters
            current_balance: Current balance (defaults to original_mortgage.current_balance)

        Returns:
            Tuple of (original_schedule, new_schedule, benefit_analysis)
        """
        if current_balance is None:
            current_balance = original_mortgage.current_balance

        # Generate original schedule from current point
        original_schedule = MortgageCalculator.generate_amortization_schedule(
            original_mortgage
        )

        # Create new mortgage with refinancing parameters
        new_balance = (
            current_balance + refinancing.closing_costs + refinancing.cash_out_amount
        )
        new_mortgage = Mortgage(
            name=f"{original_mortgage.name} (Refinanced)",
            original_balance=new_balance,
            current_balance=new_balance,
            interest_rate=refinancing.new_rate,
            term_years=refinancing.new_term_years,
            start_year=refinancing.refinance_year,
            property_value=original_mortgage.property_value,
            extra_payment=original_mortgage.extra_payment,
        )

        # Generate new schedule
        new_schedule = MortgageCalculator.generate_amortization_schedule(
            new_mortgage,
            start_year=refinancing.refinance_year,
            start_month=refinancing.refinance_month,
        )

        # Calculate benefits
        interest_savings = (
            original_schedule.total_interest - new_schedule.total_interest
        )
        payment_reduction = MortgageCalculator.calculate_monthly_payment(
            original_mortgage.current_balance,
            original_mortgage.interest_rate,
            original_mortgage.term_years,
        ) - MortgageCalculator.calculate_monthly_payment(
            new_balance, refinancing.new_rate, refinancing.new_term_years
        )

        # Calculate break-even point (months to recover closing costs)
        if payment_reduction > 0:
            break_even_months = refinancing.closing_costs / payment_reduction
        else:
            break_even_months = float("inf")

        benefit_analysis = {
            "interest_savings": interest_savings,
            "monthly_payment_reduction": payment_reduction,
            "break_even_months": break_even_months,
            "net_benefit": interest_savings - refinancing.closing_costs,
            "new_balance": new_balance,
            "closing_costs": refinancing.closing_costs,
            "cash_out_amount": refinancing.cash_out_amount,
        }

        return original_schedule, new_schedule, benefit_analysis

    @staticmethod
    def calculate_pmi(
        loan_balance: float, property_value: float, pmi_rate: float = 0.0055
    ) -> float:
        """
        Calculate Private Mortgage Insurance (PMI) payment.

        Args:
            loan_balance: Current loan balance
            property_value: Current property value
            pmi_rate: Annual PMI rate (default 0.55%)

        Returns:
            Monthly PMI payment
        """
        loan_to_value = loan_balance / property_value if property_value > 0 else 1.0

        # PMI typically required when LTV > 80%
        if loan_to_value <= 0.80:
            return 0.0

        # Calculate PMI based on loan balance
        annual_pmi = loan_balance * pmi_rate
        return round(annual_pmi / 12, 2)

    @staticmethod
    def calculate_equity(property_value: float, loan_balance: float) -> float:
        """
        Calculate home equity.

        Args:
            property_value: Current property value
            loan_balance: Current loan balance

        Returns:
            Home equity amount
        """
        return max(0, property_value - loan_balance)

    @staticmethod
    def calculate_loan_to_value_ratio(
        loan_balance: float, property_value: float
    ) -> float:
        """
        Calculate loan-to-value ratio.

        Args:
            loan_balance: Current loan balance
            property_value: Current property value

        Returns:
            LTV ratio (0-1)
        """
        if property_value <= 0:
            return 1.0
        return min(1.0, loan_balance / property_value)


def create_sample_mortgage() -> Mortgage:
    """Create a sample mortgage for testing purposes."""
    return Mortgage(
        name="Primary Residence",
        original_balance=500000.0,
        current_balance=450000.0,
        interest_rate=0.055,  # 5.5%
        term_years=30,
        start_year=2024,
        property_value=600000.0,
        extra_payment=200.0,
    )


def create_sample_refinancing() -> RefinancingScenario:
    """Create a sample refinancing scenario for testing purposes."""
    return RefinancingScenario(
        new_rate=0.045,  # 4.5%
        new_term_years=30,
        refinance_year=2025,
        refinance_month=6,
        closing_costs=5000.0,
        cash_out_amount=0.0,
    )
