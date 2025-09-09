"""
Tests for mortgage amortization calculations.

This module tests the mortgage amortization functionality including
payment calculations, amortization schedules, extra payments, and refinancing scenarios.
"""

import pytest
from decimal import Decimal

from app.models.mortgage_amortization import (
    AmortizationSchedule,
    MortgageCalculator,
    PaymentBreakdown,
    RefinancingScenario,
    create_sample_mortgage,
    create_sample_refinancing,
)
from app.models.scenario import Mortgage


class TestMortgageCalculator:
    """Test cases for MortgageCalculator class."""

    def test_calculate_monthly_payment_basic(self):
        """Test basic monthly payment calculation."""
        # Standard 30-year mortgage: $300,000 at 6% interest
        payment = MortgageCalculator.calculate_monthly_payment(
            principal=300000, annual_rate=0.06, term_years=30
        )

        # Expected payment should be approximately $1,798.65
        expected = 1798.65
        assert abs(payment - expected) < 0.01

    def test_calculate_monthly_payment_zero_interest(self):
        """Test monthly payment calculation with zero interest rate."""
        payment = MortgageCalculator.calculate_monthly_payment(
            principal=300000, annual_rate=0.0, term_years=30
        )

        # With zero interest, payment should be principal / (years * 12)
        expected = 300000 / (30 * 12)  # $833.33
        assert abs(payment - expected) < 0.01

    def test_calculate_monthly_payment_zero_principal(self):
        """Test monthly payment calculation with zero principal."""
        payment = MortgageCalculator.calculate_monthly_payment(
            principal=0, annual_rate=0.06, term_years=30
        )

        assert payment == 0.0

    def test_calculate_interest_payment(self):
        """Test interest payment calculation."""
        # $300,000 balance at 6% annual rate
        interest = MortgageCalculator.calculate_interest_payment(
            balance=300000, annual_rate=0.06
        )

        # Monthly interest should be $300,000 * 0.06 / 12 = $1,500
        expected = 1500.0
        assert abs(interest - expected) < 0.01

    def test_calculate_principal_payment(self):
        """Test principal payment calculation."""
        # Total payment $1,800, interest $1,500, extra $100
        principal = MortgageCalculator.calculate_principal_payment(
            total_payment=1800, interest_payment=1500, extra_payment=100
        )

        # Principal should be (1800 - 1500) + 100 = 400
        expected = 400.0
        assert abs(principal - expected) < 0.01

    def test_calculate_principal_payment_negative_interest(self):
        """Test principal payment when interest exceeds total payment."""
        principal = MortgageCalculator.calculate_principal_payment(
            total_payment=1000, interest_payment=1200, extra_payment=0
        )

        # Principal should be 0 (can't be negative)
        assert principal == 0.0


class TestAmortizationSchedule:
    """Test cases for amortization schedule generation."""

    def test_generate_amortization_schedule_basic(self):
        """Test basic amortization schedule generation."""
        mortgage = Mortgage(
            name="Test Mortgage",
            original_balance=100000,
            current_balance=100000,
            interest_rate=0.05,  # 5%
            term_years=30,
            start_year=2024,
            extra_payment=0,
        )

        schedule = MortgageCalculator.generate_amortization_schedule(mortgage)

        # Basic validations
        assert isinstance(schedule, AmortizationSchedule)
        assert len(schedule.payments) > 0
        assert schedule.total_payments > 0
        assert schedule.total_interest > 0
        assert schedule.total_principal > 0

        # First payment should have full balance
        first_payment = schedule.payments[0]
        assert first_payment.beginning_balance == 100000
        assert first_payment.interest_payment > 0
        assert first_payment.principal_payment > 0
        assert (
            first_payment.payment_amount
            == first_payment.interest_payment + first_payment.principal_payment
        )

        # Last payment should have near-zero balance (allow for small rounding errors)
        last_payment = schedule.payments[-1]
        assert last_payment.ending_balance < 2.0  # Allow for small rounding errors

    def test_generate_amortization_schedule_with_extra_payments(self):
        """Test amortization schedule with extra payments."""
        mortgage = Mortgage(
            name="Test Mortgage",
            original_balance=100000,
            current_balance=100000,
            interest_rate=0.05,
            term_years=30,
            start_year=2024,
            extra_payment=100,  # $100 extra per month
        )

        schedule = MortgageCalculator.generate_amortization_schedule(mortgage)

        # Should have fewer total payments due to extra payments
        assert len(schedule.payments) < 360  # Less than 30 years * 12 months

        # All payments should have the extra payment amount
        for payment in schedule.payments:
            assert payment.extra_payment == 100

    def test_generate_amortization_schedule_rounding(self):
        """Test that amortization schedule handles rounding correctly."""
        mortgage = Mortgage(
            name="Test Mortgage",
            original_balance=100000,
            current_balance=100000,
            interest_rate=0.055,  # 5.5% - creates non-round numbers
            term_years=30,
            start_year=2024,
            extra_payment=0,
        )

        schedule = MortgageCalculator.generate_amortization_schedule(mortgage)

        # All payment amounts should be properly rounded to 2 decimal places
        for payment in schedule.payments:
            assert (
                abs(payment.payment_amount - round(payment.payment_amount, 2)) < 0.001
            )
            assert (
                abs(payment.interest_payment - round(payment.interest_payment, 2))
                < 0.001
            )
            assert (
                abs(payment.principal_payment - round(payment.principal_payment, 2))
                < 0.001
            )

    def test_amortization_schedule_balance_consistency(self):
        """Test that balances are consistent throughout the schedule."""
        mortgage = create_sample_mortgage()
        schedule = MortgageCalculator.generate_amortization_schedule(mortgage)

        # Check balance consistency
        for i, payment in enumerate(schedule.payments):
            if i == 0:
                # First payment should start with current balance
                assert payment.beginning_balance == mortgage.current_balance
            else:
                # Each payment should start where the previous one ended
                assert (
                    payment.beginning_balance == schedule.payments[i - 1].ending_balance
                )

            # Ending balance should be beginning balance minus principal payment
            expected_ending = payment.beginning_balance - payment.principal_payment
            assert abs(payment.ending_balance - expected_ending) < 0.01


class TestRefinancingCalculations:
    """Test cases for refinancing calculations."""

    def test_calculate_refinancing_benefit_basic(self):
        """Test basic refinancing benefit calculation."""
        original_mortgage = Mortgage(
            name="Original",
            original_balance=300000,
            current_balance=250000,
            interest_rate=0.065,  # 6.5%
            term_years=30,
            start_year=2020,
            extra_payment=0,
        )

        refinancing = RefinancingScenario(
            new_rate=0.045,  # 4.5%
            new_term_years=30,
            refinance_year=2024,
            refinance_month=6,
            closing_costs=3000,
            cash_out_amount=0,
        )

        original_schedule, new_schedule, benefits = (
            MortgageCalculator.calculate_refinancing_benefit(
                original_mortgage, refinancing
            )
        )

        # Basic validations
        assert isinstance(original_schedule, AmortizationSchedule)
        assert isinstance(new_schedule, AmortizationSchedule)
        assert isinstance(benefits, dict)

        # Should have lower interest rate, so should save money
        assert benefits["interest_savings"] > 0
        assert benefits["monthly_payment_reduction"] > 0
        assert benefits["break_even_months"] > 0
        assert benefits["break_even_months"] < float("inf")

    def test_calculate_refinancing_benefit_with_cash_out(self):
        """Test refinancing benefit calculation with cash out."""
        original_mortgage = Mortgage(
            name="Original",
            original_balance=300000,
            current_balance=200000,
            interest_rate=0.06,
            term_years=30,
            start_year=2020,
            extra_payment=0,
        )

        refinancing = RefinancingScenario(
            new_rate=0.05,
            new_term_years=30,
            refinance_year=2024,
            refinance_month=1,
            closing_costs=5000,
            cash_out_amount=50000,  # Take out $50k
        )

        original_schedule, new_schedule, benefits = (
            MortgageCalculator.calculate_refinancing_benefit(
                original_mortgage, refinancing
            )
        )

        # New balance should include cash out and closing costs
        expected_new_balance = 200000 + 5000 + 50000  # $255,000
        assert abs(benefits["new_balance"] - expected_new_balance) < 0.01
        assert benefits["cash_out_amount"] == 50000

    def test_calculate_refinancing_benefit_no_savings(self):
        """Test refinancing when there are no savings."""
        original_mortgage = Mortgage(
            name="Original",
            original_balance=300000,
            current_balance=250000,
            interest_rate=0.05,  # 5%
            term_years=30,
            start_year=2020,
            extra_payment=0,
        )

        refinancing = RefinancingScenario(
            new_rate=0.06,  # Higher rate - no savings
            new_term_years=30,
            refinance_year=2024,
            refinance_month=1,
            closing_costs=3000,
            cash_out_amount=0,
        )

        original_schedule, new_schedule, benefits = (
            MortgageCalculator.calculate_refinancing_benefit(
                original_mortgage, refinancing
            )
        )

        # Should have negative interest savings (cost more)
        assert benefits["interest_savings"] < 0
        assert benefits["monthly_payment_reduction"] < 0


class TestPMIAndEquityCalculations:
    """Test cases for PMI and equity calculations."""

    def test_calculate_pmi_basic(self):
        """Test PMI calculation."""
        # LTV > 80%, should have PMI
        pmi = MortgageCalculator.calculate_pmi(
            loan_balance=250000, property_value=300000, pmi_rate=0.0055
        )

        # PMI should be $250,000 * 0.0055 / 12 = $114.58
        expected = 114.58
        assert abs(pmi - expected) < 0.01

    def test_calculate_pmi_no_pmi(self):
        """Test PMI calculation when LTV <= 80%."""
        # LTV = 80%, should have no PMI
        pmi = MortgageCalculator.calculate_pmi(
            loan_balance=240000, property_value=300000, pmi_rate=0.0055
        )

        assert pmi == 0.0

    def test_calculate_equity(self):
        """Test equity calculation."""
        equity = MortgageCalculator.calculate_equity(
            property_value=400000, loan_balance=300000
        )

        assert equity == 100000

    def test_calculate_equity_negative(self):
        """Test equity calculation when underwater."""
        equity = MortgageCalculator.calculate_equity(
            property_value=250000, loan_balance=300000
        )

        # Should return 0 (can't have negative equity in this calculation)
        assert equity == 0.0

    def test_calculate_loan_to_value_ratio(self):
        """Test LTV ratio calculation."""
        ltv = MortgageCalculator.calculate_loan_to_value_ratio(
            loan_balance=240000, property_value=300000
        )

        assert ltv == 0.8

    def test_calculate_loan_to_value_ratio_max(self):
        """Test LTV ratio calculation when loan exceeds property value."""
        ltv = MortgageCalculator.calculate_loan_to_value_ratio(
            loan_balance=350000, property_value=300000
        )

        # Should be capped at 1.0
        assert ltv == 1.0


class TestSampleData:
    """Test cases for sample data creation."""

    def test_create_sample_mortgage(self):
        """Test sample mortgage creation."""
        mortgage = create_sample_mortgage()

        assert isinstance(mortgage, Mortgage)
        assert mortgage.name == "Primary Residence"
        assert mortgage.original_balance == 500000.0
        assert mortgage.current_balance == 450000.0
        assert mortgage.interest_rate == 0.055
        assert mortgage.term_years == 30
        assert mortgage.start_year == 2024
        assert mortgage.property_value == 600000.0
        assert mortgage.extra_payment == 200.0

    def test_create_sample_refinancing(self):
        """Test sample refinancing creation."""
        refinancing = create_sample_refinancing()

        assert isinstance(refinancing, RefinancingScenario)
        assert refinancing.new_rate == 0.045
        assert refinancing.new_term_years == 30
        assert refinancing.refinance_year == 2025
        assert refinancing.refinance_month == 6
        assert refinancing.closing_costs == 5000.0
        assert refinancing.cash_out_amount == 0.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_loan(self):
        """Test amortization with very small loan amount."""
        mortgage = Mortgage(
            name="Small Loan",
            original_balance=1000,
            current_balance=1000,
            interest_rate=0.05,
            term_years=5,
            start_year=2024,
            extra_payment=0,
        )

        schedule = MortgageCalculator.generate_amortization_schedule(mortgage)

        # Should still generate valid schedule
        assert len(schedule.payments) > 0
        assert schedule.total_interest > 0
        assert schedule.total_principal > 0

    def test_very_high_interest_rate(self):
        """Test amortization with very high interest rate."""
        mortgage = Mortgage(
            name="High Rate",
            original_balance=100000,
            current_balance=100000,
            interest_rate=0.20,  # 20%
            term_years=30,
            start_year=2024,
            extra_payment=0,
        )

        schedule = MortgageCalculator.generate_amortization_schedule(mortgage)

        # Should generate valid schedule
        assert len(schedule.payments) > 0
        # Interest should be very high
        assert schedule.total_interest > schedule.total_principal

    def test_extra_payment_larger_than_balance(self):
        """Test when extra payment is larger than remaining balance."""
        mortgage = Mortgage(
            name="Large Extra Payment",
            original_balance=10000,
            current_balance=10000,
            interest_rate=0.05,
            term_years=30,
            start_year=2024,
            extra_payment=20000,  # Much larger than balance
        )

        schedule = MortgageCalculator.generate_amortization_schedule(mortgage)

        # Should handle gracefully and pay off quickly
        assert len(schedule.payments) < 12  # Should pay off in less than a year
        last_payment = schedule.payments[-1]
        assert last_payment.ending_balance < 0.01


class TestValidationAndConstraints:
    """Test validation and constraint handling."""

    def test_payment_breakdown_validation(self):
        """Test PaymentBreakdown model validation."""
        # Valid payment breakdown
        payment = PaymentBreakdown(
            payment_number=1,
            year=2024,
            month=1,
            beginning_balance=100000,
            payment_amount=1000,
            principal_payment=500,
            interest_payment=500,
            extra_payment=0,
            ending_balance=99500,
            cumulative_interest=500,
            cumulative_principal=500,
        )

        assert payment.payment_number == 1
        assert payment.year == 2024
        assert payment.month == 1

    def test_refinancing_scenario_validation(self):
        """Test RefinancingScenario model validation."""
        # Valid refinancing scenario
        refinancing = RefinancingScenario(
            new_rate=0.045,
            new_term_years=30,
            refinance_year=2024,
            refinance_month=6,
            closing_costs=3000,
            cash_out_amount=0,
        )

        assert refinancing.new_rate == 0.045
        assert refinancing.new_term_years == 30
        assert refinancing.refinance_year == 2024
        assert refinancing.refinance_month == 6

    def test_invalid_refinancing_month(self):
        """Test RefinancingScenario with invalid month."""
        with pytest.raises(Exception):  # Pydantic validation error
            RefinancingScenario(
                new_rate=0.045,
                new_term_years=30,
                refinance_year=2024,
                refinance_month=13,  # Invalid month
                closing_costs=3000,
                cash_out_amount=0,
            )


if __name__ == "__main__":
    pytest.main([__file__])
