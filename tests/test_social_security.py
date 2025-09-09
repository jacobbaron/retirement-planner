"""
Tests for Social Security benefit calculation engine.

This module tests the Social Security benefit calculation functionality including
fixed annual benefit amounts, claim timing, benefit types, COLA adjustments,
and integration with income projections.
"""

import pytest
from decimal import Decimal

from app.models.social_security import (
    SocialSecurityBenefit,
    SocialSecurityEngine,
    create_social_security_engine_from_scenario,
    validate_social_security_timing,
    calculate_social_security_present_value,
)
from app.models.time_grid import TimeGrid, InflationAdjuster
from app.models.scenario import (
    SocialSecurity,
    SocialSecurityStrategy,
    Household,
    Incomes,
    Strategy,
    MarketModel,
    ExpectedReturns,
    Volatility,
    Correlations,
    Scenario,
)


class TestSocialSecurityBenefit:
    """Test the SocialSecurityBenefit model."""

    def test_basic_benefit_creation(self):
        """Test basic Social Security benefit creation."""
        benefit = SocialSecurityBenefit(
            year=2024,
            person="primary",
            benefit_type="retirement",
            annual_amount=30000.0,
            cola_adjustment=0.02,
            description="Primary retirement benefit",
        )

        assert benefit.year == 2024
        assert benefit.person == "primary"
        assert benefit.benefit_type == "retirement"
        assert benefit.annual_amount == 30000.0
        assert benefit.cola_adjustment == 0.02
        assert benefit.description == "Primary retirement benefit"

    def test_benefit_with_defaults(self):
        """Test Social Security benefit with default values."""
        benefit = SocialSecurityBenefit(
            year=2025,
            person="spouse",
            benefit_type="spousal",
            annual_amount=15000.0,
            description="Spousal benefit",
        )

        assert benefit.year == 2025
        assert benefit.person == "spouse"
        assert benefit.benefit_type == "spousal"
        assert benefit.annual_amount == 15000.0
        assert benefit.cola_adjustment == 0.0  # Default value
        assert benefit.description == "Spousal benefit"

    def test_benefit_validation(self):
        """Test Social Security benefit validation."""
        # Test negative amount
        with pytest.raises(ValueError):
            SocialSecurityBenefit(
                year=2024,
                person="primary",
                benefit_type="retirement",
                annual_amount=-1000.0,
                description="Invalid negative amount",
            )

        # Test invalid person
        with pytest.raises(ValueError):
            SocialSecurityBenefit(
                year=2024,
                person="invalid",  # type: ignore
                benefit_type="retirement",
                annual_amount=30000.0,
                description="Invalid person",
            )

        # Test invalid benefit type
        with pytest.raises(ValueError):
            SocialSecurityBenefit(
                year=2024,
                person="primary",
                benefit_type="invalid",  # type: ignore
                annual_amount=30000.0,
                description="Invalid benefit type",
            )


class TestSocialSecurityEngine:
    """Test the SocialSecurityEngine class."""

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
    def household(self):
        """Create a household for testing."""
        return Household(
            primary_age=35,
            spouse_age=33,
            filing_status="married_filing_jointly",
            state="CA",
        )

    @pytest.fixture
    def social_security_benefits(self):
        """Create Social Security benefits for testing."""
        return [
            SocialSecurity(
                person="primary",
                annual_benefit=30000.0,
                claim_year=2057,  # Age 68
                benefit_type="retirement",
            ),
            SocialSecurity(
                person="spouse",
                annual_benefit=15000.0,
                claim_year=2057,  # Age 65 (same year as primary for testing)
                benefit_type="retirement",
            ),
        ]

    @pytest.fixture
    def social_security_strategy(self):
        """Create a Social Security strategy for testing."""
        return SocialSecurityStrategy(
            primary_claim_age=68,
            spouse_claim_age=65,
            file_and_suspend=False,
        )

    @pytest.fixture
    def social_security_engine(
        self, time_grid, inflation_adjuster, household, social_security_benefits, social_security_strategy
    ):
        """Create a Social Security engine for testing."""
        return SocialSecurityEngine(
            time_grid=time_grid,
            inflation_adjuster=inflation_adjuster,
            household=household,
            social_security_benefits=social_security_benefits,
            strategy=social_security_strategy,
            cola_rate=0.02,
        )

    def test_engine_creation(self, social_security_engine):
        """Test Social Security engine creation."""
        assert social_security_engine.cola_rate == 0.02
        assert len(social_security_engine.social_security_benefits) == 2
        assert social_security_engine.strategy.primary_claim_age == 68

    def test_get_benefits_for_year_before_claim(self, social_security_engine):
        """Test getting benefits before claim year."""
        benefits = social_security_engine.get_benefits_for_year(2024)
        assert len(benefits) == 0

    def test_get_benefits_for_year_after_claim(self, social_security_engine):
        """Test getting benefits after claim year."""
        benefits = social_security_engine.get_benefits_for_year(2057)
        assert len(benefits) == 2  # Both benefits start this year (primary at 68, spouse at 65)

        # Find primary and spouse benefits
        primary_benefit = next(b for b in benefits if b.person == "primary")
        spouse_benefit = next(b for b in benefits if b.person == "spouse")
        assert primary_benefit.annual_amount == 30000.0
        assert spouse_benefit.annual_amount == 15000.0

        benefits = social_security_engine.get_benefits_for_year(2058)
        assert len(benefits) == 2  # Both benefits active
        assert benefits[0].person == "primary"
        assert benefits[1].person == "spouse"

    def test_get_total_annual_benefits(self, social_security_engine):
        """Test getting total annual benefits."""
        # Before any claims
        total = social_security_engine.get_total_annual_benefits(2024)
        assert total == 0.0

        # After both claims (both start in 2057)
        total = social_security_engine.get_total_annual_benefits(2057)
        assert total == 45000.0  # 30000 + 15000

        # After both claims with COLA
        total = social_security_engine.get_total_annual_benefits(2058)
        assert total > 45000.0  # Should be higher due to COLA

    def test_get_benefits_by_person(self, social_security_engine):
        """Test getting benefits by person."""
        # Before any claims
        benefits = social_security_engine.get_benefits_by_person(2024)
        assert benefits["primary"] == 0.0
        assert benefits["spouse"] == 0.0

        # After both claims (both start in 2057)
        benefits = social_security_engine.get_benefits_by_person(2057)
        assert benefits["primary"] == 30000.0
        assert benefits["spouse"] == 15000.0

        # After both claims with COLA
        benefits = social_security_engine.get_benefits_by_person(2058)
        assert benefits["primary"] > 30000.0  # Higher due to COLA
        assert benefits["spouse"] > 15000.0   # Higher due to COLA

    def test_cola_adjustment(self, social_security_engine):
        """Test COLA adjustment over time."""
        # First year of benefit (no COLA)
        benefits = social_security_engine.get_benefits_for_year(2057)
        assert benefits[0].annual_amount == 30000.0
        assert benefits[0].cola_adjustment == 0.0

        # Second year (1 year of COLA)
        benefits = social_security_engine.get_benefits_for_year(2058)
        primary_benefit = next(b for b in benefits if b.person == "primary")
        expected_amount = 30000.0 * 1.02  # 2% COLA
        assert abs(primary_benefit.annual_amount - expected_amount) < 0.01
        assert abs(primary_benefit.cola_adjustment - 0.02) < 0.01

        # Third year (2 years of COLA)
        benefits = social_security_engine.get_benefits_for_year(2059)
        primary_benefit = next(b for b in benefits if b.person == "primary")
        expected_amount = 30000.0 * (1.02 ** 2)  # 2 years of 2% COLA
        assert abs(primary_benefit.annual_amount - expected_amount) < 0.01

    def test_benefit_summary(self, social_security_engine):
        """Test benefit summary calculation."""
        summary = social_security_engine.get_benefit_summary(2057, 2060)

        assert summary["benefit_years"] == 4
        assert summary["max_annual_benefit"] > 0
        assert summary["average_annual_benefit"] > 0
        assert len(summary["benefit_details"]) > 0

    def test_single_household(self, time_grid, inflation_adjuster):
        """Test engine with single household member."""
        household = Household(
            primary_age=35,
            filing_status="single",
            state="CA",
        )

        benefits = [
            SocialSecurity(
                person="primary",
                annual_benefit=25000.0,
                claim_year=2057,
                benefit_type="retirement",
            )
        ]

        strategy = SocialSecurityStrategy(primary_claim_age=68)

        engine = SocialSecurityEngine(
            time_grid=time_grid,
            inflation_adjuster=inflation_adjuster,
            household=household,
            social_security_benefits=benefits,
            strategy=strategy,
        )

        # Test benefits for single person
        benefits_2057 = engine.get_benefits_for_year(2057)
        assert len(benefits_2057) == 1
        assert benefits_2057[0].person == "primary"
        assert benefits_2057[0].annual_amount == 25000.0

        # Test benefits by person
        benefits_by_person = engine.get_benefits_by_person(2057)
        assert benefits_by_person["primary"] == 25000.0
        assert benefits_by_person["spouse"] == 0.0

    def test_different_benefit_types(self, time_grid, inflation_adjuster, household):
        """Test different benefit types."""
        benefits = [
            SocialSecurity(
                person="primary",
                annual_benefit=30000.0,
                claim_year=2057,
                benefit_type="retirement",
            ),
            SocialSecurity(
                person="spouse",
                annual_benefit=15000.0,
                claim_year=2055,
                benefit_type="spousal",
            ),
            SocialSecurity(
                person="spouse",
                annual_benefit=20000.0,
                claim_year=2060,
                benefit_type="survivor",
            ),
        ]

        strategy = SocialSecurityStrategy()

        engine = SocialSecurityEngine(
            time_grid=time_grid,
            inflation_adjuster=inflation_adjuster,
            household=household,
            social_security_benefits=benefits,
            strategy=strategy,
        )

        # Test retirement benefit
        benefits_2057 = engine.get_benefits_for_year(2057)
        retirement_benefit = next(b for b in benefits_2057 if b.benefit_type == "retirement")
        assert retirement_benefit.person == "primary"
        assert retirement_benefit.annual_amount == 30000.0

        # Test spousal benefit
        benefits_2055 = engine.get_benefits_for_year(2055)
        spousal_benefit = next(b for b in benefits_2055 if b.benefit_type == "spousal")
        assert spousal_benefit.person == "spouse"
        assert spousal_benefit.annual_amount == 15000.0

        # Test survivor benefit
        benefits_2060 = engine.get_benefits_for_year(2060)
        survivor_benefit = next(b for b in benefits_2060 if b.benefit_type == "survivor")
        assert survivor_benefit.person == "spouse"
        assert survivor_benefit.annual_amount == 20000.0

    def test_person_age_calculation(self, social_security_engine):
        """Test person age calculation."""
        # Primary person age in 2024 (base year)
        age = social_security_engine._get_person_age("primary", 2024)
        assert age == 35

        # Primary person age in 2057 (claim year)
        age = social_security_engine._get_person_age("primary", 2057)
        assert age == 68

        # Spouse age in 2024
        age = social_security_engine._get_person_age("spouse", 2024)
        assert age == 33

        # Spouse age in 2057 (claim year)
        age = social_security_engine._get_person_age("spouse", 2057)
        assert age == 66  # 33 + (2057-2024) = 66

    def test_benefit_active_check(self, social_security_engine):
        """Test benefit active status check."""
        primary_benefit = social_security_engine.social_security_benefits[0]
        spouse_benefit = social_security_engine.social_security_benefits[1]

        # Before claim year
        assert not social_security_engine._is_benefit_active(primary_benefit, 2056)
        assert not social_security_engine._is_benefit_active(spouse_benefit, 2056)

        # At claim year
        assert social_security_engine._is_benefit_active(primary_benefit, 2057)
        assert social_security_engine._is_benefit_active(spouse_benefit, 2057)

        # After claim year
        assert social_security_engine._is_benefit_active(primary_benefit, 2058)
        assert social_security_engine._is_benefit_active(spouse_benefit, 2058)


class TestSocialSecurityEngineIntegration:
    """Test Social Security engine integration functions."""

    @pytest.fixture
    def sample_scenario(self):
        """Create a sample scenario for testing."""
        from app.models.scenario import (
            Accounts, Liabilities, Expenses, Policies
        )
        household = Household(
            primary_age=35,
            spouse_age=33,
            filing_status="married_filing_jointly",
            state="CA",
        )

        social_security_benefits = [
            SocialSecurity(
                person="primary",
                annual_benefit=30000.0,
                claim_year=2057,
                benefit_type="retirement",
            ),
            SocialSecurity(
                person="spouse",
                annual_benefit=15000.0,
                claim_year=2055,
                benefit_type="retirement",
            ),
        ]

        incomes = Incomes(social_security=social_security_benefits)

        social_security_strategy = SocialSecurityStrategy(
            primary_claim_age=68,
            spouse_claim_age=65,
        )

        strategy = Strategy(social_security_strategy=social_security_strategy)

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

        engine = create_social_security_engine_from_scenario(
            sample_scenario, time_grid, inflation_adjuster
        )

        assert engine.household.primary_age == 35
        assert engine.household.spouse_age == 33
        assert len(engine.social_security_benefits) == 2
        assert engine.strategy.primary_claim_age == 68
        assert engine.cola_rate == 0.025  # From market model inflation

    def test_validate_social_security_timing_valid(self):
        """Test validation with valid timing."""
        household = Household(
            primary_age=35,
            spouse_age=33,
            filing_status="married_filing_jointly",
            state="CA",
        )

        benefits = [
            SocialSecurity(
                person="primary",
                annual_benefit=30000.0,
                claim_year=2057,  # Age 68
                benefit_type="retirement",
            ),
            SocialSecurity(
                person="spouse",
                annual_benefit=15000.0,
                claim_year=2055,  # Age 65
                benefit_type="retirement",
            ),
        ]

        errors = validate_social_security_timing(benefits, household)
        assert len(errors) == 0

    def test_validate_social_security_timing_invalid_ages(self):
        """Test validation with invalid claim ages."""
        household = Household(
            primary_age=35,
            spouse_age=33,
            filing_status="married_filing_jointly",
            state="CA",
        )

        benefits = [
            SocialSecurity(
                person="primary",
                annual_benefit=30000.0,
                claim_year=2051,  # Age 62 (at minimum, should be valid)
                benefit_type="retirement",
            ),
            SocialSecurity(
                person="spouse",
                annual_benefit=15000.0,
                claim_year=2062,  # Age 71 (too late)
                benefit_type="retirement",
            ),
        ]

        errors = validate_social_security_timing(benefits, household)
        assert len(errors) == 1
        # Check that we have an error for the spouse
        error_messages = " ".join(errors)
        assert "spouse" in error_messages
        assert "above maximum 70" in error_messages

    def test_validate_social_security_timing_spouse_benefit_no_spouse(self):
        """Test validation with spouse benefit but no spouse."""
        household = Household(
            primary_age=35,
            filing_status="single",
            state="CA",
        )

        benefits = [
            SocialSecurity(
                person="spouse",
                annual_benefit=15000.0,
                claim_year=2055,
                benefit_type="retirement",
            ),
        ]

        errors = validate_social_security_timing(benefits, household)
        assert len(errors) == 1
        assert "no spouse in household" in errors[0]

    def test_validate_social_security_timing_negative_benefit(self):
        """Test validation with negative benefit amount."""
        household = Household(
            primary_age=35,
            filing_status="single",
            state="CA",
        )

        # This should be caught by Pydantic validation, not our custom validation
        with pytest.raises(ValueError):
            SocialSecurity(
                person="primary",
                annual_benefit=-1000.0,  # Invalid negative amount
                claim_year=2057,
                benefit_type="retirement",
            )

    def test_calculate_social_security_present_value(self):
        """Test present value calculation."""
        benefits = [
            SocialSecurityBenefit(
                year=2057,
                person="primary",
                benefit_type="retirement",
                annual_amount=30000.0,
                description="Primary benefit year 1",
            ),
            SocialSecurityBenefit(
                year=2058,
                person="primary",
                benefit_type="retirement",
                annual_amount=30600.0,  # With COLA
                description="Primary benefit year 2",
            ),
        ]

        present_value = calculate_social_security_present_value(benefits, 0.03)
        assert present_value > 0
        assert present_value < 60000  # Should be less than sum due to discounting

    def test_calculate_social_security_present_value_empty(self):
        """Test present value calculation with empty benefits."""
        benefits = []
        present_value = calculate_social_security_present_value(benefits, 0.03)
        assert present_value == 0.0

    def test_calculate_social_security_present_value_zero_discount(self):
        """Test present value calculation with zero discount rate."""
        benefits = [
            SocialSecurityBenefit(
                year=2057,
                person="primary",
                benefit_type="retirement",
                annual_amount=30000.0,
                description="Primary benefit",
            ),
        ]

        present_value = calculate_social_security_present_value(benefits, 0.0)
        assert present_value == 30000.0


class TestSocialSecurityEngineEdgeCases:
    """Test edge cases and error conditions."""

    def test_engine_with_no_benefits(self):
        """Test engine with no Social Security benefits."""
        time_grid = TimeGrid(start_year=2024, end_year=2060, base_year=2024)
        inflation_adjuster = InflationAdjuster(base_year=2024, inflation_rate=0.025)
        household = Household(
            primary_age=35,
            filing_status="single",
            state="CA",
        )
        strategy = SocialSecurityStrategy()

        engine = SocialSecurityEngine(
            time_grid=time_grid,
            inflation_adjuster=inflation_adjuster,
            household=household,
            social_security_benefits=[],
            strategy=strategy,
        )

        # Should return empty results
        benefits = engine.get_benefits_for_year(2057)
        assert len(benefits) == 0

        total = engine.get_total_annual_benefits(2057)
        assert total == 0.0

        benefits_by_person = engine.get_benefits_by_person(2057)
        assert benefits_by_person["primary"] == 0.0
        assert benefits_by_person["spouse"] == 0.0

    def test_engine_with_high_cola_rate(self):
        """Test engine with high COLA rate."""
        time_grid = TimeGrid(start_year=2024, end_year=2060, base_year=2024)
        inflation_adjuster = InflationAdjuster(base_year=2024, inflation_rate=0.025)
        household = Household(
            primary_age=35,
            filing_status="single",
            state="CA",
        )

        benefits = [
            SocialSecurity(
                person="primary",
                annual_benefit=30000.0,
                claim_year=2057,
                benefit_type="retirement",
            )
        ]

        strategy = SocialSecurityStrategy()

        engine = SocialSecurityEngine(
            time_grid=time_grid,
            inflation_adjuster=inflation_adjuster,
            household=household,
            social_security_benefits=benefits,
            strategy=strategy,
            cola_rate=0.10,  # 10% COLA
        )

        # Test high COLA impact
        benefits_2057 = engine.get_benefits_for_year(2057)
        assert benefits_2057[0].annual_amount == 30000.0

        benefits_2058 = engine.get_benefits_for_year(2058)
        expected_amount = 30000.0 * 1.10  # 10% COLA
        assert abs(benefits_2058[0].annual_amount - expected_amount) < 0.01

    def test_engine_validation_errors(self):
        """Test engine validation with invalid parameters."""
        time_grid = TimeGrid(start_year=2024, end_year=2060, base_year=2024)
        inflation_adjuster = InflationAdjuster(base_year=2024, inflation_rate=0.025)
        household = Household(
            primary_age=35,
            filing_status="single",
            state="CA",
        )

        benefits = [
            SocialSecurity(
                person="primary",
                annual_benefit=30000.0,
                claim_year=2057,
                benefit_type="retirement",
            )
        ]

        strategy = SocialSecurityStrategy()

        # Test negative COLA rate
        with pytest.raises(ValueError):
            SocialSecurityEngine(
                time_grid=time_grid,
                inflation_adjuster=inflation_adjuster,
                household=household,
                social_security_benefits=benefits,
                strategy=strategy,
                cola_rate=-0.01,  # Invalid negative COLA
            )

        # Test COLA rate too high
        with pytest.raises(ValueError):
            SocialSecurityEngine(
                time_grid=time_grid,
                inflation_adjuster=inflation_adjuster,
                household=household,
                social_security_benefits=benefits,
                strategy=strategy,
                cola_rate=0.15,  # Invalid high COLA
            )
