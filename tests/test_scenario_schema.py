"""
Tests for retirement planning scenario Pydantic models and validation.

This module tests the Pydantic models against sample fixtures
to ensure proper validation of retirement planning scenarios.
"""

import json
import pytest
from pathlib import Path
from datetime import datetime
from pydantic import ValidationError

from app.models.scenario import (
    Scenario, Household, Accounts, Liabilities, Incomes, Expenses, 
    Policies, MarketModel, Strategy, AssetAllocation
)


class TestScenarioModels:
    """Test suite for scenario Pydantic model validation."""
    
    @pytest.fixture
    def sample_scenario_data(self):
        """Load the sample scenario fixture data."""
        fixture_path = Path(__file__).parent / "fixtures" / "sample_scenario.json"
        with open(fixture_path, 'r') as f:
            return json.load(f)
    
    @pytest.fixture
    def sample_scenario(self, sample_scenario_data):
        """Create a Scenario instance from sample data."""
        return Scenario(**sample_scenario_data)
    
    def test_scenario_model_creation(self, sample_scenario):
        """Test that the Scenario model can be created from sample data."""
        assert isinstance(sample_scenario, Scenario)
        assert sample_scenario.household.primary_age == 35
        assert sample_scenario.household.spouse_age == 33
        assert sample_scenario.household.filing_status == "married_filing_jointly"
    
    def test_scenario_has_all_required_fields(self, sample_scenario):
        """Test that the sample scenario has all required fields."""
        assert sample_scenario.household is not None
        assert sample_scenario.accounts is not None
        assert sample_scenario.liabilities is not None
        assert sample_scenario.incomes is not None
        assert sample_scenario.expenses is not None
        assert sample_scenario.policies is not None
        assert sample_scenario.market_model is not None
        assert sample_scenario.strategy is not None
    
    def test_scenario_serialization(self, sample_scenario):
        """Test that the scenario can be serialized to JSON."""
        json_data = sample_scenario.model_dump_json()
        assert isinstance(json_data, str)
        
        # Should be able to parse it back
        parsed_data = json.loads(json_data)
        assert parsed_data["household"]["primary_age"] == 35
    
    def test_household_validation(self, sample_scenario):
        """Test household section validation."""
        household = sample_scenario.household
        
        # Test required fields
        assert household.primary_age == 35
        assert household.spouse_age == 33
        assert household.filing_status == "married_filing_jointly"
        assert household.state == "CA"
        
        # Test data types and constraints
        assert isinstance(household.primary_age, int)
        assert 18 <= household.primary_age <= 100
        assert household.filing_status in [
            "single", "married_filing_jointly", "married_filing_separately", 
            "head_of_household", "qualifying_widow"
        ]
        assert len(household.state) == 2
        
        # Test children array
        assert len(household.children) == 2
        for child in household.children:
            assert child.birth_year in [2010, 2015]
            assert child.college_start_year in [2028, 2033]
            assert child.college_type == "public"
    
    def test_accounts_validation(self, sample_scenario):
        """Test accounts section validation."""
        accounts = sample_scenario.accounts
        
        # Test taxable accounts
        assert len(accounts.taxable) == 1
        taxable_account = accounts.taxable[0]
        assert taxable_account.name == "Vanguard Brokerage"
        assert taxable_account.current_balance == 50000
        assert taxable_account.cost_basis == 45000
        
        # Test asset allocation
        allocation = taxable_account.asset_allocation
        assert allocation.stocks == 0.8
        assert allocation.bonds == 0.15
        assert allocation.cash == 0.05
        
        # Test that allocation sums to 1.0
        total = allocation.stocks + allocation.bonds + allocation.cash
        assert abs(total - 1.0) < 0.001, f"Asset allocation doesn't sum to 1.0: {total}"
        
        # Test 401k accounts
        assert len(accounts.traditional_401k) == 1
        trad_401k = accounts.traditional_401k[0]
        assert trad_401k.name == "Company 401k"
        assert trad_401k.current_balance == 150000
        assert trad_401k.employer_match == 5000
        
        # Test cash accounts
        assert accounts.cash.checking == 5000
        assert accounts.cash.savings == 15000
        assert accounts.cash.cds == 10000
        assert accounts.cash.money_market == 5000
    
    def test_liabilities_validation(self, sample_scenario):
        """Test liabilities section validation."""
        liabilities = sample_scenario.liabilities
        
        # Test mortgage structure
        assert len(liabilities.mortgages) == 1
        mortgage = liabilities.mortgages[0]
        assert mortgage.name == "Primary Residence"
        assert mortgage.current_balance == 350000
        assert mortgage.interest_rate == 0.035
        assert mortgage.term_years == 30
        assert mortgage.start_year == 2020
        
        # Test student loans
        assert len(liabilities.student_loans) == 1
        student_loan = liabilities.student_loans[0]
        assert student_loan.name == "Primary Student Loan"
        assert student_loan.current_balance == 25000
        assert student_loan.loan_type == "federal"
    
    def test_incomes_validation(self, sample_scenario):
        """Test incomes section validation."""
        incomes = sample_scenario.incomes
        
        # Test salary structure
        assert len(incomes.salary) == 2
        primary_salary = incomes.salary[0]
        assert primary_salary.person == "primary"
        assert primary_salary.annual_amount == 120000
        assert primary_salary.growth_rate == 0.03
        assert primary_salary.bonus == 10000
        
        # Test social security structure
        assert len(incomes.social_security) == 2
        primary_ss = incomes.social_security[0]
        assert primary_ss.person == "primary"
        assert primary_ss.annual_benefit == 35000
        assert primary_ss.claim_year == 2057
    
    def test_expenses_validation(self, sample_scenario):
        """Test expenses section validation."""
        expenses = sample_scenario.expenses
        
        # Test housing expenses
        housing = expenses.housing
        assert housing.mortgage_payment == 2000
        assert housing.property_tax == 6000
        assert housing.home_insurance == 1200
        assert housing.utilities == 400
        
        # Test lumpy expenses
        assert len(expenses.lumpy_expenses) == 2
        roof_expense = expenses.lumpy_expenses[0]
        assert roof_expense.name == "New Roof"
        assert roof_expense.amount == 15000
        assert roof_expense.year == 2030
        assert roof_expense.category == "home_improvement"
    
    def test_market_model_validation(self, sample_scenario):
        """Test market model section validation."""
        market_model = sample_scenario.market_model
        
        # Test expected returns
        returns = market_model.expected_returns
        assert returns.stocks == 0.08
        assert returns.bonds == 0.04
        assert returns.cash == 0.02
        
        # Test volatility
        volatility = market_model.volatility
        assert volatility.stocks == 0.18
        assert volatility.bonds == 0.05
        assert volatility.cash == 0.01
        
        # Test inflation
        assert market_model.inflation == 0.025
        assert market_model.simulation_type == "monte_carlo"
        assert market_model.num_simulations == 10000
    
    def test_strategy_validation(self, sample_scenario):
        """Test strategy section validation."""
        strategy = sample_scenario.strategy
        
        # Test withdrawal method
        assert strategy.withdrawal_method == "fixed_real"
        assert strategy.withdrawal_rate == 0.04
        assert strategy.withdrawal_start_year == 2054
        
        # Test rebalancing
        rebalancing = strategy.rebalancing
        assert rebalancing.frequency == "annual"
        assert rebalancing.threshold == 0.05
        
        # Test tax optimization
        tax_opt = strategy.tax_optimization
        assert tax_opt.withdrawal_order == ["cash", "taxable", "traditional", "roth"]
        assert tax_opt.tax_loss_harvesting is True
        assert tax_opt.roth_conversions is False
    
    def test_invalid_scenario_rejected(self):
        """Test that invalid scenarios are properly rejected."""
        # Test missing required field
        with pytest.raises(ValidationError):
            Scenario(household={"primary_age": 35})  # Missing other required fields
        
        # Test invalid data type
        with pytest.raises(ValidationError):
            Scenario(
                household={
                    "primary_age": "not_a_number",  # Should be integer
                    "spouse_age": None,
                    "filing_status": "married_filing_jointly",
                    "state": "CA"
                },
                accounts={},
                liabilities={},
                incomes={},
                expenses={},
                policies={},
                market_model={
                    "expected_returns": {"stocks": 0.08, "bonds": 0.04, "cash": 0.02},
                    "volatility": {"stocks": 0.18, "bonds": 0.05, "cash": 0.01},
                    "inflation": 0.025
                },
                strategy={
                    "withdrawal_method": "fixed_real",
                    "rebalancing": {"frequency": "annual", "threshold": 0.05}
                }
            )
    
    def test_asset_allocation_constraints(self):
        """Test that asset allocation values are properly constrained."""
        # Test invalid asset allocation (> 1.0)
        with pytest.raises(ValidationError):
            AssetAllocation(stocks=1.5, bonds=0.3, cash=0.2)
        
        # Test invalid asset allocation (doesn't sum to 1.0)
        with pytest.raises(ValidationError):
            AssetAllocation(stocks=0.5, bonds=0.3, cash=0.1)  # Sums to 0.9
    
    def test_json_schema_generation(self):
        """Test that JSON schema can be generated from Pydantic models."""
        from app.models.schema_generator import generate_scenario_schema
        
        schema = generate_scenario_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "required" in schema
        
        # Check that all required fields are in the schema
        required_fields = schema["required"]
        expected_fields = [
            "household", "accounts", "liabilities", "incomes",
            "expenses", "policies", "market_model", "strategy"
        ]
        
        for field in expected_fields:
            assert field in required_fields
    
    def test_minimal_valid_scenario(self):
        """Test that a minimal valid scenario can be created."""
        minimal_scenario = Scenario(
            household={
                "primary_age": 35,
                "spouse_age": None,
                "filing_status": "single",
                "state": "CA"
            },
            accounts={
                "taxable": [],
                "traditional_401k": [],
                "roth_401k": [],
                "traditional_ira": [],
                "roth_ira": [],
                "hsa": [],
                "college_529": [],
                "cash": {}
            },
            liabilities={
                "mortgages": [],
                "student_loans": [],
                "credit_cards": [],
                "auto_loans": []
            },
            incomes={
                "salary": [],
                "social_security": [],
                "pension": [],
                "other": []
            },
            expenses={
                "housing": {},
                "transportation": {},
                "healthcare": {},
                "food": 0,
                "entertainment": 0,
                "travel": 0,
                "education": 0,
                "other": 0,
                "lumpy_expenses": []
            },
            policies={
                "life_insurance": [],
                "disability_insurance": [],
                "long_term_care": []
            },
            market_model={
                "expected_returns": {
                    "stocks": 0.08,
                    "bonds": 0.04,
                    "cash": 0.02
                },
                "volatility": {
                    "stocks": 0.18,
                    "bonds": 0.05,
                    "cash": 0.01
                },
                "inflation": 0.025
            },
            strategy={
                "withdrawal_method": "fixed_real",
                "withdrawal_rate": 0.04,
                "rebalancing": {
                    "frequency": "annual",
                    "threshold": 0.05
                }
            }
        )
        
        assert isinstance(minimal_scenario, Scenario)
        assert minimal_scenario.household.primary_age == 35
        assert minimal_scenario.household.filing_status == "single"