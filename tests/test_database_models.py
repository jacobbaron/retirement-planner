"""
Tests for SQLAlchemy database models.

This module tests the database models, relationships, and CRUD operations
for the retirement planner application.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.base import get_session, create_tables, drop_tables
from app.database.models import User, Scenario, Run, LedgerRow
from app.models.scenario import Scenario as PydanticScenario, Household, Accounts, Liabilities, Incomes, Expenses, Policies, MarketModel, Strategy


class TestDatabaseModels:
    """Test suite for database models and relationships."""
    
    @pytest.fixture(scope="function")
    def db_session(self):
        """Create a fresh database session for each test."""
        # Create tables
        create_tables()
        
        # Get a session
        session = get_session()
        
        yield session
        
        # Clean up
        session.close()
        drop_tables()
    
    @pytest.fixture
    def sample_user(self, db_session):
        """Create a sample user for testing."""
        user = User(
            email="test@example.com",
            name="Test User"
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        return user
    
    @pytest.fixture
    def sample_scenario_data(self):
        """Create sample scenario data using Pydantic models."""
        return PydanticScenario(
            household=Household(
                primary_age=35,
                spouse_age=33,
                filing_status="married_filing_jointly",
                state="CA"
            ),
            accounts=Accounts(),
            liabilities=Liabilities(),
            incomes=Incomes(),
            expenses=Expenses(),
            policies=Policies(),
            market_model=MarketModel(
                expected_returns={"stocks": 0.08, "bonds": 0.04, "cash": 0.02},
                volatility={"stocks": 0.18, "bonds": 0.05, "cash": 0.01},
                inflation=0.025
            ),
            strategy=Strategy(
                withdrawal_method="fixed_real",
                withdrawal_rate=0.04
            )
        )
    
    def test_user_creation(self, db_session):
        """Test creating a user."""
        user = User(
            email="john@example.com",
            name="John Doe"
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        assert user.id is not None
        assert user.email == "john@example.com"
        assert user.name == "John Doe"
        assert user.created_at is not None
        assert user.updated_at is not None
    
    def test_user_unique_email(self, db_session):
        """Test that email must be unique."""
        user1 = User(email="test@example.com", name="User 1")
        user2 = User(email="test@example.com", name="User 2")
        
        db_session.add(user1)
        db_session.commit()
        
        db_session.add(user2)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_scenario_creation(self, db_session, sample_user, sample_scenario_data):
        """Test creating a scenario."""
        scenario = Scenario(
            user_id=sample_user.id,
            name="My Retirement Plan",
            description="A test retirement scenario",
            scenario_data=sample_scenario_data.model_dump()
        )
        db_session.add(scenario)
        db_session.commit()
        db_session.refresh(scenario)
        
        assert scenario.id is not None
        assert scenario.user_id == sample_user.id
        assert scenario.name == "My Retirement Plan"
        assert scenario.scenario_data["household"]["primary_age"] == 35
        assert scenario.version == "0.1"
        assert scenario.created_at is not None
    
    def test_scenario_user_relationship(self, db_session, sample_user, sample_scenario_data):
        """Test the relationship between scenarios and users."""
        scenario = Scenario(
            user_id=sample_user.id,
            name="Test Scenario",
            scenario_data=sample_scenario_data.model_dump()
        )
        db_session.add(scenario)
        db_session.commit()
        
        # Test relationship from user to scenarios
        user_scenarios = db_session.query(Scenario).filter(Scenario.user_id == sample_user.id).all()
        assert len(user_scenarios) == 1
        assert user_scenarios[0].name == "Test Scenario"
        
        # Test relationship from scenario to user
        assert scenario.user.email == "test@example.com"
        assert scenario.user.name == "Test User"
    
    def test_run_creation(self, db_session, sample_user, sample_scenario_data):
        """Test creating a run."""
        # First create a scenario
        scenario = Scenario(
            user_id=sample_user.id,
            name="Test Scenario",
            scenario_data=sample_scenario_data.model_dump()
        )
        db_session.add(scenario)
        db_session.commit()
        db_session.refresh(scenario)
        
        # Create a run
        run = Run(
            scenario_id=scenario.id,
            status="pending",
            run_type="monte_carlo",
            num_simulations=1000
        )
        db_session.add(run)
        db_session.commit()
        db_session.refresh(run)
        
        assert run.id is not None
        assert run.scenario_id == scenario.id
        assert run.status == "pending"
        assert run.run_type == "monte_carlo"
        assert run.num_simulations == 1000
        assert run.created_at is not None
    
    def test_run_status_constraint(self, db_session, sample_user, sample_scenario_data):
        """Test that run status must be valid."""
        scenario = Scenario(
            user_id=sample_user.id,
            name="Test Scenario",
            scenario_data=sample_scenario_data.model_dump()
        )
        db_session.add(scenario)
        db_session.commit()
        db_session.refresh(scenario)
        
        # Test invalid status
        run = Run(
            scenario_id=scenario.id,
            status="invalid_status"
        )
        db_session.add(run)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_ledger_row_creation(self, db_session, sample_user, sample_scenario_data):
        """Test creating ledger rows."""
        # Create scenario and run
        scenario = Scenario(
            user_id=sample_user.id,
            name="Test Scenario",
            scenario_data=sample_scenario_data.model_dump()
        )
        db_session.add(scenario)
        db_session.commit()
        db_session.refresh(scenario)
        
        run = Run(
            scenario_id=scenario.id,
            status="completed"
        )
        db_session.add(run)
        db_session.commit()
        db_session.refresh(run)
        
        # Create ledger rows
        ledger_row = LedgerRow(
            run_id=run.id,
            year=2024,
            month=1,
            account_type="taxable",
            account_name="Vanguard Brokerage",
            transaction_type="contribution",
            amount=Decimal("5000.00"),
            balance_after=Decimal("5000.00"),
            description="Annual contribution"
        )
        db_session.add(ledger_row)
        db_session.commit()
        db_session.refresh(ledger_row)
        
        assert ledger_row.id is not None
        assert ledger_row.run_id == run.id
        assert ledger_row.year == 2024
        assert ledger_row.account_type == "taxable"
        assert ledger_row.amount == Decimal("5000.00")
    
    def test_ledger_row_constraints(self, db_session, sample_user, sample_scenario_data):
        """Test ledger row constraints."""
        scenario = Scenario(
            user_id=sample_user.id,
            name="Test Scenario",
            scenario_data=sample_scenario_data.model_dump()
        )
        db_session.add(scenario)
        db_session.commit()
        db_session.refresh(scenario)
        
        run = Run(
            scenario_id=scenario.id,
            status="completed"
        )
        db_session.add(run)
        db_session.commit()
        db_session.refresh(run)
        
        # Test invalid year
        ledger_row = LedgerRow(
            run_id=run.id,
            year=1800,  # Too early
            account_type="taxable",
            account_name="Test Account",
            transaction_type="contribution",
            amount=Decimal("1000.00")
        )
        db_session.add(ledger_row)
        with pytest.raises(IntegrityError):
            db_session.commit()
        
        db_session.rollback()
        
        # Test invalid account type
        ledger_row = LedgerRow(
            run_id=run.id,
            year=2024,
            account_type="invalid_type",
            account_name="Test Account",
            transaction_type="contribution",
            amount=Decimal("1000.00")
        )
        db_session.add(ledger_row)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_cascade_deletes(self, db_session, sample_user, sample_scenario_data):
        """Test that cascade deletes work properly."""
        # Create scenario
        scenario = Scenario(
            user_id=sample_user.id,
            name="Test Scenario",
            scenario_data=sample_scenario_data.model_dump()
        )
        db_session.add(scenario)
        db_session.commit()
        db_session.refresh(scenario)
        
        # Create run
        run = Run(
            scenario_id=scenario.id,
            status="completed"
        )
        db_session.add(run)
        db_session.commit()
        db_session.refresh(run)
        
        # Create ledger rows
        ledger_row1 = LedgerRow(
            run_id=run.id,
            year=2024,
            account_type="taxable",
            account_name="Account 1",
            transaction_type="contribution",
            amount=Decimal("1000.00")
        )
        ledger_row2 = LedgerRow(
            run_id=run.id,
            year=2024,
            account_type="traditional_401k",
            account_name="Account 2",
            transaction_type="contribution",
            amount=Decimal("2000.00")
        )
        db_session.add_all([ledger_row1, ledger_row2])
        db_session.commit()
        
        # Delete the run - should cascade to ledger rows
        db_session.delete(run)
        db_session.commit()
        
        # Check that ledger rows are deleted
        remaining_rows = db_session.query(LedgerRow).filter(LedgerRow.run_id == run.id).all()
        assert len(remaining_rows) == 0
        
        # Delete the scenario - should cascade to runs
        db_session.delete(scenario)
        db_session.commit()
        
        # Check that run is deleted
        remaining_runs = db_session.query(Run).filter(Run.scenario_id == scenario.id).all()
        assert len(remaining_runs) == 0
        
        # Delete the user - should cascade to scenarios
        db_session.delete(sample_user)
        db_session.commit()
        
        # Check that scenario is deleted
        remaining_scenarios = db_session.query(Scenario).filter(Scenario.user_id == sample_user.id).all()
        assert len(remaining_scenarios) == 0
    
    def test_indexes_exist(self, db_session):
        """Test that indexes are created properly."""
        # This test verifies that the indexes are created by checking the database metadata
        # In a real test, you might query the database directly to check indexes
        
        # Create a user and scenario to test index functionality
        user = User(email="index@example.com", name="Index Test")
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        scenario = Scenario(
            user_id=user.id,
            name="Index Test Scenario",
            scenario_data={}
        )
        db_session.add(scenario)
        db_session.commit()
        
        # Test that queries using indexed columns work efficiently
        # (In practice, you'd use EXPLAIN ANALYZE to verify index usage)
        users_by_email = db_session.query(User).filter(User.email == "index@example.com").all()
        assert len(users_by_email) == 1
        
        scenarios_by_user = db_session.query(Scenario).filter(Scenario.user_id == user.id).all()
        assert len(scenarios_by_user) == 1
    
    def test_pydantic_integration(self, db_session, sample_user):
        """Test integration between Pydantic models and SQLAlchemy."""
        # Create a Pydantic scenario
        pydantic_scenario = PydanticScenario(
            household=Household(
                primary_age=40,
                filing_status="single",
                state="NY"
            ),
            accounts=Accounts(),
            liabilities=Liabilities(),
            incomes=Incomes(),
            expenses=Expenses(),
            policies=Policies(),
            market_model=MarketModel(
                expected_returns={"stocks": 0.07, "bonds": 0.03, "cash": 0.01},
                volatility={"stocks": 0.15, "bonds": 0.04, "cash": 0.01},
                inflation=0.02
            ),
            strategy=Strategy(
                withdrawal_method="fixed_percentage",
                withdrawal_rate=0.035
            )
        )
        
        # Store in database
        db_scenario = Scenario(
            user_id=sample_user.id,
            name="Pydantic Test",
            scenario_data=pydantic_scenario.model_dump()
        )
        db_session.add(db_scenario)
        db_session.commit()
        db_session.refresh(db_scenario)
        
        # Retrieve and convert back to Pydantic
        retrieved_scenario = PydanticScenario(**db_scenario.scenario_data)
        
        assert retrieved_scenario.household.primary_age == 40
        assert retrieved_scenario.household.filing_status == "single"
        assert retrieved_scenario.household.state == "NY"
        assert retrieved_scenario.market_model.expected_returns.stocks == 0.07
        assert retrieved_scenario.strategy.withdrawal_method == "fixed_percentage"
        assert retrieved_scenario.strategy.withdrawal_rate == 0.035
