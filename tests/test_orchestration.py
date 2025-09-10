"""
Tests for the orchestration blueprint and service.

This module tests the API endpoints and orchestration service for running
retirement planning simulations.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
from flask import Flask
from sqlalchemy.orm import Session

from app import create_app
from app.database.base import get_db
from app.database.models import Run, VersionedScenario, User
from app.services.orchestration_service import OrchestrationService


class TestOrchestrationService:
    """Test the OrchestrationService class."""

    def test_initialization(self):
        """Test service initialization."""
        service = OrchestrationService()
        assert service is not None
        assert service.logger is not None

    def test_run_deterministic_simulation(self):
        """Test deterministic simulation."""
        service = OrchestrationService()
        
        # Create a simple scenario
        scenario_data = {
            "household": {
                "primary_age": 35,
                "spouse_age": 33,
                "filing_status": "married_filing_jointly",
                "state": "CA",
                "children": []
            },
            "accounts": {
                "taxable": [{"balance": 100000, "asset_mix": {"stocks": 0.7, "bonds": 0.3}}],
                "traditional_401k": [{"balance": 200000, "asset_mix": {"stocks": 0.6, "bonds": 0.4}}],
                "roth_401k": [],
                "traditional_ira": [],
                "roth_ira": [{"balance": 50000, "asset_mix": {"stocks": 0.8, "bonds": 0.2}}],
                "hsa": [],
                "college_529": [],
                "cash": {"checking": 10000, "savings": 20000}
            },
            "liabilities": {
                "mortgages": [],
                "student_loans": [],
                "credit_cards": [],
                "auto_loans": []
            },
            "incomes": {
                "employment": [{"annual_amount": 120000, "growth_rate": 0.03}],
                "self_employment": [],
                "business": [],
                "investment": [],
                "rental": [],
                "retirement": [],
                "variable": []
            },
            "expenses": {
                "housing": {
                    "mortgage_payment": 2000,
                    "property_tax": 6000,
                    "home_insurance": 1200,
                    "hoa_fees": 0,
                    "maintenance": 3000,
                    "utilities": 300
                },
                "transportation": {
                    "auto_payment": 400,
                    "auto_insurance": 1200,
                    "maintenance": 2000,
                    "gas": 200
                },
                "healthcare": {
                    "insurance": 500,
                    "out_of_pocket": 2000,
                    "medicare": 0
                },
                "food": 800,
                "entertainment": 300,
                "travel": 5000,
                "education": 0,
                "other": 200,
                "lumpy_expenses": []
            },
            "policies": {
                "life_insurance": [],
                "disability_insurance": [],
                "long_term_care_insurance": []
            },
            "market_model": {
                "expected_returns": {"stocks": 0.07, "bonds": 0.03},
                "volatility": {"stocks": 0.18, "bonds": 0.06},
                "correlations": {"matrix": [[1.0, -0.2], [-0.2, 1.0]]},
                "inflation": 0.025,
                "simulation_type": "deterministic",
                "num_simulations": 1000
            },
            "strategy": {
                "withdrawal": {"type": "fixed_percentage", "initial_rate": 0.04},
                "rebalancing": {"enabled": True, "frequency": "annual"},
                "tax_optimization": {"enabled": False},
                "social_security": {"primary_start_age": 67, "spouse_start_age": 67}
            }
        }
        
        result = service._run_deterministic_simulation(scenario_data)
        
        assert result["simulation_type"] == "deterministic"
        assert result["years"] == 30
        assert result["initial_balance"] > 0
        assert result["annual_return"] == 0.07
        assert result["annual_withdrawal"] > 0
        assert len(result["portfolio_evolution"]) == 30
        assert result["success_rate"] in [0.0, 1.0]
        assert result["terminal_wealth"] >= 0

    def test_run_monte_carlo_simulation(self):
        """Test Monte Carlo simulation."""
        service = OrchestrationService()
        
        # Create a simple scenario
        scenario_data = {
            "household": {
                "primary_age": 35,
                "spouse_age": 33,
                "filing_status": "married_filing_jointly",
                "state": "CA",
                "children": []
            },
            "accounts": {
                "taxable": [{"balance": 100000, "asset_mix": {"stocks": 0.7, "bonds": 0.3}}],
                "traditional_401k": [{"balance": 200000, "asset_mix": {"stocks": 0.6, "bonds": 0.4}}],
                "roth_401k": [],
                "traditional_ira": [],
                "roth_ira": [{"balance": 50000, "asset_mix": {"stocks": 0.8, "bonds": 0.2}}],
                "hsa": [],
                "college_529": [],
                "cash": {"checking": 10000, "savings": 20000}
            },
            "liabilities": {
                "mortgages": [],
                "student_loans": [],
                "credit_cards": [],
                "auto_loans": []
            },
            "incomes": {
                "employment": [{"annual_amount": 120000, "growth_rate": 0.03}],
                "self_employment": [],
                "business": [],
                "investment": [],
                "rental": [],
                "retirement": [],
                "variable": []
            },
            "expenses": {
                "housing": {
                    "mortgage_payment": 2000,
                    "property_tax": 6000,
                    "home_insurance": 1200,
                    "hoa_fees": 0,
                    "maintenance": 3000,
                    "utilities": 300
                },
                "transportation": {
                    "auto_payment": 400,
                    "auto_insurance": 1200,
                    "maintenance": 2000,
                    "gas": 200
                },
                "healthcare": {
                    "insurance": 500,
                    "out_of_pocket": 2000,
                    "medicare": 0
                },
                "food": 800,
                "entertainment": 300,
                "travel": 5000,
                "education": 0,
                "other": 200,
                "lumpy_expenses": []
            },
            "policies": {
                "life_insurance": [],
                "disability_insurance": [],
                "long_term_care_insurance": []
            },
            "market_model": {
                "expected_returns": {"stocks": 0.07, "bonds": 0.03},
                "volatility": {"stocks": 0.18, "bonds": 0.06},
                "correlations": {"matrix": [[1.0, -0.2], [-0.2, 1.0]]},
                "inflation": 0.025,
                "simulation_type": "monte_carlo",
                "num_simulations": 1000
            },
            "strategy": {
                "withdrawal": {"type": "fixed_percentage", "initial_rate": 0.04},
                "rebalancing": {"enabled": True, "frequency": "annual"},
                "tax_optimization": {"enabled": False},
                "social_security": {"primary_start_age": 67, "spouse_start_age": 67}
            }
        }
        
        result = service._run_monte_carlo_simulation(scenario_data, 100)
        
        assert result["simulation_type"] == "monte_carlo"
        assert result["years"] == 30
        assert result["num_simulations"] == 100
        assert result["initial_balance"] > 0
        assert result["annual_withdrawal"] > 0
        assert result["mean_return"] == 0.07
        assert result["std_return"] == 0.18
        assert 0 <= result["success_rate"] <= 1
        assert "terminal_wealth_percentiles" in result
        assert "p5" in result["terminal_wealth_percentiles"]
        assert "p50" in result["terminal_wealth_percentiles"]
        assert "p95" in result["terminal_wealth_percentiles"]

    def test_invalid_run_type(self):
        """Test invalid run type raises error."""
        service = OrchestrationService()
        
        with pytest.raises(ValueError, match="Unsupported run type"):
            service.run_simulation(
                run_id=1,
                scenario_config={},
                run_type="invalid_type",
                num_simulations=100
            )


class TestOrchestrationBlueprint:
    """Test the orchestration blueprint endpoints."""

    @pytest.fixture
    def app(self, db_engine):
        """Create test app."""
        app = create_app("testing")
        
        # Override the database URL to use the test database
        app.config["DATABASE_URL"] = str(db_engine.url)
        
        with app.app_context():
            yield app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()

    @pytest.fixture
    def db_session(self, app):
        """Create database session."""
        with app.app_context():
            db = next(get_db())
            yield db
            db.rollback()

    @pytest.fixture
    def sample_user(self, db_session):
        """Create sample user."""
        user = User(
            email="test@example.com",
            name="Test User"
        )
        db_session.add(user)
        db_session.commit()
        return user

    @pytest.fixture
    def sample_scenario(self, db_session, sample_user):
        """Create sample scenario."""
        scenario = VersionedScenario(
            user_id=sample_user.id,
            scenario_name="Test Scenario",
            scenario_config={
                "household": {
                    "primary_age": 35,
                    "spouse_age": 33,
                    "filing_status": "married_filing_jointly",
                    "state": "CA",
                    "children": []
                },
                "accounts": {
                    "taxable": [{"balance": 100000, "asset_mix": {"stocks": 0.7, "bonds": 0.3}}],
                    "traditional_401k": [{"balance": 200000, "asset_mix": {"stocks": 0.6, "bonds": 0.4}}],
                    "roth_401k": [],
                    "traditional_ira": [],
                    "roth_ira": [{"balance": 50000, "asset_mix": {"stocks": 0.8, "bonds": 0.2}}],
                    "hsa": [],
                    "college_529": [],
                    "cash": {"checking": 10000, "savings": 20000}
                },
                "liabilities": {
                    "mortgages": [],
                    "student_loans": [],
                    "credit_cards": [],
                    "auto_loans": []
                },
                "incomes": {
                    "employment": [{"annual_amount": 120000, "growth_rate": 0.03}],
                    "self_employment": [],
                    "business": [],
                    "investment": [],
                    "rental": [],
                    "retirement": [],
                    "variable": []
                },
                "expenses": {
                    "housing": {
                        "mortgage_payment": 2000,
                        "property_tax": 6000,
                        "home_insurance": 1200,
                        "hoa_fees": 0,
                        "maintenance": 3000,
                        "utilities": 300
                    },
                    "transportation": {
                        "auto_payment": 400,
                        "auto_insurance": 1200,
                        "maintenance": 2000,
                        "gas": 200
                    },
                    "healthcare": {
                        "insurance": 500,
                        "out_of_pocket": 2000,
                        "medicare": 0
                    },
                    "food": 800,
                    "entertainment": 300,
                    "travel": 5000,
                    "education": 0,
                    "other": 200,
                    "lumpy_expenses": []
                },
                "policies": {
                    "life_insurance": [],
                    "disability_insurance": [],
                    "long_term_care_insurance": []
                },
                "market_model": {
                    "expected_returns": {"stocks": 0.07, "bonds": 0.03},
                    "volatility": {"stocks": 0.18, "bonds": 0.06},
                    "correlations": {"matrix": [[1.0, -0.2], [-0.2, 1.0]]},
                    "inflation": 0.025,
                    "simulation_type": "monte_carlo",
                    "num_simulations": 1000
                },
                "strategy": {
                    "withdrawal": {"type": "fixed_percentage", "initial_rate": 0.04},
                    "rebalancing": {"enabled": True, "frequency": "annual"},
                    "tax_optimization": {"enabled": False},
                    "social_security": {"primary_start_age": 67, "spouse_start_age": 67}
                }
            }
        )
        db_session.add(scenario)
        db_session.commit()
        return scenario

    def test_start_run_success(self, client, sample_scenario):
        """Test successful run start."""
        with patch('app.services.orchestration_service.OrchestrationService.run_simulation') as mock_run:
            mock_run.return_value = {"success": True}
            
            response = client.post(
                f"/api/scenarios/{sample_scenario.id}/runs",
                json={
                    "run_type": "monte_carlo",
                    "num_simulations": 1000
                }
            )
            
            assert response.status_code == 201
            data = response.get_json()
            assert "run_id" in data
            assert data["status"] == "completed"
            assert data["run_type"] == "monte_carlo"
            assert data["num_simulations"] == 1000

    def test_start_run_invalid_scenario(self, client):
        """Test starting run with invalid scenario ID."""
        with patch('app.blueprints.orchestration.get_db') as mock_get_db:
            # Mock database session to return None for scenario
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            mock_get_db.return_value = iter([mock_db])
            
            response = client.post(
                "/api/scenarios/99999/runs",
                json={
                    "run_type": "monte_carlo",
                    "num_simulations": 1000
                }
            )
            
            assert response.status_code == 404
            data = response.get_json()
            assert "error" in data

    def test_start_run_invalid_run_type(self, client, sample_scenario):
        """Test starting run with invalid run type."""
        response = client.post(
            f"/api/scenarios/{sample_scenario.id}/runs",
            json={
                "run_type": "invalid_type",
                "num_simulations": 1000
            }
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_start_run_invalid_num_simulations(self, client, sample_scenario):
        """Test starting run with invalid number of simulations."""
        response = client.post(
            f"/api/scenarios/{sample_scenario.id}/runs",
            json={
                "run_type": "monte_carlo",
                "num_simulations": -1
            }
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_get_run_status(self, client, sample_scenario, db_session):
        """Test getting run status."""
        # Create a run
        run = Run(
            versioned_scenario_id=sample_scenario.id,
            status="completed",
            run_type="monte_carlo",
            num_simulations=1000,
            results={"success_rate": 0.85}
        )
        db_session.add(run)
        db_session.commit()
        
        response = client.get(f"/api/runs/{run.id}")
        
        assert response.status_code == 200
        data = response.get_json()
        assert data["run_id"] == run.id
        assert data["status"] == "completed"
        assert data["run_type"] == "monte_carlo"
        assert data["num_simulations"] == 1000
        assert "results" in data

    def test_get_run_status_not_found(self, client):
        """Test getting status for non-existent run."""
        response = client.get("/api/runs/99999")
        
        assert response.status_code == 404
        data = response.get_json()
        assert "error" in data

    def test_get_run_results(self, client, sample_scenario, db_session):
        """Test getting run results."""
        # Create a completed run
        run = Run(
            versioned_scenario_id=sample_scenario.id,
            status="completed",
            run_type="monte_carlo",
            num_simulations=1000,
            results={"success_rate": 0.85, "terminal_wealth": 500000}
        )
        db_session.add(run)
        db_session.commit()
        
        response = client.get(f"/api/runs/{run.id}/results")
        
        assert response.status_code == 200
        data = response.get_json()
        assert data["run_id"] == run.id
        assert "results" in data
        assert data["results"]["success_rate"] == 0.85

    def test_get_run_results_not_completed(self, client, sample_scenario, db_session):
        """Test getting results for non-completed run."""
        # Create a pending run
        run = Run(
            versioned_scenario_id=sample_scenario.id,
            status="pending",
            run_type="monte_carlo",
            num_simulations=1000
        )
        db_session.add(run)
        db_session.commit()
        
        response = client.get(f"/api/runs/{run.id}/results")
        
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_list_scenario_runs(self, client, sample_scenario, db_session):
        """Test listing runs for a scenario."""
        # Create multiple runs
        run1 = Run(
            versioned_scenario_id=sample_scenario.id,
            status="completed",
            run_type="monte_carlo",
            num_simulations=1000
        )
        run2 = Run(
            versioned_scenario_id=sample_scenario.id,
            status="pending",
            run_type="deterministic",
            num_simulations=1
        )
        db_session.add_all([run1, run2])
        db_session.commit()
        
        response = client.get(f"/api/scenarios/{sample_scenario.id}/runs")
        
        assert response.status_code == 200
        data = response.get_json()
        assert data["scenario_id"] == sample_scenario.id
        assert len(data["runs"]) == 2
        assert data["runs"][0]["run_id"] == run2.id  # Most recent first
        assert data["runs"][1]["run_id"] == run1.id

    def test_list_scenario_runs_not_found(self, client):
        """Test listing runs for non-existent scenario."""
        response = client.get("/api/scenarios/99999/runs")
        
        assert response.status_code == 404
        data = response.get_json()
        assert "error" in data

    def test_run_simulation_failure(self, client, sample_scenario):
        """Test run simulation failure handling."""
        with patch('app.services.orchestration_service.OrchestrationService.run_simulation') as mock_run:
            mock_run.side_effect = Exception("Simulation failed")
            
            response = client.post(
                f"/api/scenarios/{sample_scenario.id}/runs",
                json={
                    "run_type": "monte_carlo",
                    "num_simulations": 1000
                }
            )
            
            assert response.status_code == 500
            data = response.get_json()
            assert "error" in data
            assert "message" in data
