"""
Orchestration service for coordinating retirement planning simulation runs.

This service coordinates all the simulation engines to run complete retirement
scenarios, handling the flow from scenario configuration through to final results.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from sqlalchemy.orm import Session

from app.database.base import get_db
from app.database.models import Run
from app.models.scenario import Scenario

logger = logging.getLogger(__name__)


class OrchestrationService:
    """Service for orchestrating retirement planning simulation runs."""

    def __init__(self) -> None:
        """Initialize the orchestration service."""
        self.logger = logging.getLogger(__name__)

    def run_simulation(
        self,
        run_id: int,
        scenario_config: Dict[str, Any],
        run_type: str = "monte_carlo",
        num_simulations: int = 1000,
    ) -> Dict[str, Any]:
        """Run a complete retirement planning simulation.

        Args:
            run_id: Database ID of the run
            scenario_config: Scenario configuration dictionary
            run_type: Type of simulation (monte_carlo, deterministic, historical)
            num_simulations: Number of simulation paths for Monte Carlo

        Returns:
            Dictionary containing simulation results

        Raises:
            Exception: If simulation fails
        """
        try:
            self.logger.info(f"Starting simulation run {run_id} with type {run_type}")

            # Update run status to running
            self._update_run_status(run_id, "running", started_at=datetime.utcnow())

            # Parse scenario configuration
            scenario = scenario_config  # Use dict directly for MVP

            # Run simulation based on type
            if run_type == "deterministic":
                results = self._run_deterministic_simulation(scenario)
            elif run_type == "monte_carlo":
                results = self._run_monte_carlo_simulation(scenario, num_simulations)
            else:
                raise ValueError(f"Unsupported run type: {run_type}")

            # Store results in database
            self._store_results(run_id, results)

            self.logger.info(f"Completed simulation run {run_id}")
            return results

        except Exception as e:
            self.logger.error(f"Simulation run {run_id} failed: {str(e)}")
            self._update_run_status(run_id, "failed", error_message=str(e))
            raise

    def _run_deterministic_simulation(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run deterministic simulation.

        Args:
            scenario: Scenario configuration

        Returns:
            Dictionary containing deterministic results
        """
        # Simplified deterministic simulation for MVP
        years = 30  # Default 30-year simulation

        # Basic portfolio evolution with fixed returns
        initial_balance = 0.0
        accounts = scenario.get("accounts", {})
        for account_list in [
            accounts.get("taxable", []),
            accounts.get("traditional_401k", []),
            accounts.get("roth_401k", []),
            accounts.get("traditional_ira", []),
            accounts.get("roth_ira", []),
            accounts.get("hsa", []),
        ]:
            if isinstance(account_list, list):
                initial_balance += sum(account.get("balance", 0) for account in account_list)
        
        cash = accounts.get("cash", {})
        initial_balance += cash.get("checking", 0) + cash.get("savings", 0)

        # Calculate total annual expenses
        expenses = scenario.get("expenses", {})
        housing = expenses.get("housing", {})
        transportation = expenses.get("transportation", {})
        healthcare = expenses.get("healthcare", {})
        
        annual_withdrawal = (
            housing.get("mortgage_payment", 0) * 12
            + housing.get("property_tax", 0)
            + housing.get("home_insurance", 0)
            + housing.get("maintenance", 0)
            + transportation.get("auto_payment", 0) * 12
            + transportation.get("auto_insurance", 0)
            + transportation.get("maintenance", 0)
            + healthcare.get("insurance", 0) * 12
            + healthcare.get("out_of_pocket", 0)
            + expenses.get("food", 0) * 12
            + expenses.get("entertainment", 0) * 12
            + expenses.get("travel", 0)
            + expenses.get("education", 0)
            + expenses.get("other", 0) * 12
        )
        annual_return = 0.07  # 7% annual return

        # Calculate portfolio balance over time
        portfolio_evolution = []
        current_balance = float(initial_balance)

        for year in range(years):
            # Apply return
            current_balance *= 1 + annual_return
            # Apply withdrawal
            current_balance -= annual_withdrawal
            portfolio_evolution.append(float(max(0, current_balance)))

        # Calculate success metrics
        success_rate = 1.0 if portfolio_evolution[-1] > 0 else 0.0
        terminal_wealth = portfolio_evolution[-1]

        return {
            "simulation_type": "deterministic",
            "years": years,
            "initial_balance": initial_balance,
            "annual_return": annual_return,
            "annual_withdrawal": annual_withdrawal,
            "portfolio_evolution": portfolio_evolution,
            "success_rate": success_rate,
            "terminal_wealth": terminal_wealth,
            "final_balance": portfolio_evolution[-1],
        }

    def _run_monte_carlo_simulation(
        self, scenario: Dict[str, Any], num_simulations: int
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation.

        Args:
            scenario: Scenario configuration
            num_simulations: Number of simulation paths

        Returns:
            Dictionary containing Monte Carlo results
        """
        # Simplified Monte Carlo simulation for MVP
        years = 30  # Default 30-year simulation
        initial_balance = 0.0
        accounts = scenario.get("accounts", {})
        for account_list in [
            accounts.get("taxable", []),
            accounts.get("traditional_401k", []),
            accounts.get("roth_401k", []),
            accounts.get("traditional_ira", []),
            accounts.get("roth_ira", []),
            accounts.get("hsa", []),
        ]:
            if isinstance(account_list, list):
                initial_balance += sum(account.get("balance", 0) for account in account_list)
        
        cash = accounts.get("cash", {})
        initial_balance += cash.get("checking", 0) + cash.get("savings", 0)

        # Calculate total annual expenses
        expenses = scenario.get("expenses", {})
        housing = expenses.get("housing", {})
        transportation = expenses.get("transportation", {})
        healthcare = expenses.get("healthcare", {})
        
        annual_withdrawal = (
            housing.get("mortgage_payment", 0) * 12
            + housing.get("property_tax", 0)
            + housing.get("home_insurance", 0)
            + housing.get("maintenance", 0)
            + transportation.get("auto_payment", 0) * 12
            + transportation.get("auto_insurance", 0)
            + transportation.get("maintenance", 0)
            + healthcare.get("insurance", 0) * 12
            + healthcare.get("out_of_pocket", 0)
            + expenses.get("food", 0) * 12
            + expenses.get("entertainment", 0) * 12
            + expenses.get("travel", 0)
            + expenses.get("education", 0)
            + expenses.get("other", 0) * 12
        )

        # Monte Carlo parameters
        mean_return = 0.07
        std_return = 0.18

        # Run simulations
        final_balances = []
        portfolio_evolutions = []

        for sim in range(num_simulations):
            current_balance = float(initial_balance)
            portfolio_path = [current_balance]

            for year in range(years):
                # Generate random return
                annual_return = np.random.normal(mean_return, std_return)
                # Apply return
                current_balance *= 1 + annual_return
                # Apply withdrawal
                current_balance -= annual_withdrawal
                current_balance = float(max(0, current_balance))
                portfolio_path.append(current_balance)

            final_balances.append(current_balance)
            portfolio_evolutions.append(portfolio_path)

        # Calculate success metrics
        success_count = sum(1 for balance in final_balances if balance > 0)
        success_rate = success_count / num_simulations

        # Calculate percentiles
        final_balances_array = np.array(final_balances)
        percentiles = {
            "p5": np.percentile(final_balances_array, 5),
            "p25": np.percentile(final_balances_array, 25),
            "p50": np.percentile(final_balances_array, 50),
            "p75": np.percentile(final_balances_array, 75),
            "p95": np.percentile(final_balances_array, 95),
        }

        return {
            "simulation_type": "monte_carlo",
            "years": years,
            "num_simulations": num_simulations,
            "initial_balance": initial_balance,
            "annual_withdrawal": annual_withdrawal,
            "mean_return": mean_return,
            "std_return": std_return,
            "success_rate": success_rate,
            "terminal_wealth_percentiles": percentiles,
            "portfolio_evolution_sample": portfolio_evolutions[
                0
            ],  # First path as sample
            "final_balances_sample": final_balances[:10],  # First 10 as sample
        }

    def _update_run_status(
        self,
        run_id: int,
        status: str,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update run status in database.

        Args:
            run_id: Database ID of the run
            status: New status
            started_at: Start time (if starting)
            completed_at: Completion time (if completing)
            error_message: Error message (if failed)
        """
        db: Session = next(get_db())

        run = db.query(Run).filter(Run.id == run_id).first()
        if run:
            run.status = status  # type: ignore
            if started_at:
                run.started_at = started_at  # type: ignore
            if completed_at:
                run.completed_at = completed_at  # type: ignore
            if error_message:
                run.error_message = error_message  # type: ignore
            db.commit()

    def _store_results(self, run_id: int, results: Dict[str, Any]) -> None:
        """Store simulation results in database.

        Args:
            run_id: Database ID of the run
            results: Simulation results to store
        """
        db: Session = next(get_db())

        run = db.query(Run).filter(Run.id == run_id).first()
        if run:
            run.results = results  # type: ignore
            run.completed_at = datetime.utcnow()  # type: ignore
            db.commit()
