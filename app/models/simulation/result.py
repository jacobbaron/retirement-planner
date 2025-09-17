"""
Simulation result model.

This module provides the comprehensive result model that captures all simulation
outputs for analysis, reporting, and further processing.

The SimulationResult serves as the unified output format that:
1. Contains all simulation data (portfolio, income, expenses, taxes, etc.)
2. Provides helper methods for common queries and analysis
3. Supports serialization for storage and export
4. Enables easy integration with visualization and reporting tools
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class SimulationResult(BaseModel):
    """
    Comprehensive result model for retirement planning simulations.

    This model captures all outputs from the simulation orchestrator including
    portfolio evolution, cash flows, taxes, rebalancing events, and metadata.

    All arrays follow the convention:
    - portfolio_balances: (years, paths)
    - asset_balances: (assets, years, paths)
    - Other flows: (years, paths)

    Example:
        ```python
        result = SimulationResult(
            portfolio_balances=portfolio_data,
            asset_balances=asset_data,
            withdrawals=withdrawal_data,
            # ... other arrays
            asset_names=["stocks", "bonds"],
            simulation_metadata=metadata
        )

        # Analysis
        success_rate = result.calculate_success_rate(threshold=0)
        final_balances = result.get_final_balances()
        ```
    """

    # Core portfolio data
    portfolio_balances: NDArray[np.float64] = Field(
        ..., description="Portfolio balance over time (years × paths)"
    )

    asset_balances: NDArray[np.float64] = Field(
        ..., description="Individual asset balances (assets × years × paths)"
    )

    # Cash flow data
    withdrawals: NDArray[np.float64] = Field(
        ..., description="Annual withdrawal amounts (years × paths)"
    )

    incomes: Optional[NDArray[np.float64]] = Field(
        default=None, description="Annual income amounts (years × paths)"
    )

    expenses: Optional[NDArray[np.float64]] = Field(
        default=None, description="Annual expense amounts (years × paths)"
    )

    liability_payments: Optional[NDArray[np.float64]] = Field(
        default=None, description="Annual liability payments (years × paths)"
    )

    taxes: Optional[NDArray[np.float64]] = Field(
        default=None, description="Annual tax liability (years × paths)"
    )

    # Portfolio management data
    rebalancing_events: NDArray[np.bool_] = Field(
        ..., description="Rebalancing occurred (years × paths)"
    )

    transaction_costs: NDArray[np.float64] = Field(
        ..., description="Transaction costs from rebalancing (years × paths)"
    )

    portfolio_returns: NDArray[np.float64] = Field(
        ..., description="Annual portfolio returns (years × paths)"
    )

    # Asset information
    asset_names: List[str] = Field(..., description="Names of assets in order")

    target_weights: NDArray[np.float64] = Field(
        ..., description="Target portfolio weights"
    )

    # Simulation metadata
    simulation_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the simulation configuration and execution",
    )

    # Optional detailed ledger for debugging/exports
    detailed_ledger: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Detailed transaction ledger for debugging"
    )

    # Execution metadata
    execution_time_seconds: Optional[float] = Field(
        default=None, description="Time taken to run the simulation"
    )

    created_at: datetime = Field(
        default_factory=datetime.now, description="When the simulation was completed"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow numpy arrays
        validate_assignment=True,
        extra="forbid",
    )

    @field_validator("portfolio_balances", "withdrawals", "portfolio_returns")
    @classmethod
    def validate_required_arrays_shape(cls, v: NDArray) -> NDArray:
        """Validate that required arrays have correct 2D shape (years × paths)."""
        if v.ndim != 2:
            raise ValueError(
                f"Array must be 2-dimensional (years × paths), got {v.ndim}D"
            )
        return v

    @field_validator("asset_balances")
    @classmethod
    def validate_asset_balances_shape(cls, v: NDArray) -> NDArray:
        """Validate that asset_balances has correct 3D shape (assets × years × paths)."""
        if v.ndim != 3:
            raise ValueError(
                f"asset_balances must be 3-dimensional (assets × years × paths), got {v.ndim}D"
            )
        return v

    @field_validator("rebalancing_events")
    @classmethod
    def validate_rebalancing_events_shape(cls, v: NDArray) -> NDArray:
        """Validate that rebalancing_events has correct shape and dtype."""
        if v.ndim != 2:
            raise ValueError(
                f"rebalancing_events must be 2-dimensional (years × paths), got {v.ndim}D"
            )
        if not np.issubdtype(v.dtype, np.bool_):
            raise ValueError(f"rebalancing_events must be boolean array, got {v.dtype}")
        return v

    @field_validator(
        "incomes", "expenses", "liability_payments", "taxes", "transaction_costs"
    )
    @classmethod
    def validate_optional_arrays_shape(cls, v: Optional[NDArray]) -> Optional[NDArray]:
        """Validate that optional arrays have correct 2D shape if provided."""
        if v is not None and v.ndim != 2:
            raise ValueError(
                f"Array must be 2-dimensional (years × paths), got {v.ndim}D"
            )
        return v

    @property
    def shape(self) -> Tuple[int, int]:
        """Get the simulation shape (years, paths)."""
        shape = self.portfolio_balances.shape
        return (int(shape[0]), int(shape[1]))

    @property
    def years(self) -> int:
        """Get the number of simulation years."""
        return int(self.portfolio_balances.shape[0])

    @property
    def paths(self) -> int:
        """Get the number of simulation paths."""
        return int(self.portfolio_balances.shape[1])

    @property
    def num_assets(self) -> int:
        """Get the number of assets."""
        return len(self.asset_names)

    def get_final_balances(self) -> NDArray[np.float64]:
        """Get final portfolio balances for all paths."""
        return self.portfolio_balances[-1, :]

    def get_final_asset_balances(self) -> NDArray[np.float64]:
        """Get final asset balances (assets × paths)."""
        return self.asset_balances[:, -1, :]

    def calculate_success_rate(
        self, threshold: float = 0, success_definition: str = "final_balance"
    ) -> float:
        """
        Calculate simulation success rate.

        Args:
            threshold: Minimum value for success (default: 0 = not depleted)
            success_definition: How to define success
                - "final_balance": Final portfolio balance >= threshold
                - "never_depleted": Portfolio never goes below threshold
                - "withdrawal_sustained": All withdrawals were satisfied

        Returns:
            Success rate as a fraction (0.0 to 1.0)
        """
        if success_definition == "final_balance":
            final_balances = self.get_final_balances()
            successful_paths = np.sum(final_balances >= threshold)
        elif success_definition == "never_depleted":
            min_balances = np.min(self.portfolio_balances, axis=0)
            successful_paths = np.sum(min_balances >= threshold)
        elif success_definition == "withdrawal_sustained":
            # Check if portfolio could support all withdrawals
            # (This is a simplified check - real implementation would be more complex)
            min_balances = np.min(self.portfolio_balances, axis=0)
            successful_paths = np.sum(min_balances >= 0)
        else:
            raise ValueError(f"Unknown success definition: {success_definition}")

        return float(successful_paths / self.paths)

    def get_percentiles(
        self, percentiles: List[float], data_type: str = "final_balance"
    ) -> Dict[float, float]:
        """
        Get percentiles for various data types.

        Args:
            percentiles: List of percentiles to calculate (e.g., [10, 25, 50, 75, 90])
            data_type: Type of data to analyze
                - "final_balance": Final portfolio balances
                - "min_balance": Minimum balance during simulation
                - "max_balance": Maximum balance during simulation
                - "total_withdrawals": Sum of all withdrawals
                - "total_returns": Cumulative portfolio returns

        Returns:
            Dictionary mapping percentiles to values
        """
        if data_type == "final_balance":
            data = self.get_final_balances()
        elif data_type == "min_balance":
            data = np.min(self.portfolio_balances, axis=0)
        elif data_type == "max_balance":
            data = np.max(self.portfolio_balances, axis=0)
        elif data_type == "total_withdrawals":
            data = np.sum(self.withdrawals, axis=0)
        elif data_type == "total_returns":
            # Calculate cumulative returns for each path
            cumulative_returns = np.prod(1 + self.portfolio_returns, axis=0) - 1
            data = cumulative_returns
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        # Convert to float array for percentile calculation
        float_data = np.asarray(data, dtype=float)
        return {p: float(np.percentile(float_data, p)) for p in percentiles}

    def get_balance_statistics(self) -> Dict[str, float]:
        """Get comprehensive balance statistics."""
        final_balances = self.get_final_balances()
        min_balances = np.min(self.portfolio_balances, axis=0)
        max_balances = np.max(self.portfolio_balances, axis=0)

        return {
            "final_balance_mean": float(np.mean(final_balances)),
            "final_balance_median": float(np.median(final_balances)),
            "final_balance_std": float(np.std(final_balances)),
            "min_balance_mean": float(np.mean(min_balances)),
            "min_balance_median": float(np.median(min_balances)),
            "max_balance_mean": float(np.mean(max_balances)),
            "max_balance_median": float(np.median(max_balances)),
            "depletion_rate": 1.0 - self.calculate_success_rate(threshold=0),
        }

    def get_withdrawal_statistics(self) -> Dict[str, float]:
        """Get comprehensive withdrawal statistics."""
        total_withdrawals = np.sum(self.withdrawals, axis=0)
        annual_avg_withdrawals = np.mean(self.withdrawals, axis=0)

        return {
            "total_withdrawal_mean": float(np.mean(total_withdrawals)),
            "total_withdrawal_median": float(np.median(total_withdrawals)),
            "annual_withdrawal_mean": float(np.mean(annual_avg_withdrawals)),
            "annual_withdrawal_std": float(np.std(self.withdrawals)),
            "max_annual_withdrawal": float(np.max(self.withdrawals)),
            "min_annual_withdrawal": float(np.min(self.withdrawals)),
        }

    def get_rebalancing_statistics(self) -> Dict[str, Any]:
        """Get rebalancing frequency and cost statistics."""
        total_rebalancing_events = np.sum(self.rebalancing_events)
        total_transaction_costs = np.sum(self.transaction_costs)

        return {
            "total_rebalancing_events": int(total_rebalancing_events),
            "rebalancing_frequency": float(
                total_rebalancing_events / (self.years * self.paths)
            ),
            "total_transaction_costs": float(total_transaction_costs),
            "avg_transaction_cost_per_event": float(
                total_transaction_costs / max(int(total_rebalancing_events), 1)
            ),
            "transaction_cost_as_pct_of_portfolio": float(
                total_transaction_costs / np.sum(self.portfolio_balances) * 100
            ),
        }

    def get_cash_flow_statistics(self) -> Dict[str, Any]:
        """Get cash flow statistics (income, expenses, taxes if available)."""
        stats = {}

        if self.incomes is not None:
            total_incomes = np.sum(self.incomes, axis=0)
            stats.update(
                {
                    "total_income_mean": float(np.mean(total_incomes)),
                    "total_income_median": float(np.median(total_incomes)),
                    "annual_income_mean": float(np.mean(self.incomes)),
                }
            )

        if self.expenses is not None:
            total_expenses = np.sum(self.expenses, axis=0)
            stats.update(
                {
                    "total_expenses_mean": float(np.mean(total_expenses)),
                    "total_expenses_median": float(np.median(total_expenses)),
                    "annual_expenses_mean": float(np.mean(self.expenses)),
                }
            )

        if self.liability_payments is not None:
            total_liabilities = np.sum(self.liability_payments, axis=0)
            stats.update(
                {
                    "total_liability_payments": float(np.mean(total_liabilities)),
                    "annual_liability_payments": float(
                        np.mean(self.liability_payments)
                    ),
                }
            )

        if self.taxes is not None:
            total_taxes = np.sum(self.taxes, axis=0)
            stats.update(
                {
                    "total_taxes_mean": float(np.mean(total_taxes)),
                    "annual_taxes_mean": float(np.mean(self.taxes)),
                }
            )

        return stats

    def get_return_statistics(self) -> Dict[str, float]:
        """Get portfolio return statistics."""
        # Calculate annualized returns for each path
        final_balances = self.get_final_balances()
        initial_balance = self.portfolio_balances[
            0, 0
        ]  # Assume same initial balance for all paths

        annualized_returns = (final_balances / initial_balance) ** (1 / self.years) - 1

        # Annual return statistics
        annual_returns_flat = self.portfolio_returns.flatten()

        return {
            "annualized_return_mean": float(np.mean(annualized_returns)),
            "annualized_return_median": float(np.median(annualized_returns)),
            "annualized_return_std": float(np.std(annualized_returns)),
            "annual_return_mean": float(np.mean(annual_returns_flat)),
            "annual_return_std": float(np.std(annual_returns_flat)),
            "best_annual_return": float(np.max(annual_returns_flat)),
            "worst_annual_return": float(np.min(annual_returns_flat)),
        }

    def create_summary_report(self) -> Dict[str, Any]:
        """Create a comprehensive summary report of the simulation results."""
        return {
            "simulation_info": {
                "years": self.years,
                "paths": self.paths,
                "assets": self.asset_names,
                "target_weights": self.target_weights.tolist(),
                "created_at": self.created_at.isoformat(),
                "execution_time_seconds": self.execution_time_seconds,
            },
            "success_metrics": {
                "success_rate_not_depleted": self.calculate_success_rate(threshold=0),
                "success_rate_positive_balance": self.calculate_success_rate(
                    threshold=1
                ),
            },
            "balance_statistics": self.get_balance_statistics(),
            "withdrawal_statistics": self.get_withdrawal_statistics(),
            "return_statistics": self.get_return_statistics(),
            "rebalancing_statistics": self.get_rebalancing_statistics(),
            "cash_flow_statistics": self.get_cash_flow_statistics(),
            "percentiles": {
                "final_balance": self.get_percentiles(
                    [10, 25, 50, 75, 90], "final_balance"
                ),
                "min_balance": self.get_percentiles(
                    [10, 25, 50, 75, 90], "min_balance"
                ),
            },
            "metadata": self.simulation_metadata,
        }

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """
        Convert result to dictionary for serialization.

        Args:
            include_arrays: Whether to include large numpy arrays in output

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        result = {
            "simulation_info": {
                "years": self.years,
                "paths": self.paths,
                "num_assets": self.num_assets,
                "asset_names": self.asset_names,
                "target_weights": self.target_weights.tolist(),
                "created_at": self.created_at.isoformat(),
                "execution_time_seconds": self.execution_time_seconds,
            },
            "summary_statistics": self.create_summary_report(),
        }

        if include_arrays:
            result["arrays"] = {
                "portfolio_balances": self.portfolio_balances.tolist(),
                "asset_balances": self.asset_balances.tolist(),
                "withdrawals": self.withdrawals.tolist(),
                "rebalancing_events": self.rebalancing_events.tolist(),
                "transaction_costs": self.transaction_costs.tolist(),
                "portfolio_returns": self.portfolio_returns.tolist(),
            }

            # Add optional arrays if present
            if self.incomes is not None:
                result["arrays"]["incomes"] = self.incomes.tolist()
            if self.expenses is not None:
                result["arrays"]["expenses"] = self.expenses.tolist()
            if self.liability_payments is not None:
                result["arrays"][
                    "liability_payments"
                ] = self.liability_payments.tolist()
            if self.taxes is not None:
                result["arrays"]["taxes"] = self.taxes.tolist()

        return result

    def to_json(self, include_arrays: bool = False, indent: int = 2) -> str:
        """
        Convert result to JSON string.

        Args:
            include_arrays: Whether to include large numpy arrays
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(include_arrays=include_arrays), indent=indent)

    def save_to_file(self, filepath: str, include_arrays: bool = True) -> None:
        """
        Save result to JSON file.

        Args:
            filepath: Path to save the file
            include_arrays: Whether to include numpy arrays in the file
        """
        with open(filepath, "w") as f:
            f.write(self.to_json(include_arrays=include_arrays))

    def get_path_data(self, path_index: int) -> Dict[str, Any]:
        """
        Get all data for a specific simulation path.

        Args:
            path_index: Index of the path to extract

        Returns:
            Dictionary containing all arrays for the specified path
        """
        if path_index >= self.paths:
            raise ValueError(f"Path index {path_index} >= number of paths {self.paths}")

        data = {
            "portfolio_balances": self.portfolio_balances[:, path_index],
            "asset_balances": self.asset_balances[:, :, path_index],
            "withdrawals": self.withdrawals[:, path_index],
            "rebalancing_events": self.rebalancing_events[:, path_index],
            "transaction_costs": self.transaction_costs[:, path_index],
            "portfolio_returns": self.portfolio_returns[:, path_index],
        }

        # Add optional arrays if present
        if self.incomes is not None:
            data["incomes"] = self.incomes[:, path_index]
        if self.expenses is not None:
            data["expenses"] = self.expenses[:, path_index]
        if self.liability_payments is not None:
            data["liability_payments"] = self.liability_payments[:, path_index]
        if self.taxes is not None:
            data["taxes"] = self.taxes[:, path_index]

        return data

    def compare_with(self, other: "SimulationResult") -> Dict[str, Any]:
        """
        Compare this result with another simulation result.

        Args:
            other: Another SimulationResult to compare with

        Returns:
            Dictionary containing comparison metrics
        """
        if self.shape != other.shape:
            raise ValueError(
                f"Cannot compare results with different shapes: {self.shape} vs {other.shape}"
            )

        comparison = {
            "success_rate_difference": (
                self.calculate_success_rate() - other.calculate_success_rate()
            ),
            "final_balance_difference": {
                "mean": float(
                    np.mean(self.get_final_balances() - other.get_final_balances())
                ),
                "median": float(
                    np.median(self.get_final_balances() - other.get_final_balances())
                ),
            },
            "withdrawal_difference": {
                "mean": float(
                    np.mean(
                        np.sum(self.withdrawals, axis=0)
                        - np.sum(other.withdrawals, axis=0)
                    )
                ),
            },
        }

        return comparison
