"""
Simulation configuration model.

This module provides the Pydantic configuration model that holds all simulation
parameters and provider instances for the flexible simulation orchestrator.

The SimulationConfig serves as the central configuration object that:
1. Defines simulation parameters (time horizon, paths, initial conditions)
2. Holds provider instances that implement the protocol interfaces
3. Validates configuration consistency and provider compatibility
4. Provides clear error messages for misconfiguration
"""

from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.time_grid import UnitSystem


class SimulationConfig(BaseModel):
    """
    Comprehensive configuration for retirement planning simulations.

    This model holds all parameters and provider instances needed to run
    a complete simulation including portfolio evolution, cash flows,
    withdrawals, taxes, and rebalancing.

    Example:
        ```python
        config = SimulationConfig(
            unit_system=unit_system,
            num_paths=1000,
            initial_portfolio_balance=500000.0,
            target_asset_weights={"stocks": 0.7, "bonds": 0.3},
            returns_provider=returns_provider,
            income_provider=income_provider,
            expense_provider=expense_provider,
            withdrawal_policy=withdrawal_policy,
            # ... other providers
        )
        ```
    """

    # Core simulation parameters
    unit_system: UnitSystem = Field(
        ..., description="Time grid, inflation adjuster, and display settings"
    )

    num_paths: int = Field(
        ..., gt=0, le=100000, description="Number of Monte Carlo simulation paths"
    )

    # Initial portfolio conditions
    initial_portfolio_balance: float = Field(
        ..., ge=0, description="Starting portfolio value in base year dollars"
    )

    target_asset_weights: Dict[str, float] = Field(
        ..., description="Target allocation weights by asset name (must sum to 1.0)"
    )

    # Required providers (using Any to avoid Protocol typing issues)
    returns_provider: Any = Field(
        ..., description="Provider for market returns generation"
    )

    withdrawal_policy: Any = Field(
        ..., description="Policy for determining withdrawal amounts"
    )

    portfolio_engine: Any = Field(
        ..., description="Engine for portfolio state management and evolution"
    )

    rebalancing_strategy: Any = Field(
        ..., description="Strategy for portfolio rebalancing decisions"
    )

    # Optional providers (can be None for simplified simulations)
    income_provider: Optional[Any] = Field(
        default=None,
        description="Provider for annual income (salary, SS, investment, etc.)",
    )

    expense_provider: Optional[Any] = Field(
        default=None,
        description="Provider for annual expenses (baseline + lumpy events)",
    )

    liability_provider: Optional[Any] = Field(
        default=None, description="Provider for liability payments (mortgages, loans)"
    )

    tax_calculator: Optional[Any] = Field(
        default=None, description="Calculator for annual tax liability"
    )

    # Simulation control parameters
    enable_detailed_logging: bool = Field(
        default=False, description="Whether to enable detailed simulation logging"
    )

    random_seed: Optional[int] = Field(
        default=None, ge=0, description="Random seed for reproducible results"
    )

    # Advanced configuration
    cash_buffer_months: float = Field(
        default=0.0,
        ge=0,
        le=60,
        description="Months of expenses to hold as cash buffer",
    )

    emergency_withdrawal_penalty: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Penalty rate for emergency withdrawals (0-1)",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the simulation"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow protocol types
        validate_assignment=True,  # Validate on assignment
        extra="forbid",  # Forbid extra fields
    )

    @field_validator("target_asset_weights")
    @classmethod
    def validate_asset_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that asset weights are valid and sum to 1.0."""
        if not v:
            raise ValueError("target_asset_weights cannot be empty")

        # Check individual weights
        for asset_name, weight in v.items():
            if not isinstance(weight, (int, float)):
                raise ValueError(f"Weight for {asset_name} must be numeric")
            if weight < 0:
                raise ValueError(
                    f"Weight for {asset_name} cannot be negative: {weight}"
                )
            if weight > 1:
                raise ValueError(f"Weight for {asset_name} cannot exceed 1.0: {weight}")

        # Check sum
        total_weight = sum(v.values())
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point errors
            raise ValueError(
                f"Asset weights must sum to 1.0, got {total_weight:.6f}. "
                f"Weights: {v}"
            )

        return v

    @model_validator(mode="after")
    def validate_provider_compatibility(self) -> "SimulationConfig":
        """Validate that providers are compatible with configuration."""

        # Validate returns provider compatibility
        if hasattr(self.returns_provider, "get_asset_names"):
            provider_assets = set(self.returns_provider.get_asset_names())
            config_assets = set(self.target_asset_weights.keys())

            if provider_assets != config_assets:
                raise ValueError(
                    f"Asset mismatch between returns provider and target weights. "
                    f"Provider assets: {sorted(provider_assets)}, "
                    f"Config assets: {sorted(config_assets)}"
                )

        # Validate time horizon consistency
        len(self.unit_system.time_grid.get_years())

        # Check if providers support the simulation time horizon
        # (This would be extended as we add more provider implementations)

        return self

    @model_validator(mode="after")
    def validate_simulation_feasibility(self) -> "SimulationConfig":
        """Validate that the simulation configuration is feasible."""

        # Check for basic feasibility issues
        if self.initial_portfolio_balance == 0 and self.income_provider is None:
            raise ValueError(
                "Cannot run simulation with zero initial balance and no income source"
            )

        # Validate cash buffer
        if self.cash_buffer_months > 0 and self.expense_provider is None:
            raise ValueError("Cannot set cash buffer without expense provider")

        # Check time horizon reasonableness
        years = len(self.unit_system.time_grid.get_years())
        if years > 100:
            raise ValueError(
                f"Simulation time horizon too long: {years} years. "
                "Consider reducing to improve performance."
            )

        if self.num_paths > 50000 and years > 50:
            raise ValueError(
                f"Large simulation detected: {self.num_paths} paths × {years} years. "
                "Consider reducing for better performance."
            )

        return self

    def get_simulation_years(self) -> int:
        """Get the number of years in the simulation."""
        return len(self.unit_system.time_grid.get_years())

    def get_asset_names(self) -> List[str]:
        """Get the list of asset names in order."""
        return list(self.target_asset_weights.keys())

    def get_target_weights_array(self) -> NDArray[np.float64]:
        """Get target weights as a numpy array in asset order."""
        asset_names = self.get_asset_names()
        return np.array([self.target_asset_weights[name] for name in asset_names])

    def has_cash_flows(self) -> bool:
        """Check if the simulation includes cash flow providers."""
        return (
            self.income_provider is not None
            or self.expense_provider is not None
            or self.liability_provider is not None
        )

    def has_taxes(self) -> bool:
        """Check if the simulation includes tax calculations."""
        return self.tax_calculator is not None

    def get_provider_summary(self) -> Dict[str, str]:
        """Get a summary of configured providers."""
        providers = {}

        # Required providers
        providers["returns"] = type(self.returns_provider).__name__
        providers["withdrawal"] = type(self.withdrawal_policy).__name__
        providers["portfolio"] = type(self.portfolio_engine).__name__
        providers["rebalancing"] = type(self.rebalancing_strategy).__name__

        # Optional providers
        if self.income_provider:
            providers["income"] = type(self.income_provider).__name__
        if self.expense_provider:
            providers["expenses"] = type(self.expense_provider).__name__
        if self.liability_provider:
            providers["liabilities"] = type(self.liability_provider).__name__
        if self.tax_calculator:
            providers["taxes"] = type(self.tax_calculator).__name__

        return providers

    def validate_for_simulation(self) -> None:
        """
        Perform final validation before simulation starts.

        This method performs additional runtime validation that may
        require provider method calls or expensive checks.

        Raises:
            ValueError: If configuration is invalid for simulation
        """

        # Validate returns provider can generate the required data
        try:
            # Check that provider has required methods
            if not hasattr(self.returns_provider, "generate_returns"):
                raise ValueError("Returns provider missing 'generate_returns' method")

            # Test with minimal parameters
            test_returns = self.returns_provider.generate_returns(
                years=1, num_paths=1, seed=42
            )
            expected_assets = len(self.target_asset_weights)
            if test_returns.shape != (expected_assets, 1, 1):
                raise ValueError(
                    f"Returns provider shape mismatch. Expected ({expected_assets}, 1, 1), "
                    f"got {test_returns.shape}"
                )
        except Exception as e:
            raise ValueError(f"Returns provider validation failed: {str(e)}")

        # Validate portfolio engine initialization
        try:
            # This would be called during actual simulation setup
            # For now, just check that the method exists
            if not hasattr(self.portfolio_engine, "initialize"):
                raise ValueError("Portfolio engine missing initialize method")
        except Exception as e:
            raise ValueError(f"Portfolio engine validation failed: {str(e)}")

    def create_simulation_summary(self) -> Dict[str, Any]:
        """Create a summary of the simulation configuration."""
        return {
            "simulation_years": self.get_simulation_years(),
            "num_paths": self.num_paths,
            "initial_balance": self.initial_portfolio_balance,
            "asset_allocation": self.target_asset_weights,
            "has_cash_flows": self.has_cash_flows(),
            "has_taxes": self.has_taxes(),
            "providers": self.get_provider_summary(),
            "random_seed": self.random_seed,
            "time_period": {
                "start_year": self.unit_system.time_grid.start_year,
                "end_year": self.unit_system.time_grid.end_year,
                "base_year": self.unit_system.time_grid.base_year,
            },
            "inflation_rate": self.unit_system.inflation_adjuster.inflation_rate,
            "display_mode": self.unit_system.display_mode,
        }
