"""
Account balance evolution engine for retirement planning.

This module implements the core cashflow engine that tracks money flows through
different account types over time, including contributions, withdrawals, growth,
and portfolio rebalancing.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from .time_grid import TimeGrid, InflationAdjuster
from .scenario import (
    AssetAllocation,
    TaxableAccount,
    Traditional401kAccount,
    Roth401kAccount,
    TraditionalIRAAccount,
    RothIRAAccount,
    HSAAccount,
    College529Account,
)


class Transaction(BaseModel):
    """A single financial transaction affecting an account."""

    year: int = Field(..., description="Year of the transaction")
    account_name: str = Field(..., description="Name of the account")
    account_type: str = Field(
        ..., description="Type of account (taxable, traditional_401k, etc.)"
    )
    transaction_type: Literal["contribution", "withdrawal", "growth", "rebalance"] = (
        Field(..., description="Type of transaction")
    )
    amount: float = Field(..., description="Transaction amount")
    description: str = Field(..., description="Human-readable description")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional transaction metadata"
    )


class AccountBalance(BaseModel):
    """Account balance at a specific point in time."""

    year: int = Field(..., description="Year of the balance")
    account_name: str = Field(..., description="Name of the account")
    account_type: str = Field(..., description="Type of account")
    balance: float = Field(..., ge=0, description="Account balance")
    asset_allocation: AssetAllocation = Field(
        ..., description="Current asset allocation"
    )
    cost_basis: Optional[float] = Field(
        None, ge=0, description="Cost basis for taxable accounts"
    )


class AccountEvolution(BaseModel):
    """Evolution of a single account over time."""

    account_name: str = Field(..., description="Name of the account")
    account_type: str = Field(..., description="Type of account")
    initial_balance: float = Field(..., ge=0, description="Starting balance")
    asset_allocation: AssetAllocation = Field(
        ..., description="Target asset allocation"
    )
    balances: List[AccountBalance] = Field(
        default_factory=list, description="Year-by-year balances"
    )
    transactions: List[Transaction] = Field(
        default_factory=list, description="All transactions"
    )

    def get_balance(self, year: int) -> float:
        """Get account balance for a specific year."""
        for balance in self.balances:
            if balance.year == year:
                return balance.balance
        return 0.0

    def get_latest_balance(self) -> float:
        """Get the most recent account balance."""
        if not self.balances:
            return self.initial_balance
        return max(self.balances, key=lambda b: b.year).balance


class AccountEvolutionEngine(BaseModel):
    """Engine for managing account balance evolution over time."""

    time_grid: TimeGrid = Field(..., description="Time grid for calculations")
    inflation_adjuster: InflationAdjuster = Field(
        ..., description="Inflation adjustment utilities"
    )
    accounts: List[AccountEvolution] = Field(
        default_factory=list, description="All accounts being tracked"
    )
    market_returns: Dict[str, float] = Field(
        ..., description="Expected annual returns by asset class"
    )

    def add_account(self, account_evolution: AccountEvolution) -> None:
        """Add an account to track."""
        self.accounts.append(account_evolution)

    def get_account(self, account_name: str) -> Optional[AccountEvolution]:
        """Get an account by name."""
        for account in self.accounts:
            if account.account_name == account_name:
                return account
        return None

    def add_contribution(
        self,
        account_name: str,
        year: int,
        amount: float,
        description: str = "Contribution",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a contribution to an account."""
        account = self.get_account(account_name)
        if not account:
            raise ValueError(f"Account {account_name} not found")

        if amount <= 0:
            raise ValueError("Contribution amount must be positive")

        # Add transaction
        transaction = Transaction(
            year=year,
            account_name=account_name,
            account_type=account.account_type,
            transaction_type="contribution",
            amount=amount,
            description=description,
            metadata=metadata or {},
        )
        account.transactions.append(transaction)

        # Update balance
        current_balance = account.get_balance(year)
        new_balance = current_balance + amount

        # Update or create balance record
        self._update_account_balance(account, year, new_balance)

    def add_withdrawal(
        self,
        account_name: str,
        year: int,
        amount: float,
        description: str = "Withdrawal",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a withdrawal from an account."""
        account = self.get_account(account_name)
        if not account:
            raise ValueError(f"Account {account_name} not found")

        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")

        current_balance = account.get_balance(year)
        if amount > current_balance:
            raise ValueError(f"Insufficient balance: {amount} > {current_balance}")

        # Add transaction
        transaction = Transaction(
            year=year,
            account_name=account_name,
            account_type=account.account_type,
            transaction_type="withdrawal",
            amount=-amount,  # Negative for withdrawals
            description=description,
            metadata=metadata or {},
        )
        account.transactions.append(transaction)

        # Update balance
        new_balance = current_balance - amount
        self._update_account_balance(account, year, new_balance)

    def apply_growth(self, year: int) -> None:
        """Apply growth to all accounts for a given year."""
        for account in self.accounts:
            # Get the current balance for this year (including any contributions/withdrawals)
            current_balance = account.get_balance(year)

            # If no balance record exists for this year, use the previous year's balance or initial balance
            if current_balance == 0.0:
                if year == self.time_grid.start_year:
                    current_balance = account.initial_balance
                else:
                    # Look for the most recent balance
                    for prev_year in range(year - 1, self.time_grid.start_year - 1, -1):
                        prev_balance = account.get_balance(prev_year)
                        if prev_balance > 0:
                            current_balance = prev_balance
                            break
                    else:
                        # No previous balance found, use initial balance
                        current_balance = account.initial_balance

            if current_balance <= 0:
                continue

            # Calculate weighted return based on asset allocation
            total_return = 0.0
            for (
                asset_class,
                allocation,
            ) in account.asset_allocation.model_dump().items():
                if allocation > 0 and asset_class in self.market_returns:
                    total_return += allocation * self.market_returns[asset_class]

            # Apply growth
            growth_amount = current_balance * total_return
            new_balance = current_balance + growth_amount

            # Add growth transaction
            transaction = Transaction(
                year=year,
                account_name=account.account_name,
                account_type=account.account_type,
                transaction_type="growth",
                amount=growth_amount,
                description=f"Growth at {total_return:.1%}",
                metadata={"return_rate": total_return},
            )
            account.transactions.append(transaction)

            # Update balance
            self._update_account_balance(account, year, new_balance)

    def rebalance_account(self, account_name: str, year: int) -> None:
        """Rebalance an account to its target allocation."""
        account = self.get_account(account_name)
        if not account:
            raise ValueError(f"Account {account_name} not found")

        current_balance = account.get_balance(year)
        if current_balance <= 0:
            return

        # For now, we'll just log the rebalancing
        # In a full implementation, this would involve selling/buying assets
        transaction = Transaction(
            year=year,
            account_name=account_name,
            account_type=account.account_type,
            transaction_type="rebalance",
            amount=0.0,  # No net cash flow for rebalancing
            description="Portfolio rebalancing",
            metadata={"target_allocation": account.asset_allocation.model_dump()},
        )
        account.transactions.append(transaction)

    def _update_account_balance(
        self, account: AccountEvolution, year: int, new_balance: float
    ) -> None:
        """Update or create an account balance record."""
        # Find existing balance record
        for i, balance in enumerate(account.balances):
            if balance.year == year:
                account.balances[i].balance = new_balance
                return

        # Create new balance record
        new_balance_record = AccountBalance(
            year=year,
            account_name=account.account_name,
            account_type=account.account_type,
            balance=new_balance,
            asset_allocation=account.asset_allocation,
            cost_basis=None,  # TODO: Track cost basis for taxable accounts
        )
        account.balances.append(new_balance_record)

    def get_total_balance(self, year: int) -> float:
        """Get total balance across all accounts for a given year."""
        total = 0.0
        for account in self.accounts:
            total += account.get_balance(year)
        return total

    def get_account_balances(self, year: int) -> Dict[str, float]:
        """Get balances for all accounts in a given year."""
        balances = {}
        for account in self.accounts:
            balances[account.account_name] = account.get_balance(year)
        return balances

    def get_transaction_summary(self, year: int) -> Dict[str, List[Transaction]]:
        """Get all transactions for a given year, grouped by account."""
        summary = {}
        for account in self.accounts:
            year_transactions = [t for t in account.transactions if t.year == year]
            if year_transactions:
                summary[account.account_name] = year_transactions
        return summary

    def evolve_all_accounts(self) -> None:
        """Evolve all accounts through the entire time grid."""
        for year in self.time_grid.get_years():
            # Apply growth first (on beginning-of-year balances)
            self.apply_growth(year)

            # Then apply any contributions/withdrawals for the year
            # (These would come from external sources like income/expense engines)

            # Rebalance at end of year
            for account in self.accounts:
                self.rebalance_account(account.account_name, year)


def create_account_evolution_from_scenario_account(
    account: Union[
        TaxableAccount,
        Traditional401kAccount,
        Roth401kAccount,
        TraditionalIRAAccount,
        RothIRAAccount,
        HSAAccount,
        College529Account,
    ],
    account_type: str,
) -> AccountEvolution:
    """Create an AccountEvolution from a scenario account."""
    return AccountEvolution(
        account_name=account.name,
        account_type=account_type,
        initial_balance=account.current_balance,
        asset_allocation=account.asset_allocation,
        balances=[],
        transactions=[],
    )


def create_account_evolution_engine_from_scenario(
    accounts: Any,  # Accounts type from scenario.py
    time_grid: TimeGrid,
    inflation_adjuster: InflationAdjuster,
    market_returns: Dict[str, float],
) -> AccountEvolutionEngine:
    """Create an AccountEvolutionEngine from scenario account data."""
    engine = AccountEvolutionEngine(
        time_grid=time_grid,
        inflation_adjuster=inflation_adjuster,
        market_returns=market_returns,
    )

    # Add all investment accounts
    for account in accounts.taxable:
        engine.add_account(
            create_account_evolution_from_scenario_account(account, "taxable")
        )

    for account in accounts.traditional_401k:
        engine.add_account(
            create_account_evolution_from_scenario_account(account, "traditional_401k")
        )

    for account in accounts.roth_401k:
        engine.add_account(
            create_account_evolution_from_scenario_account(account, "roth_401k")
        )

    for account in accounts.traditional_ira:
        engine.add_account(
            create_account_evolution_from_scenario_account(account, "traditional_ira")
        )

    for account in accounts.roth_ira:
        engine.add_account(
            create_account_evolution_from_scenario_account(account, "roth_ira")
        )

    for account in accounts.hsa:
        engine.add_account(
            create_account_evolution_from_scenario_account(account, "hsa")
        )

    for account in accounts.college_529:
        engine.add_account(
            create_account_evolution_from_scenario_account(account, "college_529")
        )

    # TODO: Handle cash accounts separately (they don't have asset allocation)

    return engine


def calculate_portfolio_return(
    asset_allocation: AssetAllocation, market_returns: Dict[str, float]
) -> float:
    """Calculate expected portfolio return based on asset allocation and market returns."""
    total_return = 0.0
    allocation_dict = asset_allocation.model_dump()

    for asset_class, allocation in allocation_dict.items():
        if allocation > 0 and asset_class in market_returns:
            total_return += allocation * market_returns[asset_class]

    return total_return


def validate_account_balance_consistency(engine: AccountEvolutionEngine) -> bool:
    """Validate that account balances are consistent with transactions."""
    for account in engine.accounts:
        # Group transactions by year
        transactions_by_year: Dict[int, List[Transaction]] = {}
        for transaction in account.transactions:
            if transaction.year not in transactions_by_year:
                transactions_by_year[transaction.year] = []
            transactions_by_year[transaction.year].append(transaction)

        # Process each year in order
        running_balance = account.initial_balance
        for year in sorted(transactions_by_year.keys()):
            year_transactions = transactions_by_year[year]

            # Apply all transactions for this year
            for transaction in year_transactions:
                if transaction.transaction_type == "contribution":
                    running_balance += transaction.amount
                elif transaction.transaction_type == "withdrawal":
                    running_balance += transaction.amount  # amount is already negative
                elif transaction.transaction_type == "growth":
                    running_balance += transaction.amount

            # Check if balance matches
            account_balance = account.get_balance(year)
            if abs(running_balance - account_balance) > 0.01:  # Allow for rounding
                return False

    return True
