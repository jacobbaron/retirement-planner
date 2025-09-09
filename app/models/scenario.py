"""
Pydantic models for retirement planning scenarios.

This module defines the complete data structure for retirement planning scenarios
using Pydantic for validation, serialization, and type safety.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Child(BaseModel):
    """Child information for household planning."""

    birth_year: int = Field(..., ge=1900, le=2100, description="Year of birth")
    college_start_year: int = Field(
        ..., ge=1900, le=2100, description="Year college starts"
    )
    college_type: Literal["public", "private", "community"] = Field(
        default="public", description="Type of college"
    )


class Household(BaseModel):
    """Household composition and demographics."""

    primary_age: int = Field(
        ..., ge=18, le=100, description="Age of primary household member"
    )
    spouse_age: Optional[int] = Field(
        None, ge=18, le=100, description="Age of spouse (null if single)"
    )
    filing_status: Literal[
        "single",
        "married_filing_jointly",
        "married_filing_separately",
        "head_of_household",
        "qualifying_widow",
    ] = Field(..., description="Tax filing status")
    state: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="State of residence (2-letter code)",
    )
    children: List[Child] = Field(
        default_factory=list, description="List of children in household"
    )


class AssetAllocation(BaseModel):
    """Asset allocation for investment accounts."""

    stocks: float = Field(..., ge=0, le=1, description="Stock allocation (0-1)")
    bonds: float = Field(..., ge=0, le=1, description="Bond allocation (0-1)")
    cash: float = Field(..., ge=0, le=1, description="Cash allocation (0-1)")

    @field_validator("stocks", "bonds", "cash")
    @classmethod
    def validate_allocation(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Allocation must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def validate_total_allocation(self):
        total = self.stocks + self.bonds + self.cash
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Asset allocation must sum to 1.0, got {total}")
        return self


class InvestmentAccount(BaseModel):
    """Base class for investment accounts."""

    name: str = Field(..., min_length=1, description="Account name")
    current_balance: float = Field(..., ge=0, description="Current account balance")
    asset_allocation: AssetAllocation = Field(..., description="Asset allocation")


class TaxableAccount(InvestmentAccount):
    """Taxable investment account."""

    cost_basis: Optional[float] = Field(
        None, ge=0, description="Cost basis for tax calculations"
    )


class Traditional401kAccount(InvestmentAccount):
    """Traditional 401(k) account."""

    employer_match: Optional[float] = Field(
        None, ge=0, description="Annual employer match amount"
    )


class Roth401kAccount(InvestmentAccount):
    """Roth 401(k) account."""


class TraditionalIRAAccount(InvestmentAccount):
    """Traditional IRA account."""


class RothIRAAccount(InvestmentAccount):
    """Roth IRA account."""


class HSAAccount(InvestmentAccount):
    """Health Savings Account."""


class College529Account(InvestmentAccount):
    """529 college savings account."""

    beneficiary: str = Field(..., description="Name of child beneficiary")


class CashAccounts(BaseModel):
    """Cash and cash equivalent accounts."""

    checking: float = Field(default=0, ge=0, description="Checking account balance")
    savings: float = Field(default=0, ge=0, description="Savings account balance")
    cds: float = Field(default=0, ge=0, description="Certificate of deposit balance")
    money_market: float = Field(
        default=0, ge=0, description="Money market account balance"
    )


class Accounts(BaseModel):
    """All financial accounts and assets."""

    taxable: List[TaxableAccount] = Field(
        default_factory=list, description="Taxable investment accounts"
    )
    traditional_401k: List[Traditional401kAccount] = Field(
        default_factory=list, description="Traditional 401(k) accounts"
    )
    roth_401k: List[Roth401kAccount] = Field(
        default_factory=list, description="Roth 401(k) accounts"
    )
    traditional_ira: List[TraditionalIRAAccount] = Field(
        default_factory=list, description="Traditional IRA accounts"
    )
    roth_ira: List[RothIRAAccount] = Field(
        default_factory=list, description="Roth IRA accounts"
    )
    hsa: List[HSAAccount] = Field(
        default_factory=list, description="Health Savings Accounts"
    )
    college_529: List[College529Account] = Field(
        default_factory=list, description="529 college savings accounts"
    )
    cash: CashAccounts = Field(
        default_factory=CashAccounts, description="Cash and cash equivalents"
    )


class Mortgage(BaseModel):
    """Mortgage loan information."""

    name: str = Field(..., min_length=1, description="Mortgage name")
    original_balance: float = Field(..., ge=0, description="Original loan balance")
    current_balance: float = Field(..., ge=0, description="Current loan balance")
    interest_rate: float = Field(..., ge=0, le=1, description="Interest rate (0-1)")
    term_years: int = Field(..., ge=1, le=50, description="Loan term in years")
    start_year: int = Field(..., ge=1900, le=2100, description="Loan start year")
    property_value: Optional[float] = Field(
        None, ge=0, description="Current property value"
    )
    extra_payment: float = Field(
        default=0, ge=0, description="Monthly extra payment amount"
    )


class StudentLoan(BaseModel):
    """Student loan information."""

    name: str = Field(..., min_length=1, description="Loan name")
    current_balance: float = Field(..., ge=0, description="Current loan balance")
    interest_rate: float = Field(..., ge=0, le=1, description="Interest rate (0-1)")
    minimum_payment: float = Field(..., ge=0, description="Minimum monthly payment")
    loan_type: Literal["federal", "private"] = Field(
        default="federal", description="Type of loan"
    )


class CreditCard(BaseModel):
    """Credit card debt information."""

    name: str = Field(..., min_length=1, description="Credit card name")
    current_balance: float = Field(..., ge=0, description="Current balance")
    interest_rate: float = Field(..., ge=0, le=1, description="Interest rate (0-1)")
    minimum_payment: float = Field(..., ge=0, description="Minimum monthly payment")


class AutoLoan(BaseModel):
    """Auto loan information."""

    name: str = Field(..., min_length=1, description="Loan name")
    current_balance: float = Field(..., ge=0, description="Current loan balance")
    interest_rate: float = Field(..., ge=0, le=1, description="Interest rate (0-1)")
    term_months: int = Field(..., ge=1, le=120, description="Loan term in months")
    monthly_payment: float = Field(..., ge=0, description="Monthly payment amount")


class Liabilities(BaseModel):
    """All debts and liabilities."""

    mortgages: List[Mortgage] = Field(
        default_factory=list, description="Mortgage loans"
    )
    student_loans: List[StudentLoan] = Field(
        default_factory=list, description="Student loan debt"
    )
    credit_cards: List[CreditCard] = Field(
        default_factory=list, description="Credit card debt"
    )
    auto_loans: List[AutoLoan] = Field(
        default_factory=list, description="Auto loan debt"
    )


class Salary(BaseModel):
    """Salary income information."""

    person: Literal["primary", "spouse"] = Field(
        ..., description="Which household member"
    )
    annual_amount: float = Field(..., ge=0, description="Annual salary amount")
    start_year: int = Field(..., ge=1900, le=2100, description="Start year")
    end_year: int = Field(..., ge=1900, le=2100, description="End year")
    growth_rate: float = Field(default=0.03, description="Annual salary growth rate")
    bonus: float = Field(default=0, ge=0, description="Annual bonus amount")

    @model_validator(mode="after")
    def validate_end_year(self):
        if self.end_year < self.start_year:
            raise ValueError("End year must be >= start year")
        return self


class SocialSecurity(BaseModel):
    """Social Security benefit information."""

    person: Literal["primary", "spouse"] = Field(
        ..., description="Which household member"
    )
    annual_benefit: float = Field(..., ge=0, description="Annual benefit amount")
    claim_year: int = Field(..., ge=1900, le=2100, description="Year benefits start")
    benefit_type: Literal["retirement", "spousal", "survivor"] = Field(
        default="retirement", description="Type of benefit"
    )


class Pension(BaseModel):
    """Pension income information."""

    person: Literal["primary", "spouse"] = Field(
        ..., description="Which household member"
    )
    annual_amount: float = Field(..., ge=0, description="Annual pension amount")
    start_year: int = Field(..., ge=1900, le=2100, description="Start year")
    end_year: int = Field(..., ge=1900, le=2100, description="End year")
    cola: float = Field(default=0.02, description="Annual cost of living adjustment")

    @model_validator(mode="after")
    def validate_end_year(self):
        if self.end_year < self.start_year:
            raise ValueError("End year must be >= start year")
        return self


class OtherIncome(BaseModel):
    """Other income sources."""

    name: str = Field(..., min_length=1, description="Income source name")
    annual_amount: float = Field(..., ge=0, description="Annual income amount")
    start_year: int = Field(..., ge=1900, le=2100, description="Start year")
    end_year: int = Field(..., ge=1900, le=2100, description="End year")
    growth_rate: float = Field(default=0, description="Annual growth rate")

    @model_validator(mode="after")
    def validate_end_year(self):
        if self.end_year < self.start_year:
            raise ValueError("End year must be >= start year")
        return self


class Incomes(BaseModel):
    """All income sources and projections."""

    salary: List[Salary] = Field(
        default_factory=list, description="Salary income by person"
    )
    social_security: List[SocialSecurity] = Field(
        default_factory=list, description="Social Security benefits"
    )
    pension: List[Pension] = Field(default_factory=list, description="Pension income")
    other: List[OtherIncome] = Field(
        default_factory=list, description="Other income sources"
    )


class HousingExpenses(BaseModel):
    """Housing-related expenses."""

    mortgage_payment: float = Field(
        default=0, ge=0, description="Monthly mortgage payment"
    )
    property_tax: float = Field(default=0, ge=0, description="Annual property tax")
    home_insurance: float = Field(default=0, ge=0, description="Annual home insurance")
    hoa_fees: float = Field(default=0, ge=0, description="Annual HOA fees")
    maintenance: float = Field(default=0, ge=0, description="Annual maintenance costs")
    utilities: float = Field(default=0, ge=0, description="Monthly utilities")


class TransportationExpenses(BaseModel):
    """Transportation expenses."""

    auto_payment: float = Field(
        default=0, ge=0, description="Monthly auto loan payment"
    )
    auto_insurance: float = Field(default=0, ge=0, description="Annual auto insurance")
    gas: float = Field(default=0, ge=0, description="Monthly gas expenses")
    maintenance: float = Field(default=0, ge=0, description="Annual auto maintenance")


class HealthcareExpenses(BaseModel):
    """Healthcare expenses."""

    insurance: float = Field(
        default=0, ge=0, description="Monthly health insurance premium"
    )
    out_of_pocket: float = Field(
        default=0, ge=0, description="Annual out-of-pocket healthcare costs"
    )
    medicare: float = Field(
        default=0, ge=0, description="Monthly Medicare premium (when applicable)"
    )


class LumpyExpense(BaseModel):
    """One-time or irregular expenses."""

    name: str = Field(..., min_length=1, description="Expense name")
    amount: float = Field(..., ge=0, description="Expense amount")
    year: int = Field(..., ge=1900, le=2100, description="Year expense occurs")
    category: Literal[
        "home_improvement", "vehicle", "medical", "education", "other"
    ] = Field(default="other", description="Expense category")


class Expenses(BaseModel):
    """All expense categories and projections."""

    housing: HousingExpenses = Field(
        default_factory=HousingExpenses, description="Housing-related expenses"
    )
    transportation: TransportationExpenses = Field(
        default_factory=TransportationExpenses, description="Transportation expenses"
    )
    healthcare: HealthcareExpenses = Field(
        default_factory=HealthcareExpenses, description="Healthcare expenses"
    )
    food: float = Field(default=0, ge=0, description="Monthly food expenses")
    entertainment: float = Field(
        default=0, ge=0, description="Monthly entertainment expenses"
    )
    travel: float = Field(default=0, ge=0, description="Annual travel expenses")
    education: float = Field(
        default=0, ge=0, description="Annual education expenses (for children)"
    )
    other: float = Field(default=0, ge=0, description="Monthly other expenses")
    lumpy_expenses: List[LumpyExpense] = Field(
        default_factory=list, description="One-time or irregular expenses"
    )


class LifeInsurance(BaseModel):
    """Life insurance policy information."""

    person: Literal["primary", "spouse"] = Field(..., description="Insured person")
    death_benefit: float = Field(..., ge=0, description="Death benefit amount")
    annual_premium: float = Field(..., ge=0, description="Annual premium amount")
    term_years: int = Field(..., ge=1, le=50, description="Policy term in years")
    policy_type: Literal["term", "whole", "universal"] = Field(
        default="term", description="Type of policy"
    )


class DisabilityInsurance(BaseModel):
    """Disability insurance policy information."""

    person: Literal["primary", "spouse"] = Field(..., description="Insured person")
    monthly_benefit: float = Field(..., ge=0, description="Monthly benefit amount")
    annual_premium: float = Field(..., ge=0, description="Annual premium amount")
    elimination_period: int = Field(
        default=90, ge=0, description="Days before benefits begin"
    )


class LongTermCareInsurance(BaseModel):
    """Long-term care insurance policy information."""

    person: Literal["primary", "spouse"] = Field(..., description="Insured person")
    daily_benefit: float = Field(..., ge=0, description="Daily benefit amount")
    annual_premium: float = Field(..., ge=0, description="Annual premium amount")
    benefit_period: int = Field(default=3, ge=1, description="Years of coverage")


class Policies(BaseModel):
    """Insurance and other policies."""

    life_insurance: List[LifeInsurance] = Field(
        default_factory=list, description="Life insurance policies"
    )
    disability_insurance: List[DisabilityInsurance] = Field(
        default_factory=list, description="Disability insurance policies"
    )
    long_term_care: List[LongTermCareInsurance] = Field(
        default_factory=list, description="Long-term care insurance"
    )


class ExpectedReturns(BaseModel):
    """Expected annual returns by asset class."""

    stocks: float = Field(..., ge=-1, le=1, description="Expected annual stock return")
    bonds: float = Field(..., ge=-1, le=1, description="Expected annual bond return")
    cash: float = Field(..., ge=-1, le=1, description="Expected annual cash return")


class Volatility(BaseModel):
    """Annual volatility by asset class."""

    stocks: float = Field(..., ge=0, le=1, description="Annual stock volatility")
    bonds: float = Field(..., ge=0, le=1, description="Annual bond volatility")
    cash: float = Field(..., ge=0, le=1, description="Annual cash volatility")


class Correlations(BaseModel):
    """Correlation matrix between asset classes."""

    stocks_bonds: float = Field(
        default=0.2, ge=-1, le=1, description="Stocks-Bonds correlation"
    )
    stocks_cash: float = Field(
        default=0.0, ge=-1, le=1, description="Stocks-Cash correlation"
    )
    bonds_cash: float = Field(
        default=0.1, ge=-1, le=1, description="Bonds-Cash correlation"
    )


class MarketModel(BaseModel):
    """Market assumptions and return expectations."""

    expected_returns: ExpectedReturns = Field(
        ..., description="Expected annual returns by asset class"
    )
    volatility: Volatility = Field(..., description="Annual volatility by asset class")
    correlations: Correlations = Field(
        default_factory=Correlations,
        description="Correlation matrix between asset classes",
    )
    inflation: float = Field(
        default=0.025, ge=0, le=1, description="Expected annual inflation rate"
    )
    simulation_type: Literal["deterministic", "monte_carlo", "historical_bootstrap"] = (
        Field(default="monte_carlo", description="Type of simulation to run")
    )
    num_simulations: int = Field(
        default=10000, ge=1, le=100000, description="Number of Monte Carlo simulations"
    )


class Rebalancing(BaseModel):
    """Portfolio rebalancing strategy."""

    frequency: Literal[
        "annual", "semi_annual", "quarterly", "monthly", "threshold_based"
    ] = Field(default="annual", description="Rebalancing frequency")
    threshold: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Rebalancing threshold (for threshold_based)",
    )


class TaxOptimization(BaseModel):
    """Tax optimization strategies."""

    withdrawal_order: List[Literal["cash", "taxable", "traditional", "roth"]] = Field(
        default=["cash", "taxable", "traditional", "roth"],
        description="Order of account types for withdrawals",
    )
    tax_loss_harvesting: bool = Field(
        default=True, description="Enable tax loss harvesting"
    )
    roth_conversions: bool = Field(default=False, description="Enable Roth conversions")


class SocialSecurityStrategy(BaseModel):
    """Social Security claiming strategy."""

    primary_claim_age: int = Field(
        default=67, ge=62, le=70, description="Age to claim primary Social Security"
    )
    spouse_claim_age: int = Field(
        default=67, ge=62, le=70, description="Age to claim spousal Social Security"
    )
    file_and_suspend: bool = Field(
        default=False, description="Use file and suspend strategy"
    )


class Strategy(BaseModel):
    """Withdrawal and investment strategy."""

    withdrawal_method: Literal[
        "fixed_real", "fixed_percentage", "vpw", "guyton_klinger"
    ] = Field(
        default="fixed_real", description="Method for calculating annual withdrawals"
    )
    withdrawal_rate: float = Field(
        default=0.04,
        ge=0,
        le=1,
        description="Initial withdrawal rate (for fixed methods)",
    )
    withdrawal_start_year: Optional[int] = Field(
        None, ge=1900, le=2100, description="Year to start withdrawals"
    )
    rebalancing: Rebalancing = Field(
        default_factory=Rebalancing, description="Portfolio rebalancing strategy"
    )
    tax_optimization: TaxOptimization = Field(
        default_factory=TaxOptimization, description="Tax optimization strategies"
    )
    social_security_strategy: SocialSecurityStrategy = Field(
        default_factory=SocialSecurityStrategy,
        description="Social Security claiming strategy",
    )


class ScenarioMetadata(BaseModel):
    """Metadata about the scenario."""

    name: str = Field(..., min_length=1, description="Scenario name")
    description: Optional[str] = Field(None, description="Scenario description")
    created_date: Optional[datetime] = Field(
        None, description="When scenario was created"
    )
    modified_date: Optional[datetime] = Field(
        None, description="When scenario was last modified"
    )
    version: str = Field(default="0.1", description="Schema version")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class Scenario(BaseModel):
    """Complete retirement planning scenario."""

    household: Household = Field(
        ..., description="Household composition and demographics"
    )
    accounts: Accounts = Field(..., description="Financial accounts and assets")
    liabilities: Liabilities = Field(..., description="Debts and liabilities")
    incomes: Incomes = Field(..., description="Income sources and projections")
    expenses: Expenses = Field(..., description="Expense categories and projections")
    policies: Policies = Field(..., description="Insurance and other policies")
    market_model: MarketModel = Field(
        ..., description="Market assumptions and return expectations"
    )
    strategy: Strategy = Field(..., description="Withdrawal and investment strategy")
    scenario_metadata: Optional[ScenarioMetadata] = Field(
        None, description="Metadata about the scenario"
    )

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        json_schema_extra={
            "example": {
                "household": {
                    "primary_age": 35,
                    "spouse_age": 33,
                    "filing_status": "married_filing_jointly",
                    "state": "CA",
                },
                "accounts": {
                    "taxable": [],
                    "traditional_401k": [],
                    "roth_401k": [],
                    "traditional_ira": [],
                    "roth_ira": [],
                    "hsa": [],
                    "college_529": [],
                    "cash": {},
                },
                "liabilities": {
                    "mortgages": [],
                    "student_loans": [],
                    "credit_cards": [],
                    "auto_loans": [],
                },
                "incomes": {
                    "salary": [],
                    "social_security": [],
                    "pension": [],
                    "other": [],
                },
                "expenses": {
                    "housing": {},
                    "transportation": {},
                    "healthcare": {},
                    "food": 0,
                    "entertainment": 0,
                    "travel": 0,
                    "education": 0,
                    "other": 0,
                    "lumpy_expenses": [],
                },
                "policies": {
                    "life_insurance": [],
                    "disability_insurance": [],
                    "long_term_care": [],
                },
                "market_model": {
                    "expected_returns": {"stocks": 0.08, "bonds": 0.04, "cash": 0.02},
                    "volatility": {"stocks": 0.18, "bonds": 0.05, "cash": 0.01},
                    "inflation": 0.025,
                },
                "strategy": {
                    "withdrawal_method": "fixed_real",
                    "withdrawal_rate": 0.04,
                    "rebalancing": {"frequency": "annual", "threshold": 0.05},
                },
            }
        },
    )
