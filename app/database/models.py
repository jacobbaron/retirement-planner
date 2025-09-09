"""
SQLAlchemy database models for the retirement planner.

This module defines the database tables and relationships for users, scenarios,
runs, and ledger rows.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, Text, 
    Numeric, Index, CheckConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

from .base import Base


class User(Base):
    """User model for authentication and profile information."""
    
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    scenarios = relationship("Scenario", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', name='{self.name}')>"


class Scenario(Base):
    """Scenario model for storing retirement planning scenarios."""
    
    __tablename__ = 'scenarios'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    scenario_data = Column(JSONB, nullable=False)  # Stores the Pydantic model as JSON
    version = Column(String(50), default='0.1', nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="scenarios")
    runs = relationship("Run", back_populates="scenario", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_scenarios_user_created', 'user_id', 'created_at'),
        Index('idx_scenarios_name', 'name'),
    )
    
    def __repr__(self):
        return f"<Scenario(id={self.id}, user_id={self.user_id}, name='{self.name}')>"


class Run(Base):
    """Run model for tracking simulation runs."""
    
    __tablename__ = 'runs'
    
    id = Column(Integer, primary_key=True, index=True)
    scenario_id = Column(Integer, ForeignKey('scenarios.id', ondelete='CASCADE'), nullable=False, index=True)
    status = Column(String(50), nullable=False, default='pending', index=True)
    run_type = Column(String(50), nullable=False, default='monte_carlo')  # monte_carlo, deterministic, historical
    num_simulations = Column(Integer, default=10000)
    results = Column(JSONB)  # Stores simulation results
    error_message = Column(Text)  # Store error details if run failed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    scenario = relationship("Scenario", back_populates="runs")
    ledger_rows = relationship("LedgerRow", back_populates="run", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed')", name='ck_run_status'),
        CheckConstraint("run_type IN ('monte_carlo', 'deterministic', 'historical')", name='ck_run_type'),
        CheckConstraint("num_simulations > 0", name='ck_num_simulations_positive'),
        Index('idx_runs_scenario_status', 'scenario_id', 'status'),
        Index('idx_runs_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Run(id={self.id}, scenario_id={self.scenario_id}, status='{self.status}')>"


class LedgerRow(Base):
    """LedgerRow model for storing detailed cashflow/transaction data."""
    
    __tablename__ = 'ledger_rows'
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey('runs.id', ondelete='CASCADE'), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    month = Column(Integer, default=1)  # 1-12, default to annual
    account_type = Column(String(50), nullable=False, index=True)
    account_name = Column(String(255), nullable=False)
    transaction_type = Column(String(50), nullable=False, index=True)
    amount = Column(Numeric(15, 2), nullable=False)
    balance_after = Column(Numeric(15, 2))  # Account balance after this transaction
    description = Column(Text)
    
    # Relationships
    run = relationship("Run", back_populates="ledger_rows")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("year >= 1900 AND year <= 2100", name='ck_year_range'),
        CheckConstraint("month >= 1 AND month <= 12", name='ck_month_range'),
        CheckConstraint("account_type IN ('taxable', 'traditional_401k', 'roth_401k', 'traditional_ira', 'roth_ira', 'hsa', 'college_529', 'cash')", name='ck_account_type'),
        CheckConstraint("transaction_type IN ('contribution', 'withdrawal', 'growth', 'dividend', 'interest', 'fee', 'rebalance', 'transfer')", name='ck_transaction_type'),
        Index('idx_ledger_run_year', 'run_id', 'year'),
        Index('idx_ledger_account_type', 'account_type'),
        Index('idx_ledger_transaction_type', 'transaction_type'),
        Index('idx_ledger_year_month', 'year', 'month'),
    )
    
    def __repr__(self):
        return f"<LedgerRow(id={self.id}, run_id={self.run_id}, year={self.year}, account='{self.account_name}', type='{self.transaction_type}', amount={self.amount})>"
