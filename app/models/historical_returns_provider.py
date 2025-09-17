"""
Historical returns provider for bootstrap simulations.

This module implements the ReturnsProvider protocol using historical data
for bootstrap-style Monte Carlo simulations.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from app.models.historical_data import HistoricalDataManager, HistoricalDataSet
from app.models.simulation.protocols import ReturnsProvider

logger = logging.getLogger(__name__)


class HistoricalReturnsProvider(ReturnsProvider):
    """
    Returns provider that uses historical data for bootstrap simulations.
    
    This provider samples historical returns to generate Monte Carlo paths,
    maintaining the historical correlation structure and return distributions.
    """
    
    def __init__(
        self,
        data_manager: HistoricalDataManager,
        asset_classes: List[str],
        start_date: datetime,
        end_date: datetime,
        seed: Optional[int] = None
    ):
        """Initialize the historical returns provider.
        
        Args:
            data_manager: Historical data manager instance
            asset_classes: List of asset class names to include
            start_date: Start date for historical data
            end_date: End date for historical data
            seed: Random seed for reproducibility
        """
        self.data_manager = data_manager
        self.asset_classes = asset_classes
        self.start_date = start_date
        self.end_date = end_date
        self.seed = seed
        
        # Load historical datasets
        self.datasets = self._load_historical_datasets()
        
        # Calculate historical returns matrix
        self.returns_matrix = self._calculate_returns_matrix()
        
        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
    
    def _load_historical_datasets(self) -> Dict[str, HistoricalDataSet]:
        """Load historical datasets for all asset classes."""
        datasets = {}
        
        for asset_class in self.asset_classes:
            try:
                # Try to load from storage first
                available_datasets = self.data_manager.list_available_datasets()
                
                # Find the most recent dataset for this asset class
                matching_datasets = [
                    ds for ds in available_datasets
                    if ds["metadata"].get("asset_class") == asset_class
                ]
                
                if matching_datasets:
                    # Load the most recent dataset
                    latest_dataset = max(matching_datasets, key=lambda x: x["metadata"].get("start_date", ""))
                    dataset = self.data_manager.load_dataset(latest_dataset["path"])
                    datasets[asset_class] = dataset
                    logger.info(f"Loaded historical dataset for {asset_class}")
                else:
                    logger.warning(f"No historical dataset found for {asset_class}")
                    
            except Exception as e:
                logger.error(f"Error loading dataset for {asset_class}: {e}")
        
        return datasets
    
    def _calculate_returns_matrix(self) -> NDArray[np.float64]:
        """Calculate the historical returns matrix for all asset classes."""
        if not self.datasets:
            # Return a default returns matrix with zero returns if no data is available
            logger.warning("No historical datasets available, using zero returns")
            num_assets = len(self.asset_classes)
            return np.zeros((num_assets, 1))  # 1 period of zero returns
        
        # Get the minimum number of observations across all datasets
        min_observations = min(
            len(dataset.get_returns()) for dataset in self.datasets.values()
        )
        
        if min_observations == 0:
            # Return a default returns matrix with zero returns if no returns data
            logger.warning("No historical returns data available, using zero returns")
            num_assets = len(self.asset_classes)
            return np.zeros((num_assets, 1))  # 1 period of zero returns
        
        # Create returns matrix
        num_assets = len(self.asset_classes)
        returns_matrix = np.zeros((num_assets, min_observations))
        
        for i, asset_class in enumerate(self.asset_classes):
            if asset_class in self.datasets:
                returns = self.datasets[asset_class].get_returns()
                # Take the last min_observations returns
                returns_matrix[i] = returns[-min_observations:]
            else:
                logger.warning(f"No data for asset class {asset_class}, using zeros")
                returns_matrix[i] = np.zeros(min_observations)
        
        return returns_matrix
    
    def generate_returns(
        self, years: int, num_paths: int, seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        """Generate asset returns for simulation using historical bootstrap.
        
        Args:
            years: Number of years to simulate
            num_paths: Number of Monte Carlo paths
            seed: Random seed for reproducibility
            
        Returns:
            NDArray of shape (num_assets, years, num_paths) with returns
            
        Raises:
            ValueError: If years or num_paths <= 0
        """
        if years <= 0:
            raise ValueError("Years must be positive")
        if num_paths <= 0:
            raise ValueError("Number of paths must be positive")
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        elif self.seed is not None:
            np.random.seed(self.seed)
        
        num_assets = len(self.asset_classes)
        num_historical_periods = self.returns_matrix.shape[1]
        
        # Initialize returns array
        returns = np.zeros((num_assets, years, num_paths))
        
        # Generate bootstrap samples
        for path in range(num_paths):
            for year in range(years):
                # Sample a random historical period
                historical_index = np.random.randint(0, num_historical_periods)
                
                # Use the historical returns for this period
                returns[:, year, path] = self.returns_matrix[:, historical_index]
        
        return returns
    
    def get_asset_names(self) -> List[str]:
        """Get names of assets in order."""
        return self.asset_classes.copy()
    
    def get_target_weights(self) -> NDArray[np.float64]:
        """Get target portfolio weights for assets.
        
        For historical bootstrap, we use equal weights as a default.
        In practice, this should be configured based on the user's target allocation.
        """
        num_assets = len(self.asset_classes)
        return np.ones(num_assets) / num_assets
    
    def get_expected_returns(self) -> NDArray[np.float64]:
        """Get long-term expected returns for each asset.
        
        Returns the historical mean returns for each asset class.
        """
        if not self.datasets:
            return np.zeros(len(self.asset_classes))
        
        expected_returns = np.zeros(len(self.asset_classes))
        
        for i, asset_class in enumerate(self.asset_classes):
            if asset_class in self.datasets:
                stats = self.datasets[asset_class].get_statistics()
                expected_returns[i] = stats.get("mean_return", 0.0)
        
        return expected_returns
    
    def get_historical_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get historical statistics for all asset classes."""
        statistics = {}
        
        for asset_class in self.asset_classes:
            if asset_class in self.datasets:
                statistics[asset_class] = self.datasets[asset_class].get_statistics()
            else:
                statistics[asset_class] = {}
        
        return statistics
    
    def get_correlation_matrix(self) -> NDArray[np.float64]:
        """Get the historical correlation matrix between asset classes."""
        if self.returns_matrix.shape[0] < 2:
            # Not enough assets for correlation
            return np.eye(len(self.asset_classes))
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(self.returns_matrix)
        
        # Handle NaN values (can occur if an asset has zero variance)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        return correlation_matrix
    
    def update_data(self, start_date: datetime, end_date: datetime) -> None:
        """Update historical data with new date range."""
        self.start_date = start_date
        self.end_date = end_date
        
        # Reload datasets
        self.datasets = self._load_historical_datasets()
        
        # Recalculate returns matrix
        self.returns_matrix = self._calculate_returns_matrix()
        
        logger.info(f"Updated historical data from {start_date} to {end_date}")


def create_historical_returns_provider(
    data_manager: HistoricalDataManager,
    asset_classes: Optional[List[str]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    seed: Optional[int] = None
) -> HistoricalReturnsProvider:
    """Create a historical returns provider with default settings.
    
    Args:
        data_manager: Historical data manager instance
        asset_classes: List of asset class names (defaults to common asset classes)
        start_date: Start date for historical data (defaults to 20 years ago)
        end_date: End date for historical data (defaults to today)
        seed: Random seed for reproducibility
        
    Returns:
        Configured HistoricalReturnsProvider instance
    """
    if asset_classes is None:
        asset_classes = ["stocks", "bonds", "cash", "reits"]
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=20 * 365)  # 20 years ago
    
    if end_date is None:
        end_date = datetime.now()
    
    return HistoricalReturnsProvider(
        data_manager=data_manager,
        asset_classes=asset_classes,
        start_date=start_date,
        end_date=end_date,
        seed=seed
    )
