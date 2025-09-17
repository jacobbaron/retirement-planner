"""
Historical data acquisition and management for retirement planning.

This module provides functionality to acquire, validate, and manage historical
market data from multiple sources including Yahoo Finance, FRED, and custom CSVs.
"""

import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, Field, field_validator
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.storage.base import StorageService, StorageError

logger = logging.getLogger(__name__)


class AssetClass(BaseModel):
    """Represents an asset class with metadata."""
    
    name: str = Field(..., description="Asset class name (e.g., 'stocks', 'bonds')")
    symbol: str = Field(..., description="Symbol or identifier for the asset")
    source: str = Field(..., description="Data source (yahoo, fred, csv)")
    description: Optional[str] = Field(None, description="Human-readable description")


class HistoricalDataPoint(BaseModel):
    """A single historical data point."""
    
    date: datetime = Field(..., description="Date of the data point")
    value: float = Field(..., description="Value (price, return, etc.)")
    asset_class: str = Field(..., description="Asset class identifier")
    source: str = Field(..., description="Data source")


class HistoricalDataSet(BaseModel):
    """A complete historical dataset for an asset class."""
    
    asset_class: AssetClass = Field(..., description="Asset class information")
    data_points: List[HistoricalDataPoint] = Field(..., description="Historical data points")
    start_date: datetime = Field(..., description="Start date of the dataset")
    end_date: datetime = Field(..., description="End date of the dataset")
    frequency: str = Field(default="monthly", description="Data frequency (daily, monthly, annual)")
    
    @field_validator("data_points")
    @classmethod
    def validate_data_points(cls, v: List[HistoricalDataPoint]) -> List[HistoricalDataPoint]:
        """Validate that data points are sorted by date and have no gaps."""
        if not v:
            raise ValueError("Data points cannot be empty")
        
        # Sort by date
        v.sort(key=lambda x: x.date)
        
        # Check for gaps (basic validation)
        for i in range(1, len(v)):
            prev_date = v[i-1].date
            curr_date = v[i].date
            
            # Check for reasonable gap (not more than 2 months for monthly data)
            if (curr_date - prev_date).days > 60:
                logger.warning(f"Large gap detected between {prev_date} and {curr_date}")
        
        return v
    
    def get_returns(self) -> List[float]:
        """Calculate returns from the data points."""
        if len(self.data_points) < 2:
            return []
        
        returns = []
        for i in range(1, len(self.data_points)):
            prev_value = self.data_points[i-1].value
            curr_value = self.data_points[i].value
            
            if prev_value != 0:
                return_rate = (curr_value - prev_value) / prev_value
                returns.append(return_rate)
            else:
                returns.append(0.0)
        
        return returns
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate basic statistics for the dataset."""
        returns = self.get_returns()
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        
        return {
            "mean_return": float(np.mean(returns_array)),
            "volatility": float(np.std(returns_array)),
            "min_return": float(np.min(returns_array)),
            "max_return": float(np.max(returns_array)),
            "total_return": float((self.data_points[-1].value / self.data_points[0].value) - 1),
            "num_observations": len(returns)
        }


class DataSourceConfig(BaseModel):
    """Configuration for a data source."""
    
    name: str = Field(..., description="Source name")
    base_url: Optional[str] = Field(None, description="Base URL for API calls")
    api_key: Optional[str] = Field(None, description="API key if required")
    rate_limit: Optional[int] = Field(None, description="Rate limit (requests per minute)")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class YahooFinanceSource:
    """Data source for Yahoo Finance."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[HistoricalDataPoint]:
        """Fetch historical data from Yahoo Finance."""
        try:
            # Convert dates to timestamps
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            # Yahoo Finance API endpoint
            url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
            params = {
                "period1": start_timestamp,
                "period2": end_timestamp,
                "interval": "1mo",  # Monthly data
                "events": "history"
            }
            
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            # Parse CSV data
            csv_data = response.text
            data_points = []
            
            for row in csv.DictReader(csv_data.splitlines()):
                date = datetime.strptime(row["Date"], "%Y-%m-%d")
                close_price = float(row["Close"])
                
                data_points.append(HistoricalDataPoint(
                    date=date,
                    value=close_price,
                    asset_class=symbol,
                    source="yahoo"
                ))
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance for {symbol}: {e}")
            raise StorageError(f"Failed to fetch data from Yahoo Finance: {e}")


class FREDSource:
    """Data source for Federal Reserve Economic Data (FRED)."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def fetch_data(self, series_id: str, start_date: datetime, end_date: datetime) -> List[HistoricalDataPoint]:
        """Fetch historical data from FRED."""
        try:
            if not self.config.api_key:
                raise StorageError("FRED API key is required")
            
            # FRED API endpoint
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.config.api_key,
                "file_type": "json",
                "observation_start": start_date.strftime("%Y-%m-%d"),
                "observation_end": end_date.strftime("%Y-%m-%d"),
                "frequency": "m"  # Monthly
            }
            
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            data = response.json()
            data_points = []
            
            for observation in data.get("observations", []):
                if observation.get("value") != ".":  # Skip missing values
                    date = datetime.strptime(observation["date"], "%Y-%m-%d")
                    value = float(observation["value"])
                    
                    data_points.append(HistoricalDataPoint(
                        date=date,
                        value=value,
                        asset_class=series_id,
                        source="fred"
                    ))
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error fetching data from FRED for {series_id}: {e}")
            raise StorageError(f"Failed to fetch data from FRED: {e}")


class CSVSource:
    """Data source for custom CSV files."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
    
    def fetch_data(self, file_path: str, start_date: datetime, end_date: datetime) -> List[HistoricalDataPoint]:
        """Fetch historical data from a CSV file."""
        try:
            csv_path = Path(file_path)
            if not csv_path.exists():
                raise StorageError(f"CSV file not found: {file_path}")
            
            data_points = []
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Expect columns: date, value, asset_class
                    date = datetime.strptime(row["date"], "%Y-%m-%d")
                    value = float(row["value"])
                    asset_class = row.get("asset_class", "unknown")
                    
                    # Filter by date range
                    if start_date <= date <= end_date:
                        data_points.append(HistoricalDataPoint(
                            date=date,
                            value=value,
                            asset_class=asset_class,
                            source="csv"
                        ))
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            raise StorageError(f"Failed to read CSV file: {e}")


class HistoricalDataManager:
    """Manages historical data acquisition, validation, and storage."""
    
    def __init__(self, storage_service: StorageService):
        self.storage_service = storage_service
        self.sources: Dict[str, Union[YahooFinanceSource, FREDSource, CSVSource]] = {}
        self.datasets: Dict[str, HistoricalDataSet] = {}
    
    def add_data_source(self, name: str, source: Union[YahooFinanceSource, FREDSource, CSVSource]) -> None:
        """Add a data source."""
        self.sources[name] = source
    
    def acquire_data(
        self,
        asset_class: AssetClass,
        start_date: datetime,
        end_date: datetime,
        source_name: str
    ) -> HistoricalDataSet:
        """Acquire historical data for an asset class."""
        if source_name not in self.sources:
            raise StorageError(f"Data source '{source_name}' not found")
        
        source = self.sources[source_name]
        
        # Fetch data from source
        if isinstance(source, YahooFinanceSource):
            data_points = source.fetch_data(asset_class.symbol, start_date, end_date)
        elif isinstance(source, FREDSource):
            data_points = source.fetch_data(asset_class.symbol, start_date, end_date)
        elif isinstance(source, CSVSource):
            data_points = source.fetch_data(asset_class.symbol, start_date, end_date)
        else:
            raise StorageError(f"Unsupported source type: {type(source)}")
        
        # Create dataset
        dataset = HistoricalDataSet(
            asset_class=asset_class,
            data_points=data_points,
            start_date=start_date,
            end_date=end_date
        )
        
        # Store dataset
        self.datasets[asset_class.name] = dataset
        
        return dataset
    
    def validate_data_integrity(self, dataset: HistoricalDataSet) -> Dict[str, Any]:
        """Validate data integrity and return validation results."""
        validation_results: Dict[str, Any] = {
            "is_valid": True,
            "issues": [],
            "statistics": dataset.get_statistics()
        }
        
        # Check minimum data requirements
        if len(dataset.data_points) < 240:  # 20 years of monthly data
            validation_results["issues"].append("Insufficient data: less than 20 years")
            validation_results["is_valid"] = False
        
        # Check for gaps
        for i in range(1, len(dataset.data_points)):
            prev_date = dataset.data_points[i-1].date
            curr_date = dataset.data_points[i].date
            
            # Check for reasonable gap (not more than 2 months for monthly data)
            if (curr_date - prev_date).days > 60:
                validation_results["issues"].append(f"Large gap between {prev_date} and {curr_date}")
        
        # Check for missing values
        for point in dataset.data_points:
            if point.value is None or np.isnan(point.value):
                validation_results["issues"].append(f"Missing value at {point.date}")
                validation_results["is_valid"] = False
        
        return validation_results
    
    def store_dataset(self, dataset: HistoricalDataSet) -> str:
        """Store a dataset to the storage service."""
        try:
            # Convert dataset to JSON
            dataset_json = dataset.model_dump_json()
            
            # Create storage path
            storage_path = f"historical_data/{dataset.asset_class.name}_{dataset.start_date.strftime('%Y%m%d')}_{dataset.end_date.strftime('%Y%m%d')}.json"
            
            # Store the data
            self.storage_service.store_file(
                storage_path,
                dataset_json.encode('utf-8'),
                metadata={
                    "asset_class": dataset.asset_class.name,
                    "start_date": dataset.start_date.isoformat(),
                    "end_date": dataset.end_date.isoformat(),
                    "source": dataset.asset_class.source,
                    "num_points": len(dataset.data_points)
                }
            )
            
            logger.info(f"Stored dataset for {dataset.asset_class.name} at {storage_path}")
            return storage_path
            
        except Exception as e:
            logger.error(f"Error storing dataset: {e}")
            raise StorageError(f"Failed to store dataset: {e}")
    
    def load_dataset(self, storage_path: str) -> HistoricalDataSet:
        """Load a dataset from the storage service."""
        try:
            # Retrieve data from storage
            data_bytes = self.storage_service.retrieve_file(storage_path)
            dataset_json = data_bytes.decode('utf-8')
            
            # Parse JSON and create dataset
            dataset_dict = json.loads(dataset_json)
            
            # Convert date strings back to datetime objects
            for point_dict in dataset_dict["data_points"]:
                point_dict["date"] = datetime.fromisoformat(point_dict["date"])
            
            dataset = HistoricalDataSet.model_validate(dataset_dict)
            
            logger.info(f"Loaded dataset for {dataset.asset_class.name} from {storage_path}")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise StorageError(f"Failed to load dataset: {e}")
    
    def list_available_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets in storage."""
        try:
            files = self.storage_service.list_files("historical_data/")
            datasets: List[Dict[str, Any]] = []
            
            for file_path in files:
                if file_path.endswith('.json'):
                    try:
                        metadata = self.storage_service.get_file_metadata(file_path)
                        datasets.append({
                            "path": file_path,
                            "metadata": metadata
                        })
                    except Exception as e:
                        logger.warning(f"Could not get metadata for {file_path}: {e}")
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            raise StorageError(f"Failed to list datasets: {e}")


def create_default_data_sources() -> Dict[str, Union[YahooFinanceSource, FREDSource, CSVSource]]:
    """Create default data sources with standard configurations."""
    sources: Dict[str, Union[YahooFinanceSource, FREDSource, CSVSource]] = {}
    
    # Yahoo Finance source
    yahoo_config = DataSourceConfig(
        name="yahoo",
        base_url=None,
        api_key=None,
        rate_limit=60,  # 60 requests per minute
        timeout=30
    )
    sources["yahoo"] = YahooFinanceSource(yahoo_config)
    
    # FRED source (requires API key)
    fred_config = DataSourceConfig(
        name="fred",
        base_url="https://api.stlouisfed.org",
        api_key=None,  # Will be set when needed
        rate_limit=120,  # 120 requests per minute
        timeout=30
    )
    sources["fred"] = FREDSource(fred_config)
    
    # CSV source
    csv_config = DataSourceConfig(
        name="csv",
        base_url=None,
        api_key=None,
        rate_limit=None,
        timeout=30
    )
    sources["csv"] = CSVSource(csv_config)
    
    return sources


def create_default_asset_classes() -> List[AssetClass]:
    """Create default asset classes for common market data."""
    return [
        AssetClass(
            name="stocks",
            symbol="^GSPC",  # S&P 500
            source="yahoo",
            description="S&P 500 Total Return"
        ),
        AssetClass(
            name="bonds",
            symbol="^TNX",  # 10-Year Treasury
            source="yahoo",
            description="10-Year Treasury Bond"
        ),
        AssetClass(
            name="cash",
            symbol="DGS3MO",  # 3-Month Treasury
            source="fred",
            description="3-Month Treasury Bill Rate"
        ),
        AssetClass(
            name="reits",
            symbol="^RMZ",  # REIT Index
            source="yahoo",
            description="REIT Index"
        )
    ]
