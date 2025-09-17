"""
Tests for historical data acquisition and management.

This module tests the historical data system including data sources,
validation, storage, and the historical returns provider.
"""

import csv
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from app.models.historical_data import (
    AssetClass,
    CSVSource,
    DataSourceConfig,
    FREDSource,
    HistoricalDataManager,
    HistoricalDataPoint,
    HistoricalDataSet,
    YahooFinanceSource,
    create_default_asset_classes,
    create_default_data_sources,
)
from app.models.historical_returns_provider import (
    HistoricalReturnsProvider,
    create_historical_returns_provider,
)
from app.storage.base import StorageService, StorageError


class TestAssetClass:
    """Test AssetClass model."""
    
    def test_asset_class_creation(self):
        """Test creating an asset class."""
        asset = AssetClass(
            name="stocks",
            symbol="^GSPC",
            source="yahoo",
            description="S&P 500"
        )
        
        assert asset.name == "stocks"
        assert asset.symbol == "^GSPC"
        assert asset.source == "yahoo"
        assert asset.description == "S&P 500"
    
    def test_asset_class_validation(self):
        """Test asset class validation."""
        # Valid asset class
        asset = AssetClass(name="bonds", symbol="^TNX", source="yahoo")
        assert asset.name == "bonds"
        
        # Missing required fields should raise ValidationError
        with pytest.raises(ValidationError):
            AssetClass(name="stocks")  # Missing symbol and source


class TestHistoricalDataPoint:
    """Test HistoricalDataPoint model."""
    
    def test_data_point_creation(self):
        """Test creating a historical data point."""
        date = datetime(2023, 1, 1)
        point = HistoricalDataPoint(
            date=date,
            value=100.0,
            asset_class="stocks",
            source="yahoo"
        )
        
        assert point.date == date
        assert point.value == 100.0
        assert point.asset_class == "stocks"
        assert point.source == "yahoo"


class TestHistoricalDataSet:
    """Test HistoricalDataSet model."""
    
    def test_dataset_creation(self):
        """Test creating a historical dataset."""
        asset = AssetClass(name="stocks", symbol="^GSPC", source="yahoo")
        
        data_points = [
            HistoricalDataPoint(
                date=datetime(2023, 1, 1),
                value=100.0,
                asset_class="stocks",
                source="yahoo"
            ),
            HistoricalDataPoint(
                date=datetime(2023, 2, 1),
                value=105.0,
                asset_class="stocks",
                source="yahoo"
            ),
        ]
        
        dataset = HistoricalDataSet(
            asset_class=asset,
            data_points=data_points,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 2, 1)
        )
        
        assert len(dataset.data_points) == 2
        assert dataset.start_date == datetime(2023, 1, 1)
        assert dataset.end_date == datetime(2023, 2, 1)
    
    def test_dataset_validation_sorts_data_points(self):
        """Test that dataset validation sorts data points by date."""
        asset = AssetClass(name="stocks", symbol="^GSPC", source="yahoo")
        
        # Create data points out of order
        data_points = [
            HistoricalDataPoint(
                date=datetime(2023, 2, 1),
                value=105.0,
                asset_class="stocks",
                source="yahoo"
            ),
            HistoricalDataPoint(
                date=datetime(2023, 1, 1),
                value=100.0,
                asset_class="stocks",
                source="yahoo"
            ),
        ]
        
        dataset = HistoricalDataSet(
            asset_class=asset,
            data_points=data_points,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 2, 1)
        )
        
        # Data points should be sorted by date
        assert dataset.data_points[0].date == datetime(2023, 1, 1)
        assert dataset.data_points[1].date == datetime(2023, 2, 1)
    
    def test_dataset_validation_empty_data_points(self):
        """Test that empty data points raise validation error."""
        asset = AssetClass(name="stocks", symbol="^GSPC", source="yahoo")
        
        with pytest.raises(ValidationError):
            HistoricalDataSet(
                asset_class=asset,
                data_points=[],
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 2, 1)
            )
    
    def test_get_returns(self):
        """Test calculating returns from data points."""
        asset = AssetClass(name="stocks", symbol="^GSPC", source="yahoo")
        
        data_points = [
            HistoricalDataPoint(
                date=datetime(2023, 1, 1),
                value=100.0,
                asset_class="stocks",
                source="yahoo"
            ),
            HistoricalDataPoint(
                date=datetime(2023, 2, 1),
                value=105.0,
                asset_class="stocks",
                source="yahoo"
            ),
            HistoricalDataPoint(
                date=datetime(2023, 3, 1),
                value=110.0,
                asset_class="stocks",
                source="yahoo"
            ),
        ]
        
        dataset = HistoricalDataSet(
            asset_class=asset,
            data_points=data_points,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 1)
        )
        
        returns = dataset.get_returns()
        assert len(returns) == 2
        assert abs(returns[0] - 0.05) < 1e-6  # 5% return
        assert abs(returns[1] - 0.047619) < 1e-6  # ~4.76% return
    
    def test_get_statistics(self):
        """Test calculating statistics from data points."""
        asset = AssetClass(name="stocks", symbol="^GSPC", source="yahoo")
        
        # Create data with known statistics
        data_points = []
        base_value = 100.0
        for i in range(12):  # 12 months
            date = datetime(2023, 1, 1) + timedelta(days=30 * i)
            value = base_value * (1.01 ** i)  # 1% monthly return
            data_points.append(HistoricalDataPoint(
                date=date,
                value=value,
                asset_class="stocks",
                source="yahoo"
            ))
        
        dataset = HistoricalDataSet(
            asset_class=asset,
            data_points=data_points,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 1)
        )
        
        stats = dataset.get_statistics()
        
        assert "mean_return" in stats
        assert "volatility" in stats
        assert "total_return" in stats
        assert "num_observations" in stats
        assert stats["num_observations"] == 11  # 11 returns from 12 data points


class TestDataSourceConfig:
    """Test DataSourceConfig model."""
    
    def test_config_creation(self):
        """Test creating a data source config."""
        config = DataSourceConfig(
            name="test_source",
            base_url="https://api.example.com",
            api_key="test_key",
            rate_limit=60,
            timeout=30
        )
        
        assert config.name == "test_source"
        assert config.base_url == "https://api.example.com"
        assert config.api_key == "test_key"
        assert config.rate_limit == 60
        assert config.timeout == 30


class TestCSVSource:
    """Test CSV data source."""
    
    def test_csv_source_creation(self):
        """Test creating a CSV source."""
        config = DataSourceConfig(name="csv")
        source = CSVSource(config)
        assert source.config.name == "csv"
    
    def test_csv_source_fetch_data(self):
        """Test fetching data from CSV file."""
        config = DataSourceConfig(name="csv")
        source = CSVSource(config)
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'value', 'asset_class'])
            writer.writerow(['2023-01-01', '100.0', 'stocks'])
            writer.writerow(['2023-02-01', '105.0', 'stocks'])
            writer.writerow(['2023-03-01', '110.0', 'stocks'])
            temp_file = f.name
        
        try:
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 3, 1)
            
            data_points = source.fetch_data(temp_file, start_date, end_date)
            
            assert len(data_points) == 3
            assert data_points[0].value == 100.0
            assert data_points[1].value == 105.0
            assert data_points[2].value == 110.0
            
        finally:
            Path(temp_file).unlink()
    
    def test_csv_source_file_not_found(self):
        """Test CSV source with non-existent file."""
        config = DataSourceConfig(name="csv")
        source = CSVSource(config)
        
        with pytest.raises(StorageError):
            source.fetch_data("nonexistent.csv", datetime(2023, 1, 1), datetime(2023, 2, 1))


class TestYahooFinanceSource:
    """Test Yahoo Finance data source."""
    
    def test_yahoo_source_creation(self):
        """Test creating a Yahoo Finance source."""
        config = DataSourceConfig(name="yahoo", timeout=30)
        source = YahooFinanceSource(config)
        assert source.config.name == "yahoo"
    
    @patch('requests.Session.get')
    def test_yahoo_source_fetch_data(self, mock_get):
        """Test fetching data from Yahoo Finance."""
        config = DataSourceConfig(name="yahoo", timeout=30)
        source = YahooFinanceSource(config)
        
        # Mock response
        mock_response = Mock()
        mock_response.text = """Date,Open,High,Low,Close,Adj Close,Volume
2023-01-01,100.0,105.0,95.0,102.0,102.0,1000000
2023-02-01,102.0,108.0,98.0,105.0,105.0,1100000"""
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 2, 1)
        
        data_points = source.fetch_data("^GSPC", start_date, end_date)
        
        assert len(data_points) == 2
        assert data_points[0].value == 102.0
        assert data_points[1].value == 105.0
        assert all(point.source == "yahoo" for point in data_points)


class TestFREDSource:
    """Test FRED data source."""
    
    def test_fred_source_creation(self):
        """Test creating a FRED source."""
        config = DataSourceConfig(name="fred", api_key="test_key")
        source = FREDSource(config)
        assert source.config.name == "fred"
    
    @patch('requests.Session.get')
    def test_fred_source_fetch_data(self, mock_get):
        """Test fetching data from FRED."""
        config = DataSourceConfig(name="fred", api_key="test_key")
        source = FREDSource(config)
        
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "observations": [
                {"date": "2023-01-01", "value": "4.5"},
                {"date": "2023-02-01", "value": "4.7"},
                {"date": "2023-03-01", "value": "."}  # Missing value
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 3, 1)
        
        data_points = source.fetch_data("DGS3MO", start_date, end_date)
        
        # Should skip the missing value
        assert len(data_points) == 2
        assert data_points[0].value == 4.5
        assert data_points[1].value == 4.7
        assert all(point.source == "fred" for point in data_points)
    
    def test_fred_source_no_api_key(self):
        """Test FRED source without API key."""
        config = DataSourceConfig(name="fred")
        source = FREDSource(config)
        
        with pytest.raises(StorageError):
            source.fetch_data("DGS3MO", datetime(2023, 1, 1), datetime(2023, 2, 1))


class TestHistoricalDataManager:
    """Test HistoricalDataManager."""
    
    def test_manager_creation(self):
        """Test creating a data manager."""
        mock_storage = Mock(spec=StorageService)
        manager = HistoricalDataManager(mock_storage)
        
        assert manager.storage_service == mock_storage
        assert manager.sources == {}
        assert manager.datasets == {}
    
    def test_add_data_source(self):
        """Test adding a data source."""
        mock_storage = Mock(spec=StorageService)
        manager = HistoricalDataManager(mock_storage)
        
        config = DataSourceConfig(name="csv")
        source = CSVSource(config)
        
        manager.add_data_source("csv", source)
        
        assert "csv" in manager.sources
        assert manager.sources["csv"] == source
    
    def test_acquire_data(self):
        """Test acquiring data."""
        mock_storage = Mock(spec=StorageService)
        manager = HistoricalDataManager(mock_storage)
        
        # Add CSV source
        config = DataSourceConfig(name="csv")
        source = CSVSource(config)
        manager.add_data_source("csv", source)
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'value', 'asset_class'])
            writer.writerow(['2023-01-01', '100.0', 'stocks'])
            writer.writerow(['2023-02-01', '105.0', 'stocks'])
            temp_file = f.name
        
        try:
            asset = AssetClass(name="stocks", symbol=temp_file, source="csv")
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 2, 1)
            
            dataset = manager.acquire_data(asset, start_date, end_date, "csv")
            
            assert len(dataset.data_points) == 2
            assert dataset.asset_class.name == "stocks"
            assert "stocks" in manager.datasets
            
        finally:
            Path(temp_file).unlink()
    
    def test_acquire_data_unknown_source(self):
        """Test acquiring data with unknown source."""
        mock_storage = Mock(spec=StorageService)
        manager = HistoricalDataManager(mock_storage)
        
        asset = AssetClass(name="stocks", symbol="^GSPC", source="yahoo")
        
        with pytest.raises(StorageError):
            manager.acquire_data(asset, datetime(2023, 1, 1), datetime(2023, 2, 1), "unknown")
    
    def test_validate_data_integrity(self):
        """Test data integrity validation."""
        mock_storage = Mock(spec=StorageService)
        manager = HistoricalDataManager(mock_storage)
        
        asset = AssetClass(name="stocks", symbol="^GSPC", source="yahoo")
        
        # Create dataset with sufficient data
        data_points = []
        for i in range(250):  # More than 20 years of monthly data
            date = datetime(2000, 1, 1) + timedelta(days=30 * i)
            value = 100.0 * (1.01 ** i)
            data_points.append(HistoricalDataPoint(
                date=date,
                value=value,
                asset_class="stocks",
                source="yahoo"
            ))
        
        dataset = HistoricalDataSet(
            asset_class=asset,
            data_points=data_points,
            start_date=datetime(2000, 1, 1),
            end_date=datetime(2020, 1, 1)
        )
        
        validation = manager.validate_data_integrity(dataset)
        
        assert validation["is_valid"] is True
        assert len(validation["issues"]) == 0
        assert "statistics" in validation
    
    def test_validate_data_integrity_insufficient_data(self):
        """Test data integrity validation with insufficient data."""
        mock_storage = Mock(spec=StorageService)
        manager = HistoricalDataManager(mock_storage)
        
        asset = AssetClass(name="stocks", symbol="^GSPC", source="yahoo")
        
        # Create dataset with insufficient data
        data_points = [
            HistoricalDataPoint(
                date=datetime(2023, 1, 1),
                value=100.0,
                asset_class="stocks",
                source="yahoo"
            )
        ]
        
        dataset = HistoricalDataSet(
            asset_class=asset,
            data_points=data_points,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 1)
        )
        
        validation = manager.validate_data_integrity(dataset)
        
        assert validation["is_valid"] is False
        assert any("Insufficient data" in issue for issue in validation["issues"])
    
    def test_store_and_load_dataset(self):
        """Test storing and loading a dataset."""
        mock_storage = Mock(spec=StorageService)
        manager = HistoricalDataManager(mock_storage)
        
        asset = AssetClass(name="stocks", symbol="^GSPC", source="yahoo")
        data_points = [
            HistoricalDataPoint(
                date=datetime(2023, 1, 1),
                value=100.0,
                asset_class="stocks",
                source="yahoo"
            )
        ]
        
        dataset = HistoricalDataSet(
            asset_class=asset,
            data_points=data_points,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 1)
        )
        
        # Mock storage operations
        mock_storage.store_file.return_value = "historical_data/stocks_20230101_20230101.json"
        mock_storage.retrieve_file.return_value = dataset.model_dump_json().encode('utf-8')
        
        # Store dataset
        storage_path = manager.store_dataset(dataset)
        assert storage_path == "historical_data/stocks_20230101_20230101.json"
        mock_storage.store_file.assert_called_once()
        
        # Load dataset
        loaded_dataset = manager.load_dataset("test_path")
        assert loaded_dataset.asset_class.name == "stocks"
        assert len(loaded_dataset.data_points) == 1


class TestHistoricalReturnsProvider:
    """Test HistoricalReturnsProvider."""
    
    def test_provider_creation_with_mock_data(self):
        """Test creating a historical returns provider with mock data."""
        mock_data_manager = Mock()
        mock_data_manager.list_available_datasets.return_value = []
        
        # Mock dataset with returns
        mock_dataset = Mock()
        mock_dataset.get_returns.return_value = [0.01, 0.02, -0.01, 0.03]
        mock_dataset.get_statistics.return_value = {"mean_return": 0.08}
        
        provider = HistoricalReturnsProvider(
            data_manager=mock_data_manager,
            asset_classes=["stocks"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 1, 1),
            seed=42
        )
        
        # Manually set the datasets and returns matrix for testing
        provider.datasets = {"stocks": mock_dataset}
        provider.returns_matrix = np.array([[0.01, 0.02, -0.01, 0.03]])
        
        assert provider.asset_classes == ["stocks"]
        assert provider.seed == 42
    
    def test_generate_returns(self):
        """Test generating returns."""
        mock_data_manager = Mock()
        mock_data_manager.list_available_datasets.return_value = []
        
        provider = HistoricalReturnsProvider(
            data_manager=mock_data_manager,
            asset_classes=["stocks"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 1, 1),
            seed=42
        )
        
        # Mock the returns matrix
        provider.returns_matrix = np.array([[0.01, 0.02, -0.01, 0.03]])  # 1 asset, 4 periods
        
        returns = provider.generate_returns(years=2, num_paths=3, seed=42)
        
        assert returns.shape == (1, 2, 3)  # 1 asset, 2 years, 3 paths
        assert isinstance(returns, np.ndarray)
    
    def test_generate_returns_invalid_inputs(self):
        """Test generating returns with invalid inputs."""
        mock_data_manager = Mock()
        mock_data_manager.list_available_datasets.return_value = []
        
        provider = HistoricalReturnsProvider(
            data_manager=mock_data_manager,
            asset_classes=["stocks"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 1, 1)
        )
        
        # Mock the returns matrix
        provider.returns_matrix = np.array([[0.01, 0.02, -0.01, 0.03]])
        
        with pytest.raises(ValueError):
            provider.generate_returns(years=0, num_paths=10)
        
        with pytest.raises(ValueError):
            provider.generate_returns(years=10, num_paths=0)
    
    def test_get_asset_names(self):
        """Test getting asset names."""
        mock_data_manager = Mock()
        mock_data_manager.list_available_datasets.return_value = []
        
        provider = HistoricalReturnsProvider(
            data_manager=mock_data_manager,
            asset_classes=["stocks", "bonds", "cash"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 1, 1)
        )
        
        # Mock the returns matrix
        provider.returns_matrix = np.array([[0.01, 0.02, -0.01, 0.03]])
        
        names = provider.get_asset_names()
        assert names == ["stocks", "bonds", "cash"]
    
    def test_get_target_weights(self):
        """Test getting target weights."""
        mock_data_manager = Mock()
        mock_data_manager.list_available_datasets.return_value = []
        
        provider = HistoricalReturnsProvider(
            data_manager=mock_data_manager,
            asset_classes=["stocks", "bonds"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 1, 1)
        )
        
        # Mock the returns matrix
        provider.returns_matrix = np.array([[0.01, 0.02, -0.01, 0.03]])
        
        weights = provider.get_target_weights()
        assert len(weights) == 2
        assert abs(weights[0] - 0.5) < 1e-6  # Equal weights
        assert abs(weights[1] - 0.5) < 1e-6
    
    def test_get_expected_returns(self):
        """Test getting expected returns."""
        mock_data_manager = Mock()
        mock_data_manager.list_available_datasets.return_value = []
        
        provider = HistoricalReturnsProvider(
            data_manager=mock_data_manager,
            asset_classes=["stocks"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 1, 1)
        )
        
        # Mock dataset with statistics
        mock_dataset = Mock()
        mock_dataset.get_statistics.return_value = {"mean_return": 0.08}
        provider.datasets = {"stocks": mock_dataset}
        provider.returns_matrix = np.array([[0.01, 0.02, -0.01, 0.03]])
        
        expected_returns = provider.get_expected_returns()
        assert len(expected_returns) == 1
        assert expected_returns[0] == 0.08


class TestDefaultFunctions:
    """Test default creation functions."""
    
    def test_create_default_data_sources(self):
        """Test creating default data sources."""
        sources = create_default_data_sources()
        
        assert "yahoo" in sources
        assert "fred" in sources
        assert "csv" in sources
        
        assert isinstance(sources["yahoo"], YahooFinanceSource)
        assert isinstance(sources["fred"], FREDSource)
        assert isinstance(sources["csv"], CSVSource)
    
    def test_create_default_asset_classes(self):
        """Test creating default asset classes."""
        asset_classes = create_default_asset_classes()
        
        assert len(asset_classes) == 4
        
        names = [asset.name for asset in asset_classes]
        assert "stocks" in names
        assert "bonds" in names
        assert "cash" in names
        assert "reits" in names
    
    def test_create_historical_returns_provider(self):
        """Test creating a historical returns provider with defaults."""
        mock_data_manager = Mock()
        mock_data_manager.list_available_datasets.return_value = []
        
        provider = create_historical_returns_provider(mock_data_manager)
        
        # Mock the returns matrix to avoid the ValueError
        provider.returns_matrix = np.array([[0.01, 0.02, -0.01, 0.03]])
        
        assert len(provider.asset_classes) == 4
        assert "stocks" in provider.asset_classes
        assert "bonds" in provider.asset_classes
        assert "cash" in provider.asset_classes
        assert "reits" in provider.asset_classes
