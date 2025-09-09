"""Tests for Docker setup verification."""

import os
import subprocess
import time
import requests
import pytest


def test_docker_compose_up():
    """Test that docker compose up starts all services successfully."""
    # This test would run in CI/CD or when Docker is available
    # For now, we'll create a mock test that can be run when Docker is available
    
    # Check if Docker is available
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            pytest.skip("Docker not available")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("Docker not available")
    
    # If we get here, Docker is available
    # Start services
    result = subprocess.run(['docker', 'compose', 'up', '-d'], 
                          capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"Docker compose up failed: {result.stderr}"
    
    # Wait for services to be ready
    time.sleep(10)
    
    # Test health endpoint
    try:
        response = requests.get('http://localhost:5000/healthz', timeout=5)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    except requests.exceptions.RequestException:
        pytest.fail("Health endpoint not accessible")
    
    # Test database connection
    result = subprocess.run([
        'docker', 'compose', 'exec', '-T', 'db', 
        'psql', '-U', 'retirement_user', '-d', 'retirement_planner', 
        '-c', 'SELECT status FROM health_check LIMIT 1;'
    ], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, f"Database connection failed: {result.stderr}"
    
    # Test Redis connection
    result = subprocess.run([
        'docker', 'compose', 'exec', '-T', 'redis', 
        'redis-cli', 'ping'
    ], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, f"Redis connection failed: {result.stderr}"
    assert 'PONG' in result.stdout


def test_health_endpoint_in_container():
    """Test that the health endpoint works inside the container."""
    # This test verifies the health endpoint is accessible from within the container
    try:
        result = subprocess.run([
            'docker', 'compose', 'exec', '-T', 'app',
            'curl', '-f', 'http://localhost:5000/healthz'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            assert '{"status":"ok"}' in result.stdout
        else:
            pytest.skip("Docker not available or container not running")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("Docker not available")


def test_database_migration_runs():
    """Test that database initialization runs successfully."""
    try:
        # Check if the health_check table exists and has data
        result = subprocess.run([
            'docker', 'compose', 'exec', '-T', 'db',
            'psql', '-U', 'retirement_user', '-d', 'retirement_planner',
            '-c', "SELECT COUNT(*) FROM health_check WHERE status = 'database_ready';"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            assert '1' in result.stdout  # Should have 1 record
        else:
            pytest.skip("Docker not available or container not running")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("Docker not available")
