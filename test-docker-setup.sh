#!/bin/bash

# Test script for Docker setup
# This script verifies that the Docker environment is working correctly

set -e

echo "🐳 Testing Docker setup for retirement planner..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Build and start services
echo "📦 Building and starting services..."
docker compose up -d --build

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Test health endpoint
echo "🔍 Testing health endpoint..."
if curl -f http://localhost:5000/healthz; then
    echo "✅ Health endpoint is working"
else
    echo "❌ Health endpoint failed"
    exit 1
fi

# Test database connection
echo "🗄️ Testing database connection..."
if docker compose exec -T db psql -U retirement_user -d retirement_planner -c "SELECT status FROM health_check LIMIT 1;"; then
    echo "✅ Database connection is working"
else
    echo "❌ Database connection failed"
    exit 1
fi

# Test Redis connection
echo "🔴 Testing Redis connection..."
if docker compose exec -T redis redis-cli ping; then
    echo "✅ Redis connection is working"
else
    echo "❌ Redis connection failed"
    exit 1
fi

echo "🎉 All tests passed! Docker setup is working correctly."

# Show running containers
echo "📋 Running containers:"
docker compose ps
