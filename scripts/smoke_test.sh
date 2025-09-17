#!/bin/bash
# Smoke test script for post-deployment validation

set -e

# Get the app URL from command line argument or use default
APP_URL=${1:-"http://localhost:5000"}

echo "🧪 Running smoke tests against: $APP_URL"

# Test 1: Health check endpoint
echo "1️⃣ Testing health check endpoint..."
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$APP_URL/healthz")
if [ "$HEALTH_RESPONSE" = "200" ]; then
    echo "✅ Health check passed (HTTP $HEALTH_RESPONSE)"
else
    echo "❌ Health check failed (HTTP $HEALTH_RESPONSE)"
    exit 1
fi

# Test 2: Basic connectivity
echo "2️⃣ Testing basic connectivity..."
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$APP_URL/")
if [ "$RESPONSE" = "200" ] || [ "$RESPONSE" = "404" ]; then
    echo "✅ Basic connectivity passed (HTTP $RESPONSE)"
else
    echo "❌ Basic connectivity failed (HTTP $RESPONSE)"
    exit 1
fi

# Test 3: Check if app is responding (not just returning errors)
echo "3️⃣ Testing application response..."
RESPONSE_BODY=$(curl -s "$APP_URL/healthz")
if echo "$RESPONSE_BODY" | grep -q "healthy\|ok\|status"; then
    echo "✅ Application is responding correctly"
else
    echo "❌ Application response seems incorrect"
    echo "Response: $RESPONSE_BODY"
    exit 1
fi

echo "🎉 All smoke tests passed! The application is running correctly."
