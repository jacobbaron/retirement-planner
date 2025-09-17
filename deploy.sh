#!/bin/bash
# One-click deployment script for Render

set -e

echo "�� Starting deployment to Render..."

# Check if render.yaml exists
if [ ! -f "render.yaml" ]; then
    echo "❌ render.yaml not found. Please ensure it exists in the project root."
    exit 1
fi

# Check if user is logged into Render CLI
if ! render --version > /dev/null 2>&1; then
    echo "📦 Installing Render CLI..."
    curl -fsSL https://cli.render.com/install | sh
    echo "✅ Render CLI installed. Please run 'render login' to authenticate."
    exit 1
fi

# Deploy using Render CLI
echo "🔧 Deploying to Render..."
render deploy

echo "✅ Deployment complete!"
echo "🌐 Your app should be available at the URL shown above."
