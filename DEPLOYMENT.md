# Deployment Guide

This document provides instructions for deploying the Retirement Planner application to various cloud platforms.

## Render Deployment (Recommended)

Render provides a modern, containerized deployment platform with built-in CI/CD, databases, and Redis caching.

### Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Ensure your code is pushed to GitHub
3. **Render CLI** (optional): For command-line deployment

### One-Click Deployment

#### Option 1: Using Render Dashboard (Recommended)

1. **Connect Repository**:
   - Go to [render.com/dashboard](https://render.com/dashboard)
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Select the repository containing this project

2. **Deploy**:
   - Render will automatically detect the `render.yaml` file
   - Click "Apply" to deploy all services
   - Wait for deployment to complete (5-10 minutes)

3. **Access Your App**:
   - Your app will be available at the URL shown in the dashboard
   - Example: `https://retirement-planner-app.onrender.com`

#### Option 2: Using Command Line

```bash
# Install Render CLI
curl -fsSL https://cli.render.com/install | sh

# Login to Render
render login

# Deploy using the provided script
./deploy.sh
```

### Environment Variables

The following environment variables are automatically configured by Render:

- `FLASK_ENV=production`
- `APP_ENV=production`
- `LOG_LEVEL=INFO`
- `STORAGE_TYPE=local`
- `DB_URL` (automatically set from PostgreSQL service)
- `REDIS_URL` (automatically set from Redis service)
- `SECRET_KEY` (automatically generated)

### Custom Environment Variables

To add custom environment variables:

1. Go to your service in the Render dashboard
2. Navigate to "Environment"
3. Add your custom variables

### Post-Deployment Testing

Run the smoke test to verify deployment:

```bash
# Test against your deployed app
./scripts/smoke_test.sh https://your-app-url.onrender.com

# Or test locally
./scripts/smoke_test.sh http://localhost:5000
```

## Alternative Platforms

### Fly.io Deployment

1. **Install Fly CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Create fly.toml**:
   ```toml
   app = "retirement-planner"
   primary_region = "ord"

   [build]

   [http_service]
     internal_port = 10000
     force_https = true

   [env]
     FLASK_ENV = "production"
     APP_ENV = "production"
   ```

3. **Deploy**:
   ```bash
   fly deploy
   ```

### Heroku Deployment

1. **Install Heroku CLI**:
   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku
   
   # Linux/Windows
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Procfile**:
   ```
   web: python wsgi.py
   ```

3. **Deploy**:
   ```bash
   heroku create your-app-name
   heroku addons:create heroku-postgresql:mini
   heroku addons:create heroku-redis:mini
   git push heroku main
   ```

## Monitoring and Maintenance

### Health Checks

The application includes health check endpoints:

- **Health Check**: `GET /healthz`
- **Status**: Returns 200 OK when healthy

### Logs

Access logs through your platform's dashboard:

- **Render**: Dashboard → Service → Logs
- **Fly.io**: `fly logs`
- **Heroku**: `heroku logs --tail`

### Scaling

#### Render
- Free tier: 0.5GB RAM, 0.1 CPU
- Paid tiers: Configurable in dashboard

#### Fly.io
- Hobby tier: $5/month for 1GB RAM, 1 CPU
- Scale: `fly scale count 2`

#### Heroku
- Free tier: 512MB RAM (discontinued)
- Paid tiers: Configurable dyno types

## Troubleshooting

### Common Issues

1. **Port Configuration**:
   - Ensure your app listens on the port specified by the `PORT` environment variable
   - Default ports: Render (10000), Fly.io (8080), Heroku (dynamic)

2. **Database Connection**:
   - Verify database URL is correctly set
   - Check database service is running

3. **Redis Connection**:
   - Verify Redis URL is correctly set
   - Check Redis service is running

4. **Build Failures**:
   - Check Dockerfile syntax
   - Verify all dependencies are in requirements.txt
   - Review build logs in platform dashboard

### Debug Mode

To enable debug mode (development only):

```bash
# Set environment variable
export FLASK_ENV=development
export APP_ENV=development

# Run locally
python wsgi.py
```

## Security Considerations

1. **Environment Variables**:
   - Never commit secrets to version control
   - Use platform-specific secret management

2. **HTTPS**:
   - All platforms provide automatic HTTPS
   - Ensure your app redirects HTTP to HTTPS

3. **Database Security**:
   - Use connection strings with proper authentication
   - Enable SSL connections when available

## Cost Optimization

### Free Tier Limits

- **Render**: 750 hours/month, spins down after inactivity
- **Fly.io**: $5 credit/month
- **Heroku**: No free tier (as of 2022)

### Optimization Tips

1. **Use appropriate instance sizes**
2. **Implement proper caching**
3. **Optimize database queries**
4. **Use CDN for static assets**
5. **Monitor resource usage**

## Support

For deployment issues:

1. Check platform-specific documentation
2. Review application logs
3. Run smoke tests
4. Contact platform support if needed
