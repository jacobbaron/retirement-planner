# Docker Development Environment

This document describes the Docker development environment for the retirement planner application.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose (included with Docker Desktop)

## Quick Start

1. **Start the development environment:**
   ```bash
   docker compose up -d
   ```

2. **Test the setup:**
   ```bash
   ./test-docker-setup.sh
   ```

3. **Access the application:**
   - Flask app: http://localhost:5000
   - Health endpoint: http://localhost:5000/healthz
   - PostgreSQL: localhost:5432
   - Redis: localhost:6379

## Services

### App Container
- **Image**: Custom Flask application
- **Port**: 5000
- **Health Check**: `/healthz` endpoint
- **Dependencies**: PostgreSQL, Redis

### Database (PostgreSQL)
- **Image**: postgres:15-alpine
- **Port**: 5432
- **Database**: retirement_planner
- **User**: retirement_user
- **Password**: retirement_pass
- **Health Check**: pg_isready

### Redis
- **Image**: redis:7-alpine
- **Port**: 6379
- **Health Check**: redis-cli ping

## Development Commands

### Start services
```bash
docker compose up -d
```

### Stop services
```bash
docker compose down
```

### View logs
```bash
docker compose logs -f app
docker compose logs -f db
docker compose logs -f redis
```

### Access container shell
```bash
docker compose exec app bash
```

### Run tests
```bash
docker compose exec app python -m pytest
```

### Run linting
```bash
docker compose exec app python -m flake8 .
```

### Database operations
```bash
# Connect to database
docker compose exec db psql -U retirement_user -d retirement_planner

# Run migrations (when available)
docker compose exec app python -m flask db upgrade
```

## Environment Variables

The following environment variables are configured:

- `FLASK_ENV=development`
- `DATABASE_URL=postgresql://retirement_user:retirement_pass@db:5432/retirement_planner`
- `REDIS_URL=redis://redis:6379/0`

## Volumes

- **App code**: Mounted from host for live development
- **PostgreSQL data**: Persistent volume `postgres_data`
- **Redis data**: Persistent volume `redis_data`

## Health Checks

All services include health checks:
- **App**: HTTP GET to `/healthz`
- **PostgreSQL**: `pg_isready` command
- **Redis**: `redis-cli ping` command

## Troubleshooting

### Services not starting
```bash
# Check service status
docker compose ps

# View logs
docker compose logs

# Restart services
docker compose restart
```

### Database connection issues
```bash
# Check database logs
docker compose logs db

# Test database connection
docker compose exec db pg_isready -U retirement_user -d retirement_planner
```

### Port conflicts
If ports 5000, 5432, or 6379 are already in use, modify the port mappings in `docker-compose.yml`:

```yaml
ports:
  - "5001:5000"  # Use port 5001 instead of 5000
```

## Agent Usage

For agents working on this project:

1. Always use `docker compose up -d` to start the development environment
2. Run all commands inside the app container: `docker compose exec app <command>`
3. Use the test script to verify setup: `./test-docker-setup.sh`
4. Check service health before starting work: `docker compose ps`
