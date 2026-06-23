#!/bin/sh
set -e

# Ensure the app package is importable by alembic and uvicorn.
# The WORKDIR in the Dockerfile is /app, but PYTHONPATH must be set
# explicitly when running alembic so it can resolve `from app.config...`.
export PYTHONPATH=/app

# ---------------------------------------------------------------------------
# ArchNet Backend — Container Entrypoint
#
# Execution order:
#   1. Wait for PostgreSQL to be ready (prevents migration race conditions)
#   2. Run Alembic migrations to bring schema to latest revision
#   3. Start uvicorn with the number of workers set via WEB_CONCURRENCY
# ---------------------------------------------------------------------------

# --- 1. Database readiness probe ---
# Extract host and port from DATABASE_URL for the pg_isready check.
# DATABASE_URL format: postgresql://user:password@host:port/dbname
DB_HOST=$(echo "$DATABASE_URL" | sed -E 's|.*@([^:/]+).*|\1|')
DB_PORT=$(echo "$DATABASE_URL" | sed -E 's|.*:([0-9]+)/.*|\1|')
DB_PORT=${DB_PORT:-5432}

echo "Waiting for PostgreSQL at ${DB_HOST}:${DB_PORT}..."
MAX_RETRIES=30
RETRY_INTERVAL=2
retries=0

until pg_isready -h "$DB_HOST" -p "$DB_PORT" -q; do
  retries=$((retries + 1))
  if [ "$retries" -ge "$MAX_RETRIES" ]; then
    echo "ERROR: PostgreSQL did not become ready after $((MAX_RETRIES * RETRY_INTERVAL))s. Aborting."
    exit 1
  fi
  echo "  PostgreSQL not yet ready — retrying in ${RETRY_INTERVAL}s (attempt ${retries}/${MAX_RETRIES})..."
  sleep "$RETRY_INTERVAL"
done

echo "PostgreSQL is ready."

# --- 2. Run Alembic migrations ---
echo "Running Alembic migrations..."
alembic upgrade head
echo "Migrations complete."

# --- 3. Start uvicorn ---
WORKERS=${WEB_CONCURRENCY:-1}
echo "Starting uvicorn with ${WORKERS} worker(s)..."
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers "$WORKERS"
