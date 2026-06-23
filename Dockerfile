FROM python:3.11-slim

WORKDIR /app

# Install system dependencies:
# - build-essential / libpq-dev: required for psycopg2 compilation
# - postgresql-client: provides pg_isready used by entrypoint.sh
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Build argument: set to "true" to also install dev/test dependencies.
# Usage: docker build --build-arg INSTALL_DEV=true ...
ARG INSTALL_DEV=false
COPY requirements-dev.txt .
RUN if [ "$INSTALL_DEV" = "true" ]; then \
      pip install --no-cache-dir -r requirements-dev.txt; \
    else \
      pip install --no-cache-dir -r requirements.txt; \
    fi

COPY . .

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

# entrypoint.sh: waits for Postgres → runs alembic upgrade head → starts uvicorn
# WEB_CONCURRENCY controls the number of uvicorn worker processes (default: 1)
CMD ["/app/entrypoint.sh"]
