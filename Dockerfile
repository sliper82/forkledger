# ForkLedger Docker image
# Usage:
#   docker build -t forkledger .
#   docker run -p 8000:8000 -v $(pwd)/data:/data forkledger serve \
#     --backend sqlite --store /data/store.db --host 0.0.0.0

FROM python:3.12-slim

LABEL org.opencontainers.image.title="ForkLedger"
LABEL org.opencontainers.image.description="Branch-based decision memory for AI agents"
LABEL org.opencontainers.image.source="https://github.com/sliper82/forkledger"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Install dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e ".[api]"

# Data directory (mount here for persistence)
RUN mkdir -p /data

EXPOSE 8000

ENTRYPOINT ["forkledger"]
CMD ["serve", "--backend", "sqlite", "--store", "/data/store.db", \
     "--host", "0.0.0.0", "--port", "8000"]
