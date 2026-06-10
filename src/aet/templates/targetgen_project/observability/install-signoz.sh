#!/usr/bin/env bash
# Install SigNoz using the official Docker-based installer.
# https://signoz.io/docs/install/docker/#install-signoz-using-the-install-script
#
# After this script completes:
#   UI:            http://localhost:3301
#   OTLP HTTP:     http://localhost:4318   ← use as --otel-endpoint
#   OTLP gRPC:     localhost:4317
#
# Run aet against SigNoz:
#   aet run-suite --tracking full --otel-endpoint http://localhost:4318 ...
#
# Why SigNoz instead of Jaeger + Prometheus + Grafana:
#   - Single UI for traces, metrics, and LLM observability
#   - Natively understands gen_ai.* OTel semconv attributes
#   - ClickHouse backend: fast queries on large trace volumes
#   - No manual datasource wiring needed

set -euo pipefail

# Official SigNoz install (clones their repo and runs their checked install.sh)
git clone -b main https://github.com/SigNoz/signoz.git signoz-install
cd signoz-install/deploy
sudo ./install.sh

echo ""
echo "SigNoz installed. Open http://localhost:3301 in your browser."
echo "Default credentials: admin@example.com / password"
echo ""
echo "Run aet with SigNoz tracing:"
echo "  aet run-suite --tracking full --otel-endpoint http://localhost:4318 <...>"
