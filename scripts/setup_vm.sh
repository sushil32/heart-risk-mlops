#!/usr/bin/env bash
# Quick automated deployment script for Prayogshala / Linux VM
# Usage:
#   git clone https://github.com/sushil32/heart-risk-mlops.git
#   cd heart-risk-mlops
#   bash scripts/setup_vm.sh
#
set -euo pipefail

echo "=================================================="
echo "Starting Heart Risk MLOps Deployment on VM"
echo "=================================================="

# Detect Docker
if ! command -v docker &> /dev/null; then
    echo "Docker not found! Attempting to install Docker..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y docker.io docker-compose-plugin
        sudo usermod -aG docker "$USER" || true
    else
        echo "Please ensure docker is installed and running."
        exit 1
    fi
fi

# Ensure docker service is running
if command -v systemctl &> /dev/null; then
    sudo systemctl start docker || true
fi

echo "==> Building and starting services with Docker Compose..."
if docker compose version &> /dev/null; then
    docker compose down || true
    docker compose up -d --build
elif command -v docker-compose &> /dev/null; then
    docker-compose down || true
    docker-compose up -d --build
else
    echo "docker compose not found, building single Docker container..."
    docker build -t heart-risk-api:1.0.0 .
    docker run -d --name heart-risk-api --restart unless-stopped -p 8000:8000 heart-risk-api:1.0.0
fi

echo "==> Waiting for API service to become healthy on port 8000..."
for i in {1..40}; do
    if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "API is UP and healthy!"
        break
    fi
    echo -n "."
    sleep 2
done
echo ""

echo "==> Health Check Response:"
curl -s http://127.0.0.1:8000/health | jq . 2>/dev/null || curl -s http://127.0.0.1:8000/health
echo ""

echo "==> Running High-Risk Patient Sample Inference:"
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @samples/patient_high_risk.json | jq . 2>/dev/null || curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @samples/patient_high_risk.json
echo ""

echo "==> Running Low-Risk Patient Sample Inference:"
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @samples/patient_low_risk.json | jq . 2>/dev/null || curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @samples/patient_low_risk.json
echo ""

echo "=================================================="
echo "Deployment and verification completed successfully!"
echo "Services available:"
echo "  - FastAPI & Swagger Docs : http://localhost:8000/docs"
echo "  - Prometheus Metrics     : http://localhost:9090"
echo "  - Grafana Dashboard      : http://localhost:3000 (admin/admin)"
echo "=================================================="
