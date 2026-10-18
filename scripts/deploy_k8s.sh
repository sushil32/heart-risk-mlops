#!/usr/bin/env bash
# Build the image and deploy to the current kubectl context (Docker Desktop Kubernetes).
#   ./scripts/deploy_k8s.sh            build + apply + verify
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=heart-risk-api:1.0.0
echo "==> context: $(kubectl config current-context)"

echo "==> building $IMAGE"
docker build -t "$IMAGE" .

echo "==> applying manifests"
kubectl apply -f k8s/

echo "==> waiting for rollout"
kubectl -n heart-risk rollout restart deployment/heart-risk-api
kubectl -n heart-risk rollout status deployment/heart-risk-api --timeout=120s
kubectl -n heart-risk get deploy,pods,svc,hpa -o wide

URL=http://localhost:8080
echo "==> verifying $URL"
for i in $(seq 1 30); do curl -sf "$URL/health" && break; sleep 1; done
echo
curl -s -X POST "$URL/predict" -H 'Content-Type: application/json' -d @samples/patient_high_risk.json
echo
curl -s -X POST "$URL/predict" -H 'Content-Type: application/json' -d @samples/patient_low_risk.json
echo
