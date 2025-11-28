#!/bin/bash
# Complete deployment script for LLM Context SDK

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================================"
echo "LLM Context SDK - Kubernetes Deployment Script"
echo "================================================"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed or not in PATH"
    exit 1
fi

# Function to check if using minikube
is_minikube() {
    kubectl config current-context | grep -q minikube
}

echo "Step 1: Building Docker image..."
echo "================================="
cd "$PROJECT_ROOT"

if is_minikube; then
    echo "Detected Minikube - using Minikube Docker daemon"
    eval $(minikube docker-env)
fi

docker build -t llm-context-sdk:latest .
echo "✓ Docker image built successfully"
echo ""

echo "Step 2: Generating ConfigMap..."
echo "================================"
bash "$SCRIPT_DIR/generate-configmap.sh"
echo ""

echo "Step 3: Deploying to Kubernetes..."
echo "==================================="
kubectl apply -f "$SCRIPT_DIR/namespace.yaml"
kubectl apply -f "$SCRIPT_DIR/configmap.yaml"
kubectl apply -f "$SCRIPT_DIR/deployment.yaml"
kubectl apply -f "$SCRIPT_DIR/service.yaml"
kubectl apply -f "$SCRIPT_DIR/ingress.yaml"
echo "✓ All resources deployed"
echo ""

echo "Step 4: Waiting for deployment to be ready..."
echo "=============================================="
kubectl wait --for=condition=available --timeout=300s \
    deployment/llm-context-sdk -n llm-context-sdk
echo "✓ Deployment is ready"
echo ""

echo "================================================"
echo "Deployment completed successfully!"
echo "================================================"
echo ""
echo "To check the status:"
echo "  kubectl get pods -n llm-context-sdk"
echo ""
echo "To view logs:"
echo "  kubectl logs -n llm-context-sdk -l app=llm-context-sdk -f"
echo ""
echo "To access the service:"
if is_minikube; then
    echo "  minikube service llm-context-sdk-nodeport -n llm-context-sdk"
else
    echo "  kubectl port-forward -n llm-context-sdk svc/llm-context-sdk 8000:8000"
    echo "  Then visit: http://localhost:8000"
fi
echo ""
