# Kubernetes Deployment for LLM Context SDK

This directory contains Kubernetes manifests for deploying the LLM Context SDK application.

## Prerequisites

- Kubernetes cluster (local or cloud)
- kubectl configured to communicate with your cluster
- Docker for building the container image

## Files

- `namespace.yaml` - Creates a dedicated namespace for the application
- `configmap.yaml` - Stores configuration files from `deploy/configuration/`
- `deployment.yaml` - Defines the application deployment with 2 replicas
- `service.yaml` - Exposes the application (ClusterIP and NodePort)
- `ingress.yaml` - Optional ingress for external access
- `kustomization.yaml` - Kustomize configuration for managing deployments

## Quick Start

### 1. Build the Docker Image

```bash
# From the project root directory
docker build -t llm-context-sdk:latest .
```

For Minikube, use the Minikube Docker daemon:
```bash
eval $(minikube docker-env)
docker build -t llm-context-sdk:latest .
```

### 2. Update ConfigMap with Your Configuration

Before deploying, update the `configmap.yaml` file with your actual configuration from `deploy/configuration/`:

```bash
# You can use this helper script to create the ConfigMap from your config files
kubectl create configmap llm-context-sdk-config \
  --from-file=../deploy/configuration/ \
  --dry-run=client -o yaml > configmap.yaml
```

### 3. Deploy to Kubernetes

#### Option A: Using kubectl

```bash
# Apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

#### Option B: Using Kustomize

```bash
# Deploy using kustomize
kubectl apply -k k8s/
```

### 4. Verify Deployment

```bash
# Check pods status
kubectl get pods -n llm-context-sdk

# Check service
kubectl get svc -n llm-context-sdk

# View logs
kubectl logs -n llm-context-sdk -l app=llm-context-sdk -f
```

### 5. Access the Application

#### Via NodePort (for local development):
```bash
# Get the NodePort URL
kubectl get svc llm-context-sdk-nodeport -n llm-context-sdk

# For Minikube
minikube service llm-context-sdk-nodeport -n llm-context-sdk
```

#### Via Ingress (requires ingress controller):
```bash
# Add to /etc/hosts
echo "$(minikube ip) llm-context-sdk.local" | sudo tee -a /etc/hosts

# Access at http://llm-context-sdk.local
```

#### Via Port Forward:
```bash
kubectl port-forward -n llm-context-sdk svc/llm-context-sdk 8000:8000

# Access at http://localhost:8000
```

## Testing the API

```bash
# Health check
curl http://localhost:8000/

# Generate response
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai",
    "prompt": "What is Kubernetes?",
    "images": [],
    "session_id": "test-session"
  }'
```

## Scaling

```bash
# Scale the deployment
kubectl scale deployment llm-context-sdk -n llm-context-sdk --replicas=3
```

## Updating Configuration

```bash
# Update ConfigMap
kubectl create configmap llm-context-sdk-config \
  --from-file=../deploy/configuration/ \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up new config
kubectl rollout restart deployment/llm-context-sdk -n llm-context-sdk
```

## Cleanup

```bash
# Delete all resources
kubectl delete -k k8s/

# Or delete namespace (removes everything)
kubectl delete namespace llm-context-sdk
```

## Production Considerations

1. **Configuration Management**: Update the ConfigMap with actual production configuration
2. **Resource Limits**: Adjust CPU/memory based on your workload
3. **Persistence**: Add PersistentVolumeClaims if you need persistent storage
4. **Secrets**: Move sensitive data (API keys) to Kubernetes Secrets
5. **Image Registry**: Push image to a proper registry (Docker Hub, ECR, GCR, etc.)
6. **Monitoring**: Add Prometheus/Grafana for monitoring
7. **Logging**: Configure centralized logging
8. **Security**: Implement NetworkPolicies, RBAC, and security contexts
9. **High Availability**: Use pod disruption budgets and anti-affinity rules
10. **Ingress**: Configure proper ingress with TLS/SSL certificates

## Using Secrets for Sensitive Data

Create a secret for API keys:

```bash
kubectl create secret generic llm-api-keys \
  --from-literal=openai-api-key=YOUR_KEY \
  -n llm-context-sdk
```

Then reference in deployment:
```yaml
env:
- name: OPENAI_API_KEY
  valueFrom:
    secretKeyRef:
      name: llm-api-keys
      key: openai-api-key
```
