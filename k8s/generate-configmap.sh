#!/bin/bash
# Helper script to create ConfigMap from configuration files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/deploy/configuration"
K8S_DIR="$PROJECT_ROOT/k8s"

echo "Creating ConfigMap from configuration files in: $CONFIG_DIR"

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Configuration directory not found: $CONFIG_DIR"
    exit 1
fi

# Generate ConfigMap YAML from config files
kubectl create configmap llm-context-sdk-config \
    --from-file="$CONFIG_DIR" \
    --dry-run=client -o yaml > "$K8S_DIR/configmap.yaml"

echo "✓ ConfigMap generated successfully at: $K8S_DIR/configmap.yaml"
echo ""
echo "Configuration files included:"
ls -1 "$CONFIG_DIR"
