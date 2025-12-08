#!/bin/bash
# Start Ray head node (master) on database server

CONFIG_FILE="utils/config/cluster_config.yaml"

# Read config
MASTER_HOST=$(yq eval '.master_node.host' $CONFIG_FILE)
MASTER_PORT=$(yq eval '.master_node.ray_port' $CONFIG_FILE)
DASHBOARD_PORT=$(yq eval '.master_node.dashboard_port' $CONFIG_FILE)
REDIS_PASSWORD=$(yq eval '.master_node.redis_password' $CONFIG_FILE)

echo "Starting Ray Master Node..."
echo "  Host: $MASTER_HOST"
echo "  Port: $MASTER_PORT"
echo "  Dashboard: $DASHBOARD_PORT"

ray start --head \
    --port=$MASTER_PORT \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$DASHBOARD_PORT \
    --redis-password=$REDIS_PASSWORD \
    --num-cpus=2 \
    --num-gpus=0

echo "✓ Master node started!"
echo "  Dashboard: http://$MASTER_HOST:$DASHBOARD_PORT"
