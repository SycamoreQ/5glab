#!/bin/bash
# Improved master node startup script

set -e

CONFIG_FILE="${1:-utils/config/cluster_config.yaml}"

echo "=================================================="
echo "  Starting Distributed Training Master Node"
echo "=================================================="
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract config values using Python
MASTER_HOST=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['master_node']['host'])")
RAY_PORT=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['master_node']['ray_port'])")
DASHBOARD_PORT=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['master_node']['dashboard_port'])")

echo "Configuration:"
echo "  Host: $MASTER_HOST"
echo "  Ray Port: $RAY_PORT"
echo "  Dashboard: $DASHBOARD_PORT"
echo ""

# Check if Ray is already running
if ray status > /dev/null 2>&1; then
    echo "Ray is already running"
    read -p "Clean up and restart? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Running cleanup..."
        ./cleanup_ray.sh 2>/dev/null || {
            echo "Cleanup script not found, cleaning manually..."
            ray stop
            pkill -9 -f ray:: 2>/dev/null || true
            pkill -9 -f raylet 2>/dev/null || true
            rm -rf /tmp/ray/* 2>/dev/null || true
        }
        sleep 3
    else
        echo "Exiting"
        exit 1
    fi
fi

echo "Starting Ray head node..."

# Kill any existing Ray processes first
pkill -9 -f ray:: 2>/dev/null || true
pkill -9 -f raylet 2>/dev/null || true
pkill -9 -f gcs_server 2>/dev/null || true
sleep 1

# Clean up Ray temp directory
rm -rf /tmp/ray/* 2>/dev/null || true

ray start --head \
    --port=$RAY_PORT \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$DASHBOARD_PORT \
    --num-cpus=2 \
    --num-gpus=0 \
    --include-dashboard=true \
    --disable-usage-stats

# Check if Ray started successfully
sleep 2
if ! ray status > /dev/null 2>&1; then
    echo "Failed to start Ray head node"
    echo "Check logs at: /tmp/ray/session_latest/logs/raylet.out"
    exit 1
fi

echo ""
echo "✓ Ray head node started successfully"
echo "  Dashboard: http://$MASTER_HOST:$DASHBOARD_PORT"
echo ""

# Start the coordinator
echo "Starting training coordinator..."
python3 distribute/workers/distributed_master.py --config "$CONFIG_FILE" &

COORDINATOR_PID=$!
echo "✓ Coordinator started (PID: $COORDINATOR_PID)"
echo ""

echo "=================================================="
echo "  Master Node Running"
echo "=================================================="
echo ""
echo "Dashboard: http://$MASTER_HOST:$DASHBOARD_PORT"
echo "Ray Status: ray status"
echo "Stop: ray stop"
echo ""
echo "Press Ctrl+C to stop the coordinator"
echo "=================================================="
echo ""

# Wait for coordinator
wait $COORDINATOR_PID