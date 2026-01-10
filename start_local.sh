set -e

CONFIG_FILE="${1:-utils/config/cluster_config.yaml}"

echo "=================================================="
echo "  Local Development Mode"
echo "  (Single machine simulation)"
echo "=================================================="
echo ""

# Cleanup
echo "Cleaning up Ray..."
ray stop 2>/dev/null || true
sleep 2

pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true
sleep 1

echo "Starting Ray locally..."
ray start --head \
    --num-cpus=4 \
    --num-gpus=0 \
    --include-dashboard=true \
    --dashboard-host=127.0.0.1 \
    --dashboard-port=8265 \
    --disable-usage-stats

sleep 3

if ! ray status > /dev/null 2>&1; then
    echo "Failed to start Ray"
    exit 1
fi

echo "✓ Ray started"
echo "  Dashboard: http://127.0.0.1:8265"
echo ""

echo "Starting coordinator..."
python3 RL/train_rl.py --config "$CONFIG_FILE" &

COORDINATOR_PID=$!
sleep 3

if ! ps -p $COORDINATOR_PID > /dev/null 2>&1; then
    echo "Coordinator failed to start"
    echo "Check the error above"
    ray stop
    exit 1
fi

echo "✓ Coordinator started (PID: $COORDINATOR_PID)"
echo ""

echo "=================================================="
echo "  Local Development Environment Running"
echo "=================================================="
echo ""
echo "Dashboard: http://127.0.0.1:8265"
echo ""
echo "Test with:"
echo "  python3 test_local_job.py"
echo ""
echo "Stop with:"
echo "  ray stop"
echo ""
echo "Press Ctrl+C to stop coordinator"
echo ""

trap "ray stop" EXIT
wait $COORDINATOR_PID