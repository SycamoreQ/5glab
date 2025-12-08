#!/bin/bash
# Start RL training worker

if [ -z "$1" ]; then
    echo "Usage: ./start_rl_worker.sh [worker_name]"
    echo "Example: ./start_rl_worker.sh rl_worker_1"
    exit 1
fi

WORKER_NAME=$1
CONFIG_FILE="utils/config/cluster_config.yaml"

# Read master config
MASTER_HOST=$(yq eval '.master_node.host' $CONFIG_FILE)
MASTER_PORT=$(yq eval '.master_node.ray_port' $CONFIG_FILE)
REDIS_PASSWORD=$(yq eval '.master_node.redis_password' $CONFIG_FILE)

# Read worker config
NUM_CPUS=$(yq eval ".rl_training_workers[] | select(.name == \"$WORKER_NAME\") | .resources.CPU" $CONFIG_FILE)
NUM_GPUS=$(yq eval ".rl_training_workers[] | select(.name == \"$WORKER_NAME\") | .resources.GPU" $CONFIG_FILE)

echo "Starting RL Training Worker: $WORKER_NAME"
echo "  Master: $MASTER_HOST:$MASTER_PORT"
echo "  CPUs: $NUM_CPUS"
echo "  GPUs: $NUM_GPUS"

ray start \
    --address="$MASTER_HOST:$MASTER_PORT" \
    --redis-password=$REDIS_PASSWORD \
    --num-cpus=$NUM_CPUS \
    --num-gpus=$NUM_GPUS \
    --resources='{"rl_trainer": 1}'

echo "✓ Worker $WORKER_NAME started!"
