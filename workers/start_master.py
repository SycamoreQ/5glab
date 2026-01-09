import ray
import yaml
import argparse
import time
import asyncio
from pathlib import Path
from core.worker_registry import WorkerRegistry
from core.scheduler import FIFOScheduler
from core.metrics_collector import MetricsCollector


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def start_head_node(config):
    master_config = config['master_node']
    
    print("STARTING RAY HEAD NODE;")
    print(f"Host: {master_config['host']}")
    print(f"Port: {master_config['ray_port']}")
    print(f"Dashboard: http://{master_config['host']}:{master_config['dashboard_port']}")
    print("="*80 + "\n")
    
    ray.init(
        address='auto',
        _redis_password=master_config['redis_password'],
        dashboard_host='0.0.0.0',
        dashboard_port=master_config['dashboard_port'],
        include_dashboard=True,
        _system_config={
            "object_spilling_config": json.dumps({
                "type": "filesystem",
                "params": {"directory_path": "/tmp/ray_spill"}
            })
        }
    )
    
    print("✓ Ray head node started")
    print(f"✓ Cluster resources: {ray.cluster_resources()}\n")
    
    return ray


def initialize_actors(config):
    """Initialize Ray actors for scheduling and monitoring."""
    print("Initializing Ray actors...")
    
    # Create worker registry
    worker_registry = WorkerRegistry.options(
        name="worker_registry",
        lifetime="detached",
        max_restarts=-1
    ).remote(
        heartbeat_timeout=config['system']['heartbeat_timeout']
    )
    
    # Create scheduler
    scheduler = FIFOScheduler.options(
        name="scheduler",
        lifetime="detached",
        max_restarts=-1
    ).remote(worker_registry)
    
    # Create metrics collector
    metrics = MetricsCollector.options(
        name="metrics",
        lifetime="detached",
        max_restarts=-1
    ).remote()
    
    print("✓ Worker Registry initialized")
    print("✓ Scheduler initialized")
    print("✓ Metrics Collector initialized\n")
    
    return worker_registry, scheduler, metrics


def register_workers_from_config(worker_registry, config):
    """Register worker nodes from configuration."""
    print("Registering workers from config...\n")
    
    # Register head node as worker
    head_config = config['head_node']
    ray.get(worker_registry.register_worker.remote(
        worker_id="head_node",
        capabilities={
            'node_type': head_config['type'],
            'cpu_count': head_config['resources']['cpu'],
            'has_gpu': head_config['resources'].get('GPU', 0) > 0,
            'ram_total_gb': head_config['resources']['memory'] / 1e9,
        }
    ))
    
    # Register worker nodes
    for worker_config in config['worker_nodes']:
        ray.get(worker_registry.register_worker.remote(
            worker_id=worker_config['name'],
            capabilities={
                'node_type': worker_config['type'],
                'cpu_count': worker_config['resources']['CPU'],
                'has_gpu': worker_config['resources'].get('GPU', 0) > 0,
                'ram_total_gb': worker_config['resources']['memory'] / 1e9,
            }
        ))
    
    print("✓ All workers registered\n")


async def run_monitoring_loop(worker_registry, scheduler, metrics, interval=5.0):
    """Run health monitoring and stats reporting loop."""
    print("Starting monitoring loop...\n")
    
    while True:
        try:
            health = ray.get(worker_registry.check_worker_health.remote())
            
            # Get stats
            stats = ray.get(scheduler.get_queue_stats.remote())
            worker_stats = ray.get(worker_registry.get_worker_stats.remote())
            
            # Print status
            print(f"[{time.strftime('%H:%M:%S')}] Status:")
            print(f"  Workers: {worker_stats['healthy_workers']}/{worker_stats['total_workers']} healthy")
            print(f"  Jobs: {stats['running']} running, {stats['queued']} queued, {stats['completed']} completed")
            
            # Check for offline workers
            offline = [wid for wid, status in health.items() if status.value == 'offline']
            if offline:
                print(f"Offline workers: {', '.join(offline)}")
            
            print()
            
            await asyncio.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nMonitoring interrupted")
            break
        except Exception as e:
            print(f"Monitoring error: {e}")
            await asyncio.sleep(interval)


async def run_scheduling_loop(scheduler, duration_sec=None):
    """Run job scheduling loop."""
    print("Starting scheduling loop...")
    print(f"Duration: {'infinite' if duration_sec is None else f'{duration_sec}s'}\n")
    
    start_time = time.time()
    
    try:
        while True:
            # Schedule next job
            ray.get(scheduler.schedule_next_job.remote())
            
            # Check completed jobs
            ray.get(scheduler.check_job_completion.remote())
            
            # Check if duration exceeded
            if duration_sec and (time.time() - start_time) >= duration_sec:
                print("\n✓ Scheduling loop duration complete\n")
                break
            
            await asyncio.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nScheduling interrupted")


def print_final_report(metrics, worker_registry, scheduler):
    print("FINAL SYSTEM REPORT:")
    
    try:
        ray.get(metrics.print_kpi_report.remote(worker_registry))
        
        stats = ray.get(scheduler.get_queue_stats.remote())
        print("\nFinal Job Statistics:")
        print(f"  Queued:    {stats['queued']}")
        print(f"  Running:   {stats['running']}")
        print(f"  Completed: {stats['completed']}")
        print(f"  Failed:    {stats['failed']}")
        
        worker_stats = ray.get(worker_registry.get_worker_stats.remote())
        print(f"\n👷 Worker Statistics:")
        print(f"  Total Workers:     {worker_stats['total_workers']}")
        print(f"  Healthy Workers:   {worker_stats['healthy_workers']}")
        print(f"  Jobs Completed:    {worker_stats['total_jobs_completed']}")
        print(f"  Jobs Failed:       {worker_stats['total_jobs_failed']}")
        print(f"  Success Rate:      {worker_stats['success_rate']:.1%}")
        
    except Exception as e:
        print(f"Error generating report: {e}")
    
    print("="*80 + "\n")


async def main():
    parser = argparse.ArgumentParser(description='Start Ray head node for distributed training')
    parser.add_argument('--config', type=str, default='config/cluster_config.yaml',
                       help='Path to cluster configuration file')
    parser.add_argument('--duration', type=int, default=None,
                       help='Run duration in seconds (default: run indefinitely)')
    parser.add_argument('--no-monitoring', action='store_true',
                       help='Disable monitoring loop')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Start head node
    start_head_node(config)
    
    # Initialize actors
    worker_registry, scheduler, metrics = initialize_actors(config)
    
    # Register workers
    register_workers_from_config(worker_registry, config)
    
    # Wait for workers to connect
    print("Waiting for workers to connect (10s)")
    await asyncio.sleep(10)
    
    # Start loops
    if args.no_monitoring:
        await run_scheduling_loop(scheduler, args.duration)
    else:
        # Run both loops concurrently
        await asyncio.gather(
            run_scheduling_loop(scheduler, args.duration),
            run_monitoring_loop(worker_registry, scheduler, metrics, interval=10.0)
        )
    
    # Print final report
    print_final_report(metrics, worker_registry, scheduler)
    
    print("\nShutting down...")
    ray.shutdown()
    print("✓ Shutdown complete\n")


if __name__ == "__main__":
    import json
    asyncio.run(main())