import ray
import yaml
import argparse
import time
from core.job_spec import create_rl_training_job, JobPriority, JobStatus


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def connect_to_cluster(config):
    """Connect to Ray cluster."""
    master_config = config['master_node']
    
    print("CONNECTING TO RAY CLUSTER: ")
    print(f"Address: {master_config['host']}:{master_config['ray_port']}")
    print("="*80 + "\n")
    
    ray.init(
        address=f"ray://{master_config['host']}:{master_config['ray_port']}",
        _redis_password=master_config['redis_password']
    )
    
    print("✓ Connected to cluster")
    print(f"✓ Cluster resources: {ray.cluster_resources()}\n")


def submit_rl_training_job(
    scheduler,
    metrics,
    query: str,
    episodes: int,
    start_paper_id: str,
    priority: str = "high",
    db_config: dict = None
):
    """Submit an RL training job to the cluster."""
    
    # Map priority string to enum
    priority_map = {
        'low': JobPriority.LOW,
        'medium': JobPriority.MEDIUM,
        'high': JobPriority.HIGH,
        'critical': JobPriority.CRITICAL
    }
    
    job_priority = priority_map.get(priority.lower(), JobPriority.HIGH)
    
    # Create job
    job = create_rl_training_job(
        episodes=episodes,
        query=query,
        start_paper_id=start_paper_id,
        priority=job_priority,
        status=JobStatus.PENDING
    )
    
    # Add database config to parameters
    if db_config:
        job.parameters['db_config'] = db_config
    
    print(f"Submitting job:")
    print(f"  Job ID: {job.job_id}")
    print(f"  Type: {job.job_type.value}")
    print(f"  Priority: {job.job_priority.value}")
    print(f"  Query: {query}")
    print(f"  Episodes: {episodes}")
    print(f"  Start paper: {start_paper_id}")
    print(f"  Resources: {job.resources.num_cpus} CPUs, {job.resources.num_gpus} GPUs")
    print()
    
    # Record submission in metrics
    ray.get(metrics.record_job_submitted.remote(job.job_id, job.job_type.value))
    
    # Submit to scheduler
    job_id = ray.get(scheduler.submit_job.remote(job))
    
    print(f"Job submitted: {job_id}\n")
    
    return job_id


def wait_for_job(scheduler, job_id, timeout=3600, poll_interval=5):
    """Wait for job to complete and return results."""
    print(f"⏳ Waiting for job {job_id} to complete...")
    print(f"   Timeout: {timeout}s | Poll interval: {poll_interval}s\n")
    
    start_time = time.time()
    
    while (time.time() - start_time) < timeout:
        try:
            # Get job status
            status = ray.get(scheduler.get_job_status.remote(job_id))
            
            if status is None:
                print(f"Job {job_id} not found")
                return None
            
            status_value = status.value if hasattr(status, 'value') else str(status)
            elapsed = time.time() - start_time
            
            print(f"[{int(elapsed)}s] Status: {status_value}")
            
            if status_value == 'completed':
                print(f"\n✓ Job completed in {elapsed:.1f}s\n")
                
                # Get results
                completed_jobs = ray.get(scheduler.get_completed_jobs.remote())
                job_data = completed_jobs.get(job_id)
                
                if job_data:
                    return job_data['result']
                else:
                    print("⚠️  Job completed but no results found")
                    return None
            
            elif status_value == 'failed':
                print(f"\nJob failed after {elapsed:.1f}s\n")
                
                # Get error details
                completed_jobs = ray.get(scheduler.get_completed_jobs.remote())
                job_data = completed_jobs.get(job_id)
                
                if job_data and job_data.get('job'):
                    print(f"Error: {job_data['job'].error}")
                
                return None
            
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            print("\n🛑 Waiting interrupted")
            return None
        except Exception as e:
            print(f"❌ Error checking status: {e}")
            time.sleep(poll_interval)
    
    print(f"\n⏰ Timeout after {timeout}s")
    return None


def print_results(results):
    """Pretty print training results."""
    if not results:
        return
    
    print("="*80)
    print("TRAINING RESULTS")
    print("="*80)
    
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Submit training job to Ray cluster')
    parser.add_argument('--config', type=str, default='config/cluster_config.yaml',
                       help='Path to cluster configuration file')
    parser.add_argument('--query', type=str, required=True,
                       help='Research query for training')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--start-paper', type=str, default='arxiv_1706.03762',
                       help='Starting paper ID')
    parser.add_argument('--priority', type=str, default='high',
                       choices=['low', 'medium', 'high', 'critical'],
                       help='Job priority')
    parser.add_argument('--wait', action='store_true',
                       help='Wait for job to complete')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Wait timeout in seconds')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Connect to cluster
    connect_to_cluster(config)
    
    # Get scheduler and metrics actors
    try:
        scheduler = ray.get_actor("scheduler")
        metrics = ray.get_actor("metrics")
    except Exception as e:
        print(f"Failed to get actors: {e}")
        print("Make sure the head node is running with initialized actors")
        ray.shutdown()
        return
    
    # Submit job
    job_id = submit_rl_training_job(
        scheduler=scheduler,
        metrics=metrics,
        query=args.query,
        episodes=args.episodes,
        start_paper_id=args.start_paper,
        priority=args.priority,
        db_config=config.get('database')
    )
    
    # Wait for completion if requested
    if args.wait:
        results = wait_for_job(scheduler, job_id, timeout=args.timeout)
        print_results(results)
    else:
        print("Job submitted. Use --wait to wait for completion.")
    
    print("Disconnecting from cluster...")
    ray.shutdown()
    print("Done\n")


if __name__ == "__main__":
    main()