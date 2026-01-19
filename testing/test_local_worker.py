import ray
import time

def test_local_coordinator():
    """Test the coordinator in local mode"""
    print("\n" + "="*80)
    print("TESTING LOCAL COORDINATOR")
    print("="*80 + "\n")
    
    # Connect to local Ray
    print("Connecting to Ray...")
    ray.init(address='auto', ignore_reinit_error=True, namespace='distributed_training')
    
    print("✓ Connected to Ray")
    print(f"  Resources: {ray.available_resources()}\n")
    
    # Get coordinator
    print("Getting coordinator actor...")
    try:
        coordinator = ray.get_actor("training_coordinator")
        print("✓ Coordinator found\n")
    except ValueError:
        print("✗ Coordinator not found")
        print("  Make sure you started it with: ./start_local_dev.sh")
        ray.shutdown()
        return False
    
    # Check status
    print("Checking coordinator status...")
    status = ray.get(coordinator.get_status.remote())
    
    print(f"\nCoordinator Status:")
    print(f"  Workers configured: {len(status['workers'])}")
    print(f"  Pending jobs: {status['jobs']['pending']}")
    print(f"  Running jobs: {status['jobs']['running']}")
    print(f"  Completed jobs: {status['jobs']['completed']}")
    
    print(f"\nWorkers:")
    for wid, info in status['workers'].items():
        print(f"  - {wid}: {info['status']}")
    
    # Submit a test job
    print("\n" + "="*80)
    print("SUBMITTING TEST JOB")
    print("="*80 + "\n")
    
    test_query = "test distributed system"
    test_episodes = 5
    test_paper = "arxiv_1706.03762"
    
    print(f"Submitting job:")
    print(f"  Query: {test_query}")
    print(f"  Episodes: {test_episodes}")
    print(f"  Start paper: {test_paper}")
    print()
    
    try:
        job_id = ray.get(coordinator.submit_job.remote(
            test_query, test_episodes, test_paper
        ))
        print(f"✓ Job submitted: {job_id}\n")
    except Exception as e:
        print(f"✗ Failed to submit job: {e}")
        ray.shutdown()
        return False
    
    # Check job status
    print("Checking job status...")
    time.sleep(2)
    
    job_status = ray.get(coordinator.get_job_status.remote(job_id))
    print(f"  Status: {job_status['status']}\n")
    
    # Note about workers
    if job_status['status'] == 'pending':
        print("⚠️  Note: Job is pending because no actual workers are connected")
        print("   This is expected in local development mode")
        print("   In production, workers would pick up this job\n")
    
    print("="*80)
    print("✓ LOCAL TEST COMPLETE")
    print("="*80 + "\n")
    
    print("The coordinator is working correctly!")
    print("\nTo run distributed training:")
    print("  1. Deploy to your Linux servers")
    print("  2. Run ./start_cluster.sh master on 192.168.1.10")
    print("  3. Run ./start_cluster.sh worker ... on each worker")
    print("  4. Submit jobs with submit_distributed_job.py\n")
    
    ray.shutdown()
    return True


if __name__ == "__main__":
    test_local_coordinator()