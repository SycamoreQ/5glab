import ray
import yaml
import time
import asyncio


def load_config(config_path='utils/config/cluster_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_ray_connection(config):
    """Test 1: Ray cluster connectivity"""
    print("\n" + "="*80)
    print("TEST 1: Ray Cluster Connectivity")
    print("="*80)
    
    master_config = config['master_node']
    
    try:
        ray.init(
            address=f"ray://{master_config['host']}:{master_config['ray_port']}",
            namespace='distributed_training'
        )
        
        print("✓ Connected to Ray cluster")
        
        # Check resources
        resources = ray.cluster_resources()
        print(f"\nCluster Resources:")
        print(f"  CPUs: {resources.get('CPU', 0)}")
        print(f"  GPUs: {resources.get('GPU', 0)}")
        print(f"  Memory: {resources.get('memory', 0) / 1e9:.1f} GB")
        
        # Check nodes
        nodes = ray.nodes()
        print(f"\nConnected Nodes: {len(nodes)}")
        for i, node in enumerate(nodes, 1):
            alive = "✓" if node['Alive'] else "✗"
            print(f"  {i}. {alive} {node.get('NodeName', 'Unknown')} - "
                  f"CPU: {node['Resources'].get('CPU', 0)}, "
                  f"GPU: {node['Resources'].get('GPU', 0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


def test_coordinator_actor():
    """Test 2: Coordinator actor availability"""
    print("\n" + "="*80)
    print("TEST 2: Training Coordinator Actor")
    print("="*80)
    
    try:
        coordinator = ray.get_actor("training_coordinator")
        print("✓ Coordinator actor found")
        
        # Test coordinator methods
        status = ray.get(coordinator.get_status.remote())
        print(f"\nCoordinator Status:")
        print(f"  Workers: {len(status['workers'])}")
        print(f"  Pending jobs: {status['jobs']['pending']}")
        print(f"  Running jobs: {status['jobs']['running']}")
        print(f"  Completed jobs: {status['jobs']['completed']}")
        
        print(f"\nRegistered Workers:")
        for wid, info in status['workers'].items():
            print(f"  - {wid}: {info['status']} "
                  f"(completed: {info['jobs_completed']}, failed: {info['jobs_failed']})")
        
        return True
        
    except ValueError:
        print("✗ Coordinator actor not found")
        print("  Make sure the master node is running: ./start_master_improved.sh")
        return False
    except Exception as e:
        print(f"✗ Error accessing coordinator: {e}")
        return False


def test_simple_remote_function():
    """Test 3: Simple distributed computation"""
    print("\n" + "="*80)
    print("TEST 3: Simple Distributed Computation")
    print("="*80)
    
    try:
        @ray.remote
        def test_task(x):
            import time
            time.sleep(1)
            return x * x
        
        print("Submitting 5 test tasks...")
        start = time.time()
        futures = [test_task.remote(i) for i in range(5)]
        results = ray.get(futures)
        duration = time.time() - start
        
        print(f"✓ All tasks completed in {duration:.2f}s")
        print(f"  Results: {results}")
        
        return True
        
    except Exception as e:
        print(f"✗ Task execution failed: {e}")
        return False


def test_job_submission(coordinator):
    """Test 4: Submit a test training job"""
    print("\n" + "="*80)
    print("TEST 4: Job Submission")
    print("="*80)
    
    try:
        # Submit a very short test job
        test_query = "test query for distributed system"
        test_episodes = 5  # Very short for quick testing
        test_start_paper = "arxiv_1706.03762"
        
        print(f"Submitting test job:")
        print(f"  Query: {test_query}")
        print(f"  Episodes: {test_episodes}")
        
        job_id = ray.get(coordinator.submit_job.remote(
            test_query, test_episodes, test_start_paper
        ))
        
        print(f"✓ Job submitted: {job_id}")
        
        # Wait a bit for job to be scheduled
        time.sleep(3)
        
        # Check job status
        status_info = ray.get(coordinator.get_job_status.remote(job_id))
        print(f"  Job status: {status_info['status']}")
        
        if status_info['status'] == 'running':
            print(f"  Assigned to: {status_info['job'].worker_id}")
        
        print("\n⚠️  Note: This test job may take several minutes to complete.")
        print("  Check job status with: python3 submit_distributed_job.py --wait")
        
        return True
        
    except Exception as e:
        print(f"✗ Job submission failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_cache():
    """Test 5: Verify training cache files exist"""
    print("\n" + "="*80)
    print("TEST 5: Training Cache Files")
    print("="*80)
    
    import os
    
    cache_dir = 'training_cache'
    required_files = [
        'training_papers_1M.pkl',
        'edge_cache_1M.pkl',
        'paper_id_set_1M.pkl',
        'embeddings_1M.pkl'
    ]
    
    all_ok = True
    
    if not os.path.exists(cache_dir):
        print(f"✗ Cache directory not found: {cache_dir}")
        return False
    
    print(f"Checking cache files in {cache_dir}/")
    
    for filename in required_files:
        filepath = os.path.join(cache_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"  ✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {filename} NOT FOUND")
            all_ok = False
    
    return all_ok


def test_database_connection(config):
    """Test 6: Neo4j database connectivity"""
    print("\n" + "="*80)
    print("TEST 6: Neo4j Database Connection")
    print("="*80)
    
    try:
        from neo4j import GraphDatabase
        
        db_config = config['database_server']
        uri = f"neo4j://{db_config['host']}:{db_config['neo4j_port']}"
        
        print(f"Connecting to {uri}...")
        
        driver = GraphDatabase.driver(
            uri,
            auth=(db_config['neo4j_user'], db_config['neo4j_password'])
        )
        
        driver.verify_connectivity()
        print("✓ Database connection successful")
        
        # Test query
        with driver.session(database=db_config['database_name']) as session:
            result = session.run("MATCH (n:Paper) RETURN count(n) as count")
            count = result.single()['count']
            print(f"  Papers in database: {count:,}")
        
        driver.close()
        return True
        
    except ImportError:
        print("⚠️  neo4j package not installed")
        print("  Install with: pip install neo4j")
        return False
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("DISTRIBUTED TRAINING SYSTEM TEST SUITE")
    print("="*80)
    
    config = load_config()
    results = {}
    
    # Test 1: Ray connection
    results['ray_connection'] = test_ray_connection(config)
    
    if not results['ray_connection']:
        print("\nRay connection failed. Cannot proceed with other tests.")
        print("   Make sure the master node is running: ./start_master_improved.sh")
        return results
    
    # Test 2: Coordinator
    results['coordinator'] = test_coordinator_actor()
    
    # Test 3: Simple computation
    results['simple_task'] = test_simple_remote_function()
    
    # Test 4: Job submission (only if coordinator is available)
    if results['coordinator']:
        try:
            coordinator = ray.get_actor("training_coordinator")
            results['job_submission'] = test_job_submission(coordinator)
        except:
            results['job_submission'] = False
    else:
        results['job_submission'] = False
        print("\nSkipping job submission test (coordinator not available)")
    
    # Test 5: Training cache
    results['training_cache'] = test_training_cache()
    
    # Test 6: Database
    results['database'] = test_database_connection(config)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name.replace('_', ' ').title()}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nYour distributed training system is ready!")
        print("\nNext steps:")
        print("  1. Submit jobs: python3 submit_distributed_job.py --query 'your query' --episodes 100 --wait")
        print("  2. Submit batch: python3 submit_distributed_job.py --batch-file batch_jobs_example.yaml")
        print("  3. Monitor: http://{}:{}".format(
            config['master_node']['host'],
            config['master_node']['dashboard_port']
        ))
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nCheck the output above for details.")
        print("\nCommon issues:")
        print("  - Master not running: ./start_master_improved.sh")
        print("  - Workers not connected: ./start_worker_improved.sh <worker_name>")
        print("  - Cache files missing: See training cache preparation guide")
    print("="*80 + "\n")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test distributed training system')
    parser.add_argument('--quick', action='store_true',
                       help='Run only quick tests (skip job submission)')
    
    args = parser.parse_args()
    
    try:
        results = asyncio.run(run_all_tests())
        
        # Exit code
        exit(0 if all(results.values()) else 1)
        
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()