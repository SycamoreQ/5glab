import ray
import time
import asyncio
from core.worker_registry import WorkerRegistry
from core.scheduler import FIFOScheduler
from core.metrics_collector import MetricsCollector
from core.job_spec import JobSpec, create_rl_training_job


class MasterNode:
    
    def __init__(self): 
        ray.init(address = 'auto' , _redis_password='5G_cluster')
        print("MASTER NODE INITIALIZATION")
        print(f"Ray cluster: {ray.cluster_resources()}\n")
        self.worker_registry = WorkerRegistry.remote()
        self.scheduler = FIFOScheduler.remote(self.worker_registry)
        self.metrics = MetricsCollector.remote()
    
    def register_worker_from_config(self , config_path: str = "utils/config/cluster_config.yaml"):
        import yaml
        
        with open(config_path) as r: 
            config = yaml.safe_load(r)


        head_config = config['head_node']
        ray.get(self.worker_registry.register_worker.remote(
            worker_id= "head_node",
            capabilities = {
                'node_type' : head_config['type'],
                'cpu_count' : head_config['resources']['cpu'],
                'has_gpu': head_config['resources'].get('GPU', 0) > 0,
                'ram_total_gb': head_config['resources']['memory'] /1e-9,   
            }
        ))

        for worker_config in config['worker_nodes']:
            ray.get(self.worker_registry.register_worker.remote(
                worker_id=worker_config['name'],
                capabilities={
                    'node_type': worker_config['type'],
                    'cpu_count': worker_config['resources']['CPU'],
                    'has_gpu': worker_config['resources'].get('GPU', 0) > 0,
                    'ram_total_gb': worker_config['resources']['memory'] / 1e9,
                }
            ))


    def submit_jobs(self , job: JobSpec) -> str: 
        ray.get(self.metrics.record_job_submitted.remote(job.job_id , job.job_type.value))
        
        return ray.get(self.scheduler.submit_job.remote(job))
    

    async def run_scheduling_loop(self, duration_sec: float = 60.0):
        print("STARTING SCHEDULING LOOP")
        print(f"Duration: {duration_sec}s\n")
        
        start_time = time.time()
        
        while (time.time() - start_time) < duration_sec:
            # Schedule next job
            future = ray.get(self.scheduler.schedule_next_job.remote())
            
            # Check completed jobs
            completed = ray.get(self.scheduler.check_job_completion.remote())
            
            # Health check
            ray.get(self.worker_registry.check_worker_health.remote())
            
            await asyncio.sleep(1.0)
        
        print("\n✓ Scheduling loop complete\n")

    
    def print_final_report(self):
        """Print final KPI report."""
        ray.get(self.metrics.print_kpi_report.remote(self.worker_registry))
        
        # Scheduler stats
        stats = ray.get(self.scheduler.get_queue_stats.remote())
        print("📋 Scheduler Statistics:")
        print(f"  Queued:    {stats['queued']}")
        print(f"  Running:   {stats['running']}")
        print(f"  Completed: {stats['completed']}")
        print(f"  Failed:    {stats['failed']}")
        print()
    
    def shutdown(self):
        """Shutdown Ray cluster."""
        print("Shutting down Ray cluster...")
        ray.shutdown()


if __name__ == "__main__":
    master = MasterNode()
    
    master.register_worker_from_config()
    
    job1 = create_rl_training_job(episodes=50, query="attention mechanisms", start_paper_id="arxiv_1234")
    
    master.submit_jobs(job1)
    
    asyncio.run(master.run_scheduling_loop(duration_sec=30.0))
    
    # Final report
    master.print_final_report()
    
    master.shutdown()