import ray 
import time 
import heapq
from typing import Dict, List, Optional 
from dataclasses import dataclass, field
from core.job_spec import JobSpec, JobStatus, JobType


@dataclass(order=True)
class PrioritizedJob: 
    priority: int 
    timestamp: float = field(compare=False)
    job: JobSpec = field(compare=False)
    

@ray.remote
class FIFOScheduler: 
    
    def __init__(self, worker_registry): 
        self.worker_registry = worker_registry
        self.job_queue = []
        self.running_jobs: Dict[str, JobSpec] = {}
        self.completed_jobs: Dict[str, JobSpec] = {}
        self.job_futures: Dict[str, ray.ObjectRef] = {}
        print("Scheduler initialized")

    def submit_job(self, job: JobSpec) -> str: 
        prioritized_job = PrioritizedJob(
            priority=-job.job_priority.value,  # Negative for max-heap
            timestamp=time.time(),
            job=job
        )

        heapq.heappush(self.job_queue, prioritized_job)
        print(f"Job queued: {job.job_id} (priority={job.job_priority.value}, type={job.job_type.value})")

        return job.job_id
    
    def schedule_next_job(self) -> Optional[ray.ObjectRef]: 
        if not self.job_queue: 
            return None 
        
        prioritized_job = heapq.heappop(self.job_queue)
        job = prioritized_job.job 

        # Get available workers
        available_workers = ray.get(self.worker_registry.get_available_workers.remote(
            min_cpus=job.resources.num_cpus, 
            min_gpu_memory=job.resources.num_gpus * 2.0,
            custom_resource=list(job.resources.custom_resources.keys())[0] if job.resources.custom_resources else None
        ))
        
        if not available_workers:
            # No workers available, requeue job
            heapq.heappush(self.job_queue, prioritized_job)
            return None
        
        worker_id = available_workers[0]
        
        try: 
            future = self._dispatch_job(job, worker_id)
            job.mark_started(worker_id)
            self.running_jobs[job.job_id] = job 
            self.job_futures[job.job_id] = future

            print(f"Job dispatched: {job.job_id} → {worker_id}")
            return future

        except Exception as e: 
            print(f"Failed to dispatch job {job.job_id}: {e}")
            job.mark_failed(str(e))
            self.completed_jobs[job.job_id] = job
            return None
        
    
    def _dispatch_job(self, job: JobSpec, worker_id: str) -> ray.ObjectRef: 
        """Dispatch job to appropriate worker based on job type."""
        if job.job_type == JobType.RL_TRAINING: 
            from workers.rl_trainer import DistributedRLTrainer
            
            # Get database config from job parameters or use defaults
            db_config = job.parameters.get('db_config', {})
            
            trainer = DistributedRLTrainer.options(
                name=f"trainer_{worker_id}_{job.job_id}",
                **job.resources.to_dict()
            ).remote(
                worker_id=worker_id,
                db_host=db_config.get('host', '192.168.1.10'),
                db_port=db_config.get('port', 7687),
                db_user=db_config.get('user', 'neo4j'),
                db_password=db_config.get('password', 'diam0ndman@3'),
                db_name=db_config.get('database', 'researchdbv3')
            )
            
            return trainer.train_episodes.remote(
                episodes=job.parameters['episodes'],
                query=job.parameters['query'],
                start_paper_id=job.parameters['start_paper_id']
            )

    def check_job_completion(self) -> List[str]:
        """Check for completed jobs and update status."""
        completed_job_ids = []
        
        for job_id, future in list(self.job_futures.items()):
            ready, _ = ray.wait([future], timeout=0)
            
            if ready:
                job = self.running_jobs.pop(job_id)
                del self.job_futures[job_id]
                
                try:
                    result = ray.get(ready[0])
                    job.mark_completed(result)
                    
                    # Update worker registry
                    duration = job.get_duration()
                    ray.get(self.worker_registry.mark_job_completed.remote(job.worker_id, duration))
                    
                    print(f"✓ Job completed: {job_id} (duration={duration:.1f}s)")
                    
                except Exception as e:
                    job.mark_failed(str(e))
                    ray.get(self.worker_registry.mark_job_failed.remote(job.worker_id))
                    
                    print(f"✗ Job failed: {job_id} - {e}")
                
                self.completed_jobs[job_id] = job
                completed_job_ids.append(job_id)
        
        return completed_job_ids
    
    def get_queue_stats(self) -> Dict:
        return {
            'queued': len(self.job_queue),
            'running': len(self.running_jobs),
            'completed': len(self.completed_jobs),
            'failed': sum(1 for j in self.completed_jobs.values() if j.status == JobStatus.FAILED)
        }
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        if job_id in self.running_jobs:
            return JobStatus.RUNNING
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id].status
        else:
            for pj in self.job_queue:
                if pj.job.job_id == job_id:
                    return JobStatus.PENDING
        return None
    
    def get_completed_jobs(self) -> Dict[str, Dict]:
        completed = {}
        
        for job_id, job in self.completed_jobs.items():
            completed[job_id] = {
                'job': job,
                'result': job.result,
                'status': job.status.value,
                'duration': job.get_duration()
            }
        
        return completed
    

class RoundRobinScheduler:
    """Round Robin Scheduler implementation."""
    
    def __init__(self , worker_registry): 
        self.worker_registry = worker_registry
        self.job_queue = []
        self.running_jobs : Dict[str , JobSpec] ={}
        self.completed_jobs: Dict[str , JobSpec] = {}
        self.job_futures: Dict[str , ray.ObjectRef] = {}
        self.quantum_time = float
        self.time_hash = 

    
    def submit_job(self , job: JobSpec) -> str:
        prioritized_job = PrioritizedJob(
            priority = job.job_priority.value,
            timestamp= time.time(),
            job = job 
        )

        heapq.heappush(self.job_queue , prioritized_job)
        print(f"Job queued: {job.job_id} (priority={job.job_priority.value}, type={job.job_type.value})")

    def schedule_next_job(self) -> Optional[ray.ObjectRef]:
        if not self.job_queue:
            return None
        
        prioriizted_job = heapq.heappop(self.job_queue)
