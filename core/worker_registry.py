import ray 
import time
import psutil
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
from enum import Enum


class WorkerStatus(Enum):
    """Worker health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass
class WorkerCapability:
    """Worker node capabilities."""
    worker_id: str
    node_type: str

    cpu_count: int
    cpu_freq_mhz: float
    has_gpu: bool
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]
    ram_total_gb: float
    
    status: WorkerStatus = WorkerStatus.HEALTHY
    last_heartbeat: float = 0.0

    cpu_usage_percent: float = 0.0
    ram_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    
    jobs_completed: int = 0
    jobs_failed: int = 0
    total_compute_time: float = 0.0



@ray.remote()
class WorkerRegistry: 
    def __init__(self, heartbeat_timeout: float = 30.0):
        self.workers: Dict[str, WorkerCapability] = {}
        self.heartbeat_timeout = heartbeat_timeout
        print("Worker Registry initialized")


    def registry_worker(self , worker_id:str , capabilities: Dict) -> bool: 
        worker = WorkerCapability(
            worker_id=worker_id,
            node_type=capabilities.get('node_type', 'unknown'),
            cpu_count=capabilities.get('cpu_count', 1),
            cpu_freq_mhz=capabilities.get('cpu_freq_mhz', 0.0),
            has_gpu=capabilities.get('has_gpu', False),
            gpu_name=capabilities.get('gpu_name'),
            gpu_memory_gb=capabilities.get('gpu_memory_gb'),
            ram_total_gb=capabilities.get('ram_total_gb', 0.0),
            last_heartbeat=time.time()
        )
        
        self.workers[worker_id] = worker
        print(f"Worker registered: {worker_id} ({worker.node_type})")
        print(f"CPU: {worker.cpu_count} cores, RAM: {worker.ram_total_gb:.1f}GB, GPU: {worker.gpu_name or 'None'}")
        return True 
    
    def update_heartbeat(self , worker_id: str , cpu_usage: float , ram_usage: float , gpu_usage: float) -> bool: 
        if worker_id not in self.workers: 
            print("unknown worker")
            return False 
        
        worker = self.workers[worker_id]
        worker.last_heartbeat = time.time()
        worker.cpu_usage_percent = cpu_usage
        worker.ram_usage_percent = ram_usage
        worker.gpu_usage_percent = gpu_usage
        worker.status = WorkerStatus.HEALTHY
        
        return True
    
    def mark_job_completion(self , worker_id: str , compute_time: float): 
        if worker_id in self.workers: 
            self.workers[worker_id].jobs_completed += 1 
            self.workers[worker_id].total_compute_time += compute_time 


    def check_worker_health(self) -> Dict[str, WorkerStatus]:
        current_time = time.time()
        health_status = {}
        
        for worker_id, worker in self.workers.items():
            time_since_heartbeat = current_time - worker.last_heartbeat
            
            if time_since_heartbeat > self.heartbeat_timeout:
                worker.status = WorkerStatus.OFFLINE
                print(f"⚠ Worker {worker_id} is OFFLINE (no heartbeat for {time_since_heartbeat:.1f}s)")
            elif worker.cpu_usage_percent > 90 or worker.ram_usage_percent > 90:
                worker.status = WorkerStatus.DEGRADED
            else:
                worker.status = WorkerStatus.HEALTHY
            
            health_status[worker_id] = worker.status
        
        return health_status
    
    def mark_job_failed(self, worker_id: str):
        if worker_id in self.workers:
            self.workers[worker_id].jobs_failed += 1
    
    def check_worker_health(self) -> Dict[str, WorkerStatus]:
        current_time = time.time()
        health_status = {}
        
        for worker_id, worker in self.workers.items():
            time_since_heartbeat = current_time - worker.last_heartbeat
            
            if time_since_heartbeat > self.heartbeat_timeout:
                worker.status = WorkerStatus.OFFLINE
                print(f"⚠ Worker {worker_id} is OFFLINE (no heartbeat for {time_since_heartbeat:.1f}s)")
            elif worker.cpu_usage_percent > 90 or worker.ram_usage_percent > 90:
                worker.status = WorkerStatus.DEGRADED
            else:
                worker.status = WorkerStatus.HEALTHY
            
            health_status[worker_id] = worker.status
        
        return health_status
    
    def get_available_workers()