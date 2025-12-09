import ray
import psutil
import time
import socket
from typing import Dict, Optional


@ray.remote
class BaseWorker:
    """
    Base class for all workers.
    Handles capability reporting and heartbeat monitoring.
    """
    
    def __init__(self, worker_id: str, node_type: str = "unknown"):
        self.worker_id = worker_id
        self.node_type = node_type
        self.start_time = time.time()
        self.last_heartbeat = time.time()
        
        # Detect capabilities
        self.capabilities = self._detect_capabilities()
        
        print(f"Worker {self.worker_id} initialized on {socket.gethostname()}")
    
    def _detect_capabilities(self) -> Dict:
        """Detect hardware capabilities."""
        capabilities = {
            'worker_id': self.worker_id,
            'node_type': self.node_type,
            'hostname': socket.gethostname(),
            'cpu_count': psutil.cpu_count(logical=True),
            'cpu_freq_mhz': psutil.cpu_freq().max if psutil.cpu_freq() else 0.0,
            'ram_total_gb': psutil.virtual_memory().total / (1024**3),
        }
        
        # Detect GPU
        try:
            import torch
            if torch.cuda.is_available():
                capabilities['has_gpu'] = True
                capabilities['gpu_name'] = torch.cuda.get_device_name(0)
                capabilities['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                capabilities['has_gpu'] = False
                capabilities['gpu_name'] = None
                capabilities['gpu_memory_gb'] = None
        except:
            capabilities['has_gpu'] = False
            capabilities['gpu_name'] = None
            capabilities['gpu_memory_gb'] = None
        
        return capabilities
    
    def get_capabilities(self) -> Dict:
        """Return worker capabilities."""
        return self.capabilities
    
    def get_utilization(self) -> Dict:
        """Get current resource utilization."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram_percent = psutil.virtual_memory().percent
        
        gpu_percent = 0.0
        if self.capabilities['has_gpu']:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_percent = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100
            except:
                pass
        
        return {
            'cpu_usage': cpu_percent,
            'ram_usage': ram_percent,
            'gpu_usage': gpu_percent
        }
    
    def heartbeat(self, worker_registry):
        """Send heartbeat to worker registry."""
        self.last_heartbeat = time.time()
        util = self.get_utilization()
        
        ray.get(worker_registry.update_heartbeat.remote(
            self.worker_id,
            util['cpu_usage'],
            util['ram_usage'],
            util['gpu_usage']
        ))
    
    def get_uptime(self) -> float:
        """Get worker uptime in seconds."""
        return time.time() - self.start_time
