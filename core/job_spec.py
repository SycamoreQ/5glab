from typing import Dict , List , Any , Optional
from dataclasses import dataclass , asdict, field
from enum import Enum
import json 
import time 



class JobType(Enum):
    RL_TRAINING = "rl_training"
    EMBEDDING = "embedding"
    LLM_INFERENCE = "llm_inference"


class JobPriority(Enum): 
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10 

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResourceRequirement:
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    memory_mb: int = 1000
    custom_resources: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        resources = {
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "memory": self.memory_mb * 1024 * 1024
        }
        if self.custom_resources:
            resources.update(self.custom_resources)
        return resources
    

@dataclass 
class JobSpec: 
    job_id: str 
    job_type: JobType 
    job_priority: JobPriority 
    job_status: JobStatus

    resources: ResourceRequirement
    parameters: Dict


    deadline_sec: Optional[float] = None
    max_retries: int = 3
    timeout_sec: Optional[float] = 3600.0
    dependencies: List[str] = field(default_factory=list)
    
    status: JobStatus = JobStatus.PENDING
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    worker_id: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None


    def to_json(self) -> str:
        data = asdict(self)
        data['job_type'] = self.job_type.value
        data['priority'] = self.job_priority.value
        data['status'] = self.status.value
        return json.dumps(data, indent=2)


    def from_json(self , json_str) -> 'JobSpec': 
        data = json.loads(json_str)
        data['job_type'] = JobType(data['job_type'])
        data['priority'] = JobPriority(data['priority'])
        data['status'] = JobStatus(data['status'])
        data['resources'] = ResourceRequirement(**data['resources'])
        return JobSpec(**data)
    

    def mark_started(self, worker_id: str):
        """Mark job as started."""
        self.status = JobStatus.RUNNING
        self.started_at = time.time()
        self.worker_id = worker_id
    
    def mark_completed(self, result: Dict):
        """Mark job as completed."""
        self.status = JobStatus.COMPLETED
        self.completed_at = time.time()
        self.result = result
    
    def mark_failed(self, error: str):
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.completed_at = time.time()
        self.error = error
    
    def get_duration(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    

def create_rl_training_job(
    episodes: int,
    query: str,
    start_paper_id: str,
    priority: JobPriority = JobPriority.HIGH,
    status: JobStatus = JobStatus.PENDING
) -> JobSpec:
    """Create RL training job specification."""
    return JobSpec(
        job_id = f"rl_train_{int(time.time())}",
        job_type= JobType.RL_TRAINING,
        job_status= status,
        job_priority= priority,
        
        resources =ResourceRequirement(
            num_cpus=2.0,
            num_gpus=1.0,
            memory_mb=4096
        ),
        parameters={
            'episodes': episodes,
            'query': query,
            'start_paper_id': start_paper_id
        },
        timeout_sec=7200.0 
    )

def create_embedding_job(
    texts: List[str],
    priority: JobPriority = JobPriority.MEDIUM
) -> JobSpec:
    """Create batch embedding job specification."""
    return JobSpec(
        job_id=f"embed_{int(time.time())}",
        job_type=JobType.EMBEDDING,
        job_status=JobStatus.PENDING,
        job_priority=priority,
        resources=ResourceRequirement(
            num_cpus=4.0,
            num_gpus=0.0,
            memory_mb=2048,
            custom_resources={"pi": 1} 
        ),
        parameters={
            'texts': texts
        },
        timeout_sec=300.0 
    )