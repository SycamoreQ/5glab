import random 
from typing import List
from core.job_spec import (
    JobSpec, create_rl_training_job,
    JobPriority
)

class WorkloadGenerator:
    """Generate diverse workloads for testing."""
    
    # Sample queries
    QUERIES = [
        "attention mechanisms in transformers",
        "reinforcement learning for robotics",
        "graph neural networks",
        "computer vision object detection",
        "natural language processing BERT",
        "deep learning optimization",
        "generative adversarial networks",
        "meta learning few shot",
        "neural architecture search",
        "transfer learning"
    ]
    
    START_PAPERS = [
        "arxiv_1706.03762",  # Attention is All You Need
        "arxiv_1810.04805",  # BERT
        "arxiv_1406.2661",   # GAN
        "arxiv_1512.03385",  # ResNet
        "arxiv_1611.01578",  # Pixel2Pixel
    ]


    @staticmethod
    def generate_mixed_workload(num_jobs: int = 50) -> List[JobSpec]: 
        jobs = []
        
        for i in range(num_jobs):
            if random.random() < 0.7:  # 70% training
                job = create_rl_training_job(
                    episodes=random.choice([50, 100, 200]),
                    query=random.choice(WorkloadGenerator.QUERIES),
                    start_paper_id=random.choice(WorkloadGenerator.START_PAPERS),
                    priority=random.choice(list(JobPriority))
                )
            else:  # 30% inference
                job = create_rag_inference_job(
                    query=random.choice(WorkloadGenerator.QUERIES),
                    start_paper_id=random.choice(WorkloadGenerator.START_PAPERS),
                    max_steps=random.randint(5, 15),
                    priority=random.choice(list(JobPriority))
                )
            
            jobs.append(job)
        
        return jobs
    
    @staticmethod
    def generate_training_heavy_workload(num_jobs: int = 50) -> List[JobSpec]:
        """Generate training-heavy workload (90% training)."""
        jobs = []
        
        for i in range(num_jobs):
            if random.random() < 0.9:
                job = create_rl_training_job(
                    episodes=100,
                    query=random.choice(WorkloadGenerator.QUERIES),
                    start_paper_id=random.choice(WorkloadGenerator.START_PAPERS)
                )
            else:
                job = create_rag_inference_job(
                    query=random.choice(WorkloadGenerator.QUERIES),
                    start_paper_id=random.choice(WorkloadGenerator.START_PAPERS),
                    max_steps=10
                )
            
            jobs.append(job)
        
        return jobs
    
    @staticmethod
    def generate_burst_workload(
        burst_size: int = 50, 
        num_bursts: int = 3,
    ) -> List[JobSpec]: 
        
        jobs = []
        for burst in range(num_bursts):
            for i in range(burst_size):
                job = create_rl_training_job(
                    episodes=100,
                    query=random.choice(WorkloadGenerator.QUERIES),
                    start_paper_id=random.choice(WorkloadGenerator.START_PAPERS),
                    priority=JobPriority.HIGH if burst == 0 else JobPriority.MEDIUM
                )
                jobs.append(job)
        
        return jobs
