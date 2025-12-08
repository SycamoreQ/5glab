import ray
import time
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class JobMetrics:
    """Metrics for a single job."""
    job_id: str
    job_type: str
    submitted_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    worker_id: Optional[str] = None
    
    def get_queue_time(self) -> Optional[float]:
        """Time spent in queue."""
        if self.started_at:
            return self.started_at - self.submitted_at
        return None
    
    def get_execution_time(self) -> Optional[float]:
        """Time spent executing."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    def get_total_time(self) -> Optional[float]:
        """Total time from submission to completion."""
        if self.completed_at:
            return self.completed_at - self.submitted_at
        return None


@ray.remote
class MetricsCollector:
    """
    Ray Actor for collecting and analyzing system metrics.
    Calculates KPIs: speedup, resilience, utilization.
    """
    
    def __init__(self):
        self.job_metrics: Dict[str, JobMetrics] = {}
        self.worker_utilization: Dict[str, List[float]] = defaultdict(list)
        self.system_start_time = time.time()
        
        # Baseline for speedup calculation
        self.baseline_execution_time: Optional[float] = None
        
        print("✓ Metrics Collector initialized")
    
    def record_job_submitted(self, job_id: str, job_type: str):
        """Record job submission."""
        self.job_metrics[job_id] = JobMetrics(
            job_id=job_id,
            job_type=job_type,
            submitted_at=time.time()
        )
    
    def record_job_started(self, job_id: str, worker_id: str):
        """Record job start."""
        if job_id in self.job_metrics:
            self.job_metrics[job_id].started_at = time.time()
            self.job_metrics[job_id].worker_id = worker_id
    
    def record_job_completed(self, job_id: str):
        """Record job completion."""
        if job_id in self.job_metrics:
            self.job_metrics[job_id].completed_at = time.time()
    
    def record_worker_utilization(self, worker_id: str, utilization: float):
        """Record worker CPU/GPU utilization."""
        self.worker_utilization[worker_id].append(utilization)
    
    def calculate_speedup(self) -> float:
        """
        Calculate speedup compared to baseline (single-node execution).
        Speedup = T_baseline / T_distributed
        """
        completed_jobs = [
            m for m in self.job_metrics.values() 
            if m.completed_at is not None
        ]
        
        if not completed_jobs:
            return 0.0
        
        # Average execution time in distributed setup
        avg_distributed_time = sum(
            m.get_execution_time() for m in completed_jobs
        ) / len(completed_jobs)
        
        # If no baseline, use first job as baseline
        if self.baseline_execution_time is None:
            self.baseline_execution_time = completed_jobs[0].get_execution_time()
        
        speedup = self.baseline_execution_time / avg_distributed_time
        return speedup
    
    def calculate_node_utilization(self) -> Dict[str, float]:
        """
        Calculate average utilization per worker node.
        Returns: {worker_id: avg_utilization_percent}
        """
        utilization = {}
        
        for worker_id, util_samples in self.worker_utilization.items():
            if util_samples:
                utilization[worker_id] = sum(util_samples) / len(util_samples)
            else:
                utilization[worker_id] = 0.0
        
        return utilization
    
    def calculate_resilience_score(self, worker_registry) -> float:
        """
        Calculate system resilience.
        Resilience = (healthy_workers / total_workers) * 
                     (1 - failure_rate)
        """
        worker_stats = ray.get(worker_registry.get_worker_stats.remote())
        
        if worker_stats['total_workers'] == 0:
            return 0.0
        
        health_ratio = worker_stats['healthy_workers'] / worker_stats['total_workers']
        
        total_jobs = worker_stats['total_jobs_completed'] + worker_stats['total_jobs_failed']
        failure_rate = (
            worker_stats['total_jobs_failed'] / total_jobs 
            if total_jobs > 0 else 0.0
        )
        
        resilience = health_ratio * (1 - failure_rate)
        return resilience
    
    def get_kpi_summary(self, worker_registry) -> Dict:
        """Get comprehensive KPI summary."""
        completed_jobs = [
            m for m in self.job_metrics.values() 
            if m.completed_at is not None
        ]
        
        avg_queue_time = (
            sum(m.get_queue_time() for m in completed_jobs) / len(completed_jobs)
            if completed_jobs else 0.0
        )
        
        avg_execution_time = (
            sum(m.get_execution_time() for m in completed_jobs) / len(completed_jobs)
            if completed_jobs else 0.0
        )
        
        throughput = len(completed_jobs) / (time.time() - self.system_start_time)
        
        return {
            'speedup': self.calculate_speedup(),
            'resilience': self.calculate_resilience_score(worker_registry),
            'node_utilization': self.calculate_node_utilization(),
            'total_jobs': len(self.job_metrics),
            'completed_jobs': len(completed_jobs),
            'avg_queue_time_sec': avg_queue_time,
            'avg_execution_time_sec': avg_execution_time,
            'throughput_jobs_per_sec': throughput,
            'system_uptime_sec': time.time() - self.system_start_time
        }
    
    def print_kpi_report(self, worker_registry):
        """Print formatted KPI report."""
        kpis = self.get_kpi_summary(worker_registry)
        
        print("\n" + "="*80)
        print("DISTRIBUTED SYSTEM KPI REPORT")
        print("="*80)
        
        print(f"\n Performance Metrics:")
        print(f"  Speedup:        {kpis['speedup']:.2f}x")
        print(f"  Resilience:     {kpis['resilience']:.2%}")
        print(f"  Throughput:     {kpis['throughput_jobs_per_sec']:.2f} jobs/sec")
        
        print(f"\n Timing Metrics:")
        print(f"  Avg Queue Time:     {kpis['avg_queue_time_sec']:.2f}s")
        print(f"  Avg Execution Time: {kpis['avg_execution_time_sec']:.2f}s")
        print(f"  System Uptime:      {kpis['system_uptime_sec']:.1f}s")
        
        print(f"\n Job Statistics:")
        print(f"  Total Jobs:      {kpis['total_jobs']}")
        print(f"  Completed Jobs:  {kpis['completed_jobs']}")
        
        print(f"\n Node Utilization:")
        for worker_id, util in kpis['node_utilization'].items():
            print(f"  {worker_id}: {util:.1f}%")
        
        print("="*80 + "\n")
