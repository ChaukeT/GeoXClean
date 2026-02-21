"""
Performance Monitoring System

Provides performance monitoring and metrics collection.
Tracks system performance, operation times, and resource usage.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import time
import sys
import psutil
import threading

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    OPERATION_TIME = "operation_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    QC_TIME = "qc_time"
    FIX_TIME = "fix_time"
    IMPORT_TIME = "import_time"
    EXPORT_TIME = "export_time"


@dataclass
class PerformanceMetric:
    """A single performance metric."""
    metric_id: str
    metric_type: MetricType
    timestamp: datetime
    value: float
    unit: str = ""
    operation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Performance monitoring system.
    
    Tracks system performance metrics and operation times.
    """
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self._metric_counter = 0
        self._max_metrics = 50000  # Keep last 50k metrics
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        logger.info("PerformanceMonitor initialized")
    
    def _generate_metric_id(self) -> str:
        """Generate a unique metric ID."""
        self._metric_counter += 1
        return f"METRIC-{self._metric_counter:08d}"
    
    def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        unit: str = "",
        operation: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PerformanceMetric:
        """
        Record a performance metric.
        
        Args:
            metric_type: Type of metric
            value: Metric value
            unit: Unit of measurement
            operation: Operation name
            metadata: Optional metadata
        
        Returns:
            Created PerformanceMetric
        """
        metric = PerformanceMetric(
            metric_id=self._generate_metric_id(),
            metric_type=metric_type,
            timestamp=datetime.now(),
            value=value,
            unit=unit,
            operation=operation,
            metadata=metadata or {},
        )
        
        self.metrics.append(metric)
        
        # Trim old metrics if we exceed max
        if len(self.metrics) > self._max_metrics:
            self.metrics = self.metrics[-self._max_metrics:]
        
        logger.debug(f"Recorded metric: {metric_type.value}={value}{unit} for {operation}")
        return metric
    
    def time_operation(
        self,
        operation: str,
        func: callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Time an operation and record the metric.
        
        Args:
            operation: Operation name
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        """
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Determine metric type based on operation
            metric_type = MetricType.OPERATION_TIME
            if "qc" in operation.lower():
                metric_type = MetricType.QC_TIME
            elif "fix" in operation.lower():
                metric_type = MetricType.FIX_TIME
            elif "import" in operation.lower():
                metric_type = MetricType.IMPORT_TIME
            elif "export" in operation.lower():
                metric_type = MetricType.EXPORT_TIME
            
            self.record_metric(
                metric_type=metric_type,
                value=elapsed,
                unit="seconds",
                operation=operation,
            )
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.record_metric(
                metric_type=MetricType.OPERATION_TIME,
                value=elapsed,
                unit="seconds",
                operation=f"{operation} (failed)",
                metadata={"error": str(e)},
            )
            raise
    
    def get_current_system_metrics(self) -> Dict[str, float]:
        """
        Get current system metrics.
        
        Returns:
            Dictionary of current system metrics
        """
        if not PSUTIL_AVAILABLE:
            return {}
        
        try:
            process = psutil.Process()
            
            metrics = {
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_mb": process.memory_info().rss / (1024 * 1024),
                "memory_percent": process.memory_percent(),
                "num_threads": process.num_threads(),
            }
            
            # System-wide metrics
            try:
                if PSUTIL_AVAILABLE:
                    metrics["system_cpu_percent"] = psutil.cpu_percent(interval=0.1)
                    metrics["system_memory_percent"] = psutil.virtual_memory().percent
                    metrics["system_memory_available_mb"] = psutil.virtual_memory().available / (1024 * 1024)
            except Exception:
                pass  # System metrics may not be available
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error getting system metrics: {e}")
            return {}
    
    def start_monitoring(self, interval_seconds: float = 5.0):
        """
        Start background monitoring of system metrics.
        
        Args:
            interval_seconds: Monitoring interval in seconds
        """
        if self._monitoring:
            logger.warning("Monitoring already started")
            return
        
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                try:
                    if PSUTIL_AVAILABLE:
                        metrics = self.get_current_system_metrics()
                        
                        if "cpu_percent" in metrics:
                            self.record_metric(
                                metric_type=MetricType.CPU_USAGE,
                                value=metrics["cpu_percent"],
                                unit="percent",
                                operation="system_monitoring",
                            )
                        
                        if "memory_mb" in metrics:
                            self.record_metric(
                                metric_type=MetricType.MEMORY_USAGE,
                                value=metrics["memory_mb"],
                                unit="MB",
                                operation="system_monitoring",
                            )
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(interval_seconds)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Started performance monitoring (interval={interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Stopped performance monitoring")
    
    def get_statistics(
        self,
        metric_type: Optional[MetricType] = None,
        operation: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Args:
            metric_type: Filter by metric type
            operation: Filter by operation
            start_date: Filter by start date
            end_date: Filter by end date
        
        Returns:
            Dictionary with statistics
        """
        filtered = self.metrics
        
        if metric_type:
            filtered = [m for m in filtered if m.metric_type == metric_type]
        
        if operation:
            filtered = [m for m in filtered if m.operation == operation]
        
        if start_date:
            filtered = [m for m in filtered if m.timestamp >= start_date]
        
        if end_date:
            filtered = [m for m in filtered if m.timestamp <= end_date]
        
        if not filtered:
            return {
                "count": 0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
            }
        
        values = [m.value for m in filtered]
        
        return {
            "count": len(filtered),
            "average": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "median": sorted(values)[len(values) // 2],
        }


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

