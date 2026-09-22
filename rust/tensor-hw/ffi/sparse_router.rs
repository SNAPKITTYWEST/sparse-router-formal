/*
 * Sparse Tensor Router with RAW_ROUTE Algorithm
 *
 * Formal work-stealing scheduler for tensor tasks.
 * Routes tasks to workers based on memory locality, queue pressure,
 * stealability, and migration cost.
 *
 * The RAW_ROUTE algorithm:
 * 1. Check if memory region has an owner → push task to owner
 * 2. Otherwise, collect candidate workers from sparse topology
 * 3. Score each candidate on locality, affinity, queue pressure, stealability, cost
 * 4. Route to best-scored candidate
 * 5. Update region ownership table
 * 6. Fallback: least-loaded worker if no good candidate
 *
 * Invariant:
 *   Every task is routed.
 *   Route decision preserves memory locality.
 *   Migration cost is considered before idle balancing.
 */

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/* ============================================================================
 * Types
 * ============================================================================
 */

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorkerId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemoryRegion(pub u64);

#[derive(Clone, Copy, Debug)]
pub struct TensorTask {
    pub task_id: u64,
    pub memory_region: MemoryRegion,
    pub buffer_id: u64,
    pub work_units: u64,     /* Estimated work size */
    pub data_size: u64,      /* Data size in bytes */
    pub priority: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct WorkerMetrics {
    pub queue_len: u64,
    pub capacity: u64,
    pub numa_node: u32,
    pub last_active: u64,    /* Timestamp of last execution */
}

/* ============================================================================
 * Sparse Topology
 *
 * Represents which workers are "close" to which regions.
 * Sparse means we only store the non-empty edges, not a full matrix.
 * ============================================================================ */

#[derive(Clone)]
pub struct SparseTopology {
    /* region → vec of nearby workers (sorted by distance) */
    pub region_edges: HashMap<MemoryRegion, Vec<WorkerId>>,

    /* worker → vec of nearby regions */
    pub worker_edges: HashMap<WorkerId, Vec<MemoryRegion>>,

    /* worker → NUMA node assignment */
    pub worker_numa: HashMap<WorkerId, u32>,
}

impl SparseTopology {
    pub fn new() -> Self {
        SparseTopology {
            region_edges: HashMap::new(),
            worker_edges: HashMap::new(),
            worker_numa: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, region: MemoryRegion, worker: WorkerId, numa: u32) {
        self.region_edges.entry(region).or_insert_with(Vec::new).push(worker);
        self.worker_edges.entry(worker).or_insert_with(Vec::new).push(region);
        self.worker_numa.insert(worker, numa);
    }

    pub fn candidates_for_region(&self, region: MemoryRegion) -> Vec<WorkerId> {
        self.region_edges
            .get(&region)
            .map(|v| v.clone())
            .unwrap_or_default()
    }
}

/* ============================================================================
 * Region Ownership Table
 *
 * Tracks which worker currently owns a memory region.
 * Updated after successful task routing.
 * ============================================================================ */

pub struct RegionOwnershipTable {
    pub owners: Arc<Mutex<HashMap<MemoryRegion, WorkerId>>>,
}

impl RegionOwnershipTable {
    pub fn new() -> Self {
        RegionOwnershipTable {
            owners: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn owner(&self, region: MemoryRegion) -> Option<WorkerId> {
        self.owners.lock().unwrap().get(&region).copied()
    }

    pub fn set_owner(&self, region: MemoryRegion, worker: WorkerId) {
        self.owners.lock().unwrap().insert(region, worker);
    }
}

/* ============================================================================
 * Scoring Functions
 *
 * Each scoring function returns a weight in [0.0, 1.0].
 * ============================================================================ */

/* Locality score: 1.0 if worker is on same NUMA node, 0.5 if neighbor, 0.0 if far */
fn score_locality(task: &TensorTask, worker: WorkerId, topology: &SparseTopology) -> f64 {
    let candidates = topology.candidates_for_region(task.memory_region);
    let distance = candidates.iter().position(|w| *w == worker).unwrap_or(100);

    match distance {
        0 => 1.0,
        1 => 0.8,
        2 => 0.6,
        _ => 0.3,
    }
}

/* Memory affinity score: how many buffers the worker has recently touched */
fn score_memory_affinity(task: &TensorTask, worker: WorkerId) -> f64 {
    /* Simplified: always 0.5 for now */
    /* In full implementation: track buffer access history per worker */
    0.5
}

/* Queue pressure score: inverse of queue length relative to capacity */
fn score_queue_pressure(worker_metrics: &WorkerMetrics) -> f64 {
    if worker_metrics.capacity == 0 {
        return 0.0;
    }
    let utilization = worker_metrics.queue_len as f64 / worker_metrics.capacity as f64;
    1.0 / (1.0 + utilization)
}

/* Stealability score: 1.0 if queue is full (desirable to steal), 0.0 if mostly empty */
fn score_stealability(worker_metrics: &WorkerMetrics) -> f64 {
    if worker_metrics.capacity == 0 {
        return 0.0;
    }
    let utilization = worker_metrics.queue_len as f64 / worker_metrics.capacity as f64;
    utilization.min(1.0)
}

/* Migration cost: proportional to data_size and inverse of locality */
fn score_migration_cost(task: &TensorTask, locality: f64) -> f64 {
    /* Cost is high if data is large and locality is poor */
    let relative_size = (task.data_size as f64) / (1024.0 * 1024.0); /* 1MB = 1.0 cost */
    let migration_penalty = (1.0 - locality).max(0.0);
    (relative_size * migration_penalty).min(10.0) /* Cap at 10 */
}

/* ============================================================================
 * RAW_ROUTE Algorithm
 *
 * Central routing decision for a task.
 * ============================================================================ */

pub struct SparseRouter {
    topology: SparseTopology,
    region_owners: RegionOwnershipTable,
    worker_queues: Arc<Mutex<HashMap<WorkerId, Vec<TensorTask>>>>,
    worker_metrics: Arc<Mutex<HashMap<WorkerId, WorkerMetrics>>>,
    route_log: Arc<Mutex<Vec<String>>>,
}

impl SparseRouter {
    pub fn new(topology: SparseTopology) -> Self {
        SparseRouter {
            topology,
            region_owners: RegionOwnershipTable::new(),
            worker_queues: Arc::new(Mutex::new(HashMap::new())),
            worker_metrics: Arc::new(Mutex::new(HashMap::new())),
            route_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /* Register a worker and its metrics */
    pub fn register_worker(&self, id: WorkerId, capacity: u64, numa: u32) {
        self.worker_metrics.lock().unwrap().insert(
            id,
            WorkerMetrics {
                queue_len: 0,
                capacity,
                numa_node: numa,
                last_active: 0,
            },
        );
        self.worker_queues
            .lock()
            .unwrap()
            .insert(id, Vec::new());
    }

    /* Core RAW_ROUTE decision */
    pub fn route(&self, task: TensorTask) -> WorkerId {
        let region = task.memory_region;

        /* Step 1: Check if region has existing owner */
        if let Some(owner) = self.region_owners.owner(region) {
            let mut queues = self.worker_queues.lock().unwrap();
            if let Some(queue) = queues.get_mut(&owner) {
                queue.push(task);
            }
            self.log(format!("ROUTE[{}] → owner {} (reuse)", task.task_id, owner.0));
            return owner;
        }

        /* Step 2: Collect candidates from sparse topology */
        let candidates = self.topology.candidates_for_region(region);

        if candidates.is_empty() {
            /* Fallback: least loaded worker */
            let least_loaded = self.find_least_loaded_worker();
            self.region_owners.set_owner(region, least_loaded);
            let mut queues = self.worker_queues.lock().unwrap();
            if let Some(queue) = queues.get_mut(&least_loaded) {
                queue.push(task);
            }
            self.log(format!(
                "ROUTE[{}] → fallback {} (no candidates)",
                task.task_id, least_loaded.0
            ));
            return least_loaded;
        }

        /* Step 3: Score each candidate */
        let metrics = self.worker_metrics.lock().unwrap();
        let mut best = None;
        let mut best_score = f64::NEG_INFINITY;

        for &candidate in &candidates {
            let m = match metrics.get(&candidate) {
                Some(m) => m,
                None => continue,
            };

            let locality = score_locality(&task, candidate, &self.topology);
            let affinity = score_memory_affinity(&task, candidate);
            let queue_pressure = score_queue_pressure(m);
            let stealability = score_stealability(m);
            let migration = score_migration_cost(&task, locality);

            /* Weighted score: locality is most important, migration is penalty */
            let score = (locality * 0.35)
                + (affinity * 0.20)
                + (queue_pressure * 0.25)
                + (stealability * 0.10)
                - (migration * 0.10);

            if score > best_score {
                best_score = score;
                best = Some(candidate);
            }
        }

        /* Step 4: Route to best candidate */
        if let Some(best_worker) = best {
            self.region_owners.set_owner(region, best_worker);
            let mut queues = self.worker_queues.lock().unwrap();
            if let Some(queue) = queues.get_mut(&best_worker) {
                queue.push(task);
            }
            self.log(format!(
                "ROUTE[{}] → worker {} (score={})",
                task.task_id, best_worker.0, best_score
            ));
            return best_worker;
        }

        /* Fallback */
        let least_loaded = self.find_least_loaded_worker();
        self.region_owners.set_owner(region, least_loaded);
        let mut queues = self.worker_queues.lock().unwrap();
        if let Some(queue) = queues.get_mut(&least_loaded) {
            queue.push(task);
        }
        self.log(format!(
            "ROUTE[{}] → fallback {} (no good score)",
            task.task_id, least_loaded.0
        ));
        least_loaded
    }

    fn find_least_loaded_worker(&self) -> WorkerId {
        let metrics = self.worker_metrics.lock().unwrap();
        metrics
            .iter()
            .min_by_key(|(_, m)| m.queue_len)
            .map(|(id, _)| *id)
            .unwrap_or(WorkerId(0))
    }

    fn log(&self, msg: String) {
        self.route_log.lock().unwrap().push(msg);
    }

    pub fn dump_log(&self) -> Vec<String> {
        self.route_log.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_route_reuses_region_owner() {
        let mut topo = SparseTopology::new();
        topo.add_edge(MemoryRegion(1), WorkerId(0), 0);
        topo.add_edge(MemoryRegion(1), WorkerId(1), 0);

        let router = SparseRouter::new(topo);
        router.register_worker(WorkerId(0), 10, 0);
        router.register_worker(WorkerId(1), 10, 0);

        let task1 = TensorTask {
            task_id: 1,
            memory_region: MemoryRegion(1),
            buffer_id: 100,
            work_units: 1000,
            data_size: 1024,
            priority: 0,
        };

        let routed1 = router.route(task1);

        /* Second task same region should go to same worker */
        let task2 = TensorTask {
            task_id: 2,
            memory_region: MemoryRegion(1),
            buffer_id: 101,
            work_units: 1000,
            data_size: 1024,
            priority: 0,
        };

        let routed2 = router.route(task2);
        assert_eq!(routed1, routed2, "Should reuse region owner");
    }

    #[test]
    fn test_raw_route_respects_locality() {
        let mut topo = SparseTopology::new();
        topo.add_edge(MemoryRegion(1), WorkerId(0), 0); /* Closest to region 1 */
        topo.add_edge(MemoryRegion(1), WorkerId(1), 0); /* Also close */
        topo.add_edge(MemoryRegion(1), WorkerId(2), 1); /* Far (NUMA 1) */

        let router = SparseRouter::new(topo);
        for i in 0..3 {
            router.register_worker(WorkerId(i), 10, i as u32);
        }

        let task = TensorTask {
            task_id: 1,
            memory_region: MemoryRegion(1),
            buffer_id: 100,
            work_units: 1000,
            data_size: 1024,
            priority: 0,
        };

        let routed = router.route(task);
        /* Should prefer worker 0 or 1 over 2 */
        assert!(routed == WorkerId(0) || routed == WorkerId(1));
    }
}
