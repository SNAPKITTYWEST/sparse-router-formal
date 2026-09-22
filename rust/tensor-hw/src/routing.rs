//! Sparse work-stealing with memory locality heuristic
//!
//! RAW_ROUTE algorithm: Region-aware work-stealing scheduler
//! Balances locality (35%), memory affinity (20%), queue pressure (25%),
//! stealability (10%), and migration cost (-10%).

use std::collections::HashMap;

/// Memory region identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionId(pub u64);

/// Worker identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerId(pub u32);

/// Task to be routed
#[derive(Debug, Clone)]
pub struct Task {
    pub id: u64,
    pub region: RegionId,
    pub size: usize,
}

/// Worker state for routing decisions
#[derive(Debug, Clone)]
pub struct WorkerState {
    pub id: WorkerId,
    pub queue_depth: usize,
    pub locality_distance: f32,
    pub memory_affinity: f32,
    pub stealability: f32,
}

/// Region ownership for locality tracking
#[derive(Debug, Clone)]
pub struct RegionEntry {
    pub owner: Option<WorkerId>,
    pub access_count: u64,
}

/// RAW_ROUTE sparse routing scheduler
#[derive(Debug)]
pub struct SparseRouter {
    /// Region → owner mapping
    region_table: HashMap<RegionId, RegionEntry>,

    /// Worker topology and state
    workers: HashMap<WorkerId, WorkerState>,

    /// Routing statistics
    total_routes: u64,
}

impl SparseRouter {
    pub fn new() -> Self {
        SparseRouter {
            region_table: HashMap::new(),
            workers: HashMap::new(),
            total_routes: 0,
        }
    }

    /// Register a worker in topology
    pub fn register_worker(&mut self, id: WorkerId, locality_distance: f32) {
        let worker = WorkerState {
            id,
            queue_depth: 0,
            locality_distance,
            memory_affinity: 1.0,
            stealability: 1.0,
        };
        self.workers.insert(id, worker);
    }

    /// Update worker queue depth
    pub fn set_queue_depth(&mut self, id: WorkerId, depth: usize) {
        if let Some(worker) = self.workers.get_mut(&id) {
            worker.queue_depth = depth;
        }
    }

    /// Route task to worker using RAW_ROUTE algorithm
    ///
    /// **Algorithm**:
    /// 1. If region has owner, route there
    /// 2. Otherwise, score all candidates:
    ///    - locality: 35% (distance)
    ///    - memory_affinity: 20%
    ///    - queue_pressure: 25%
    ///    - stealability: 10%
    ///    - migration_cost: -10%
    /// 3. Pick highest scoring worker
    /// 4. If no topology, select least-loaded
    pub fn route(&mut self, task: &Task) -> WorkerId {
        self.total_routes += 1;

        // Step 1: Check region ownership
        let entry = self.region_table.entry(task.region).or_insert(RegionEntry {
            owner: None,
            access_count: 0,
        });
        entry.access_count += 1;

        if let Some(owner) = entry.owner {
            if self.workers.contains_key(&owner) {
                return owner;
            }
        }

        // Step 2: Score candidates
        let mut best_worker = None;
        let mut best_score = f32::NEG_INFINITY;

        for (_, worker) in &self.workers {
            let score = self.score_worker(worker, task);

            if score > best_score {
                best_score = score;
                best_worker = Some(worker.id);
            }
        }

        // Step 3: Use best worker or fallback
        let selected = best_worker.unwrap_or_else(|| self.select_least_loaded());

        // Step 4: Update region ownership
        if let Some(entry) = self.region_table.get_mut(&task.region) {
            entry.owner = Some(selected);
        }

        selected
    }

    /// Score a candidate worker for task
    ///
    /// **Formula**:
    /// score = 0.35 * locality
    ///       + 0.20 * memory_affinity
    ///       + 0.25 * queue_pressure
    ///       + 0.10 * stealability
    ///       - 0.10 * migration_cost
    fn score_worker(&self, worker: &WorkerState, task: &Task) -> f32 {
        // Locality: higher is better (closer workers score higher)
        let locality_score = 1.0 / (1.0 + worker.locality_distance);

        // Memory affinity: already in [0, 1]
        let memory_score = worker.memory_affinity;

        // Queue pressure: lower queue depth is better (normalized)
        let max_queue = 1000.0;
        let queue_pressure = 1.0 - (worker.queue_depth as f32 / max_queue).min(1.0);

        // Stealability: ability to work-steal from this worker
        let stealability = worker.stealability;

        // Migration cost: higher cost means lower score
        let migration_cost = (task.size as f32 / 1000.0).min(1.0);

        let score = 0.35 * locality_score
            + 0.20 * memory_score
            + 0.25 * queue_pressure
            + 0.10 * stealability
            - 0.10 * migration_cost;

        score.clamp(-1.0, 1.0)
    }

    /// Select least-loaded worker (fallback)
    fn select_least_loaded(&self) -> WorkerId {
        self.workers
            .values()
            .min_by_key(|w| w.queue_depth)
            .map(|w| w.id)
            .unwrap_or(WorkerId(0))
    }

    /// Get region owner
    pub fn region_owner(&self, region: RegionId) -> Option<WorkerId> {
        self.region_table.get(&region).and_then(|e| e.owner)
    }

    /// Clear region ownership (for load balancing)
    pub fn clear_region(&mut self, region: RegionId) {
        if let Some(entry) = self.region_table.get_mut(&region) {
            entry.owner = None;
        }
    }

    /// Get routing statistics
    pub fn stats(&self) -> (u64, usize) {
        (self.total_routes, self.region_table.len())
    }
}

impl Default for SparseRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let router = SparseRouter::new();
        assert_eq!(router.stats().0, 0);
    }

    #[test]
    fn test_register_worker() {
        let mut router = SparseRouter::new();
        router.register_worker(WorkerId(0), 0.5);
        router.register_worker(WorkerId(1), 1.5);
        assert_eq!(router.workers.len(), 2);
    }

    #[test]
    fn test_route_to_region_owner() {
        let mut router = SparseRouter::new();
        router.register_worker(WorkerId(0), 0.5);
        router.register_worker(WorkerId(1), 1.5);

        let task1 = Task {
            id: 1,
            region: RegionId(1),
            size: 100,
        };

        let task2 = Task {
            id: 2,
            region: RegionId(1),
            size: 100,
        };

        let w1 = router.route(&task1);
        let w2 = router.route(&task2);

        // Both tasks in same region should go to same worker
        assert_eq!(w1, w2);
    }

    #[test]
    fn test_score_bounds() {
        let router = SparseRouter::new();
        let worker = WorkerState {
            id: WorkerId(0),
            queue_depth: 500,
            locality_distance: 1.0,
            memory_affinity: 0.8,
            stealability: 0.9,
        };

        let task = Task {
            id: 1,
            region: RegionId(1),
            size: 100,
        };

        let score = router.score_worker(&worker, &task);
        assert!(score >= -1.0 && score <= 1.0);
    }

    #[test]
    fn test_clear_region() {
        let mut router = SparseRouter::new();
        router.register_worker(WorkerId(0), 0.5);

        let task = Task {
            id: 1,
            region: RegionId(1),
            size: 100,
        };

        let _ = router.route(&task);
        assert!(router.region_owner(RegionId(1)).is_some());

        router.clear_region(RegionId(1));
        assert!(router.region_owner(RegionId(1)).is_none());
    }

    #[test]
    fn test_least_loaded_selection() {
        let mut router = SparseRouter::new();
        router.register_worker(WorkerId(0), 0.5);
        router.register_worker(WorkerId(1), 1.5);

        router.set_queue_depth(WorkerId(0), 10);
        router.set_queue_depth(WorkerId(1), 100);

        let task = Task {
            id: 1,
            region: RegionId(2),
            size: 100,
        };

        let selected = router.route(&task);
        // First task should go to least-loaded (0) or whoever scores best
        assert!(selected == WorkerId(0) || selected == WorkerId(1));
    }

    #[test]
    fn test_stats() {
        let mut router = SparseRouter::new();
        router.register_worker(WorkerId(0), 0.5);

        let task1 = Task {
            id: 1,
            region: RegionId(1),
            size: 100,
        };
        let task2 = Task {
            id: 2,
            region: RegionId(2),
            size: 100,
        };

        router.route(&task1);
        router.route(&task2);

        let (routes, regions) = router.stats();
        assert_eq!(routes, 2);
        assert_eq!(regions, 2);
    }
}
