//! R36.8G.2C.4 — deterministic stage-0 key-cache policy simulation.
//!
//! All policies use identical request sequences and equal-size,
//! equal-generation-cost keys.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyPolicy {
    Lru,
    Belady,
    FixedIntersection,
    Sapkr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyPolicyStats {
    pub requests: usize,
    pub hits: usize,
    pub generations: usize,
    pub peak_keys: usize,
}

pub fn simulate_key_policy(
    schedule: &[isize],
    windows: usize,
    capacity: usize,
    policy: KeyPolicy,
) -> KeyPolicyStats {
    simulate_key_policy_with_boundary(schedule, windows, capacity, policy, schedule.len())
}

pub fn simulate_key_policy_with_boundary(
    schedule: &[isize],
    windows: usize,
    capacity: usize,
    policy: KeyPolicy,
    group_boundary: usize,
) -> KeyPolicyStats {
    assert!(group_boundary <= schedule.len());
    assert!(!schedule.is_empty());
    assert!(windows > 0);

    let requests: Vec<isize> = (0..windows)
        .flat_map(|_| schedule.iter().copied())
        .collect();

    let mut frequency = BTreeMap::<isize, usize>::new();
    for &offset in &requests {
        *frequency.entry(offset).or_default() += 1;
    }

    let mut retained = BTreeSet::<isize>::new();

    if matches!(policy, KeyPolicy::FixedIntersection | KeyPolicy::Sapkr) {
        let mut candidates: Vec<(isize, usize)> = frequency
            .iter()
            .map(|(&offset, &count)| (offset, count))
            .collect();

        if policy == KeyPolicy::Sapkr {
            candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        } else {
            let group0: BTreeSet<isize> = schedule[..group_boundary].iter().copied().collect();
            let group1: BTreeSet<isize> = schedule[group_boundary..].iter().copied().collect();
            let shared: BTreeSet<isize> = group0.intersection(&group1).copied().collect();

            if group_boundary < schedule.len() {
                candidates.retain(|(offset, _)| shared.contains(offset));
            }
            candidates.sort_by_key(|&(offset, _)| offset);
        }

        retained.extend(
            candidates
                .into_iter()
                .take(capacity)
                .map(|(offset, _)| offset),
        );
    }

    let mut resident = VecDeque::<isize>::new();
    let mut hits = 0usize;
    let mut generations = 0usize;
    let mut peak_keys = 0usize;

    for (index, &offset) in requests.iter().enumerate() {
        if let Some(position) = resident.iter().position(|&v| v == offset) {
            hits += 1;

            if policy == KeyPolicy::Lru {
                resident.remove(position);
                resident.push_back(offset);
            }
            continue;
        }

        generations += 1;

        if capacity == 0 {
            continue;
        }

        match policy {
            KeyPolicy::FixedIntersection | KeyPolicy::Sapkr => {
                if retained.contains(&offset) {
                    resident.push_back(offset);
                }
            }
            KeyPolicy::Lru => {
                if resident.len() == capacity {
                    resident.pop_front();
                }
                resident.push_back(offset);
            }
            KeyPolicy::Belady => {
                if resident.len() == capacity {
                    let next_use = |key: isize| {
                        requests[index + 1..]
                            .iter()
                            .position(|&next| next == key)
                            .unwrap_or(usize::MAX)
                    };

                    let incoming_next = next_use(offset);

                    let victim = resident
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, cached)| next_use(**cached))
                        .map(|(position, &key)| (position, next_use(key)))
                        .expect("nonempty cache");

                    // Bypass admission if the incoming key is
                    // no more valuable than the eviction candidate.
                    if incoming_next < victim.1 {
                        resident.remove(victim.0);
                        resident.push_back(offset);
                    }
                } else {
                    resident.push_back(offset);
                }
            }
        }

        peak_keys = peak_keys.max(resident.len());
    }

    KeyPolicyStats {
        requests: requests.len(),
        hits,
        generations,
        peak_keys,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_policies_preserve_request_accounting() {
        let schedule = [1, 2, 3, 1, 2];

        for policy in [
            KeyPolicy::Lru,
            KeyPolicy::Belady,
            KeyPolicy::FixedIntersection,
            KeyPolicy::Sapkr,
        ] {
            let stats = simulate_key_policy(&schedule, 3, 2, policy);

            assert_eq!(stats.requests, 15);
            assert_eq!(stats.hits + stats.generations, 15);
            assert!(stats.peak_keys <= 2);
        }
    }

    #[test]
    fn full_capacity_generates_each_key_once() {
        let schedule = [1, 2, 3, 1, 2];

        for policy in [
            KeyPolicy::Lru,
            KeyPolicy::Belady,
            KeyPolicy::FixedIntersection,
            KeyPolicy::Sapkr,
        ] {
            let stats = simulate_key_policy(&schedule, 3, 3, policy);
            assert_eq!(stats.generations, 3);
        }
    }
}
