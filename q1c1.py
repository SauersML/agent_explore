# ─── Imports ───────────────────────────────────────────────────────────────────

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.lax as lax
import jax.nn
import jax.numpy as jnp
import jax.random
import numpy as np
import pandas as pd


# ─── Simulation constants ──────────────────────────────────────────────────────

NUM_AGENTS                    = 64
NUM_COOPS                     = 4
MAX_LOSSES_BEFORE_ELIMINATION = 4     # k in the spec
NUM_TOURNAMENTS               = 1000
MAX_STEPS_PER_TOURNAMENT      = 50    # upper bound for lax.fori_loop;
                                      # true max with k=4 is well under 40
NUM_ABILITY_LEVELS            = 4     # equals NUM_COOPS; abilities ∈ {1…N_coop}
MAX_PAIRS_PER_COOP            = 32    # M // 2; hard ceiling for fixed-shape arrays

# Third-axis indices into zone_array  (NUM_AGENTS, NUM_COOPS, 3)
ZONE_BATTLE    = 0
ZONE_SPECTATOR = 1
ZONE_TRANSIT   = 2

# Action encoding in action_array  (NUM_AGENTS,):
#   0                                       no-op
#   ACTION_WATCH_BASE + coop_index  (1..4)  watch that coop
#   ACTION_MOVE_BASE  + coop_index  (5..8)  move to that coop
ACTION_NO_OP      = 0
ACTION_WATCH_BASE = 1
ACTION_MOVE_BASE  = ACTION_WATCH_BASE + NUM_COOPS

LOG_PROBABILITY_FLOOR       = -30.0  # clamp before jnp.log to prevent -inf
ABILITY_SCORE_POLICY_WEIGHT = 0.65
KING_BELIEF_POLICY_WEIGHT   = 0.35
PAIR_SENTINEL               = NUM_AGENTS  # padding in pairs arrays; never a valid index


# ─── Ability prior ─────────────────────────────────────────────────────────────

def compute_ability_prior_log_probs() -> jnp.ndarray:
    """
    Log-normalized truncated Poisson prior over ability levels 1..NUM_ABILITY_LEVELS.

    Unnormalized weight for level v:  exp(-v / (NUM_COOPS / 3.0)).
    Rate parameter 4/3 ≈ 1.333 makes lower abilities appreciably more probable.

    Returns shape (NUM_ABILITY_LEVELS,) float32.
    """
    ability_level_values     = jnp.arange(1, NUM_ABILITY_LEVELS + 1, dtype=jnp.float32)
    poisson_rate_parameter   = float(NUM_COOPS) / 3.0
    unnormalized_log_weights = -ability_level_values / poisson_rate_parameter
    return unnormalized_log_weights - jax.nn.logsumexp(unnormalized_log_weights)


ABILITY_PRIOR_LOG_PROBS: jnp.ndarray = compute_ability_prior_log_probs()


# ─── Entity dataclasses  (Python-only; never inside a JIT boundary) ────────────

@dataclass
class Location:
    """Physical position of any entity: which coop and which zone within it."""
    coop_index:          int
    zone_index:          int
    transit_target_coop: int = -1   # destination while in ZONE_TRANSIT; -1 otherwise


@dataclass
class Cage:
    """Fixed 1-vs-1 battle arena in a coop's BattleZone. Each coop has M // 2 = 32."""
    cage_id:                  int
    home_coop_index:          int
    current_occupant_indices: Optional[Tuple[int, int]] = None


@dataclass
class Monitor:
    """
    Fixed observation point in a coop's SpectatorZone.
    Each coop has NUM_COOPS monitors, one per viewable region including itself.
    """
    monitor_id:              int
    installed_in_coop_index: int
    viewing_coop_index:      int


@dataclass
class Agent:
    """
    Python-side view of one chicken; populated by build_agents() after simulation.
    The JAX engine never touches this object.

    ability_belief  (NUM_AGENTS, NUM_COOPS, NUM_ABILITY_LEVELS)  float32
        [target_agent, coop_index, ability_level] = this agent's posterior that
        target_agent has that 0-indexed ability in that coop.  Each [:, :, :] row sums to 1.

    king_belief  (NUM_AGENTS, NUM_COOPS, 2)  float32
        [target_agent, coop_index, 1] = this agent's belief that target_agent
        is king in coop_index.

    observation_memory  (NUM_AGENTS, NUM_AGENTS, NUM_TOURNAMENTS)  int8
        [agent_i, agent_j, tournament] = outcome of i-vs-j as witnessed by this
        agent in that tournament.  +1 win, -1 loss, 0 not observed.
    """
    agent_index:        int
    observation_memory: np.ndarray
    ability_belief:     np.ndarray
    my_ability:         np.ndarray   # (NUM_COOPS,) int32, 0-indexed, known exactly
    my_location:        Location
    last_observation:   np.ndarray   # (NUM_AGENTS, NUM_AGENTS, NUM_COOPS) int8
    king_belief:        np.ndarray

    def action_policy(self) -> Tuple[str, int]:
        """
        Scores each coop as a weighted sum of this agent's known ability there
        and its belief that it will be king there.  Same strategy for every agent.

        Returns one of:
            ('no_op', -1)            battling or in transit; no choice available
            ('watch', coop_index)    already in the best-scoring coop; observe it
            ('move',  coop_index)    a different coop scores higher; travel there
        """
        if self.my_location.zone_index != ZONE_SPECTATOR:
            return ('no_op', -1)

        self_king_probability_per_coop  = self.king_belief[self.agent_index, :, 1]
        normalized_ability_per_coop     = self.my_ability / NUM_ABILITY_LEVELS

        coop_combined_score = (
            ABILITY_SCORE_POLICY_WEIGHT * normalized_ability_per_coop
            + KING_BELIEF_POLICY_WEIGHT * self_king_probability_per_coop
        )

        best_coop_index = int(np.argmax(coop_combined_score))

        if best_coop_index != self.my_location.coop_index:
            return ('move', best_coop_index)
        return ('watch', self.my_location.coop_index)
