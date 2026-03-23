from typing import NamedTuple


# ─── Simulation state ──────────────────────────────────────────────────────────

class SimulationState(NamedTuple):
    zone_array:                 jnp.ndarray  # (NUM_AGENTS, NUM_COOPS, 3)                             int8
    transit_destination_coop:   jnp.ndarray  # (NUM_AGENTS,)                                          int32
    action_array:               jnp.ndarray  # (NUM_AGENTS,)                                          int32
    agent_abilities:            jnp.ndarray  # (NUM_AGENTS, NUM_COOPS)              0-indexed          int32
    tournament_losses_per_coop: jnp.ndarray  # (NUM_AGENTS, NUM_COOPS)                                int32
    tournament_fought_matrix:   jnp.ndarray  # (NUM_AGENTS, NUM_AGENTS, NUM_COOPS)                    int8
    tournament_outcomes:        jnp.ndarray  # (NUM_AGENTS, NUM_AGENTS, NUM_COOPS)  antisymmetric      int8
    tournament_obs_coverage:    jnp.ndarray  # (NUM_AGENTS, NUM_COOPS)                                int8
    battle_history:             jnp.ndarray  # (NUM_AGENTS, NUM_AGENTS, NUM_COOPS, NUM_TOURNAMENTS)   int8
    obs_mask:                   jnp.ndarray  # (NUM_AGENTS, NUM_COOPS, NUM_TOURNAMENTS)               int8
    ability_beliefs:            jnp.ndarray  # (NUM_AGENTS, NUM_AGENTS, NUM_COOPS, NUM_ABILITY_LEVELS) float32
    king_beliefs:               jnp.ndarray  # (NUM_AGENTS, NUM_AGENTS, NUM_COOPS, 2)                 float32
    crown_counts:               jnp.ndarray  # (NUM_AGENTS, NUM_COOPS)                                int32
    tournament_index:           jnp.ndarray  # ()                                                     int32
    step_done_flag:             jnp.ndarray  # ()                                                     bool
    rng_key:                    jnp.ndarray  # (2,)                                                   uint32


# ─── Event: AssignAgentsToCage (RelocateZone is its direct outcome) ────────────

def _build_pairs_for_coop(
    zone_array,
    tournament_losses_per_coop,
    tournament_fought_matrix,
    coop_index,          # Python int; statically known at trace time — unrolled by caller
    rng_subkey,
):
    """
    Greedily pair eligible agents in one coop using lax.scan over sorted positions.

    Eligible = in ZONE_BATTLE of this coop AND losses < MAX_LOSSES_BEFORE_ELIMINATION.
    Sorted ascending by loss count; ties broken by small uniform noise.

    Returns
    -------
    final_pairs_array       (MAX_PAIRS_PER_COOP, 2)  int32  — padded with PAIR_SENTINEL
    final_pair_count        ()                        int32
    unpaired_eligible_mask  (NUM_AGENTS,)             bool
    """
    in_battle_zone_this_coop   = zone_array[:, coop_index, ZONE_BATTLE].astype(jnp.bool_)
    under_loss_limit_this_coop = tournament_losses_per_coop[:, coop_index] < MAX_LOSSES_BEFORE_ELIMINATION
    eligible_to_battle         = in_battle_zone_this_coop & under_loss_limit_this_coop  # (M,)

    tie_breaking_noise = jax.random.uniform(rng_subkey, (NUM_AGENTS,)) * 0.1
    sort_keys = jnp.where(
        eligible_to_battle,
        tournament_losses_per_coop[:, coop_index].astype(jnp.float32) + tie_breaking_noise,
        jnp.float32(NUM_AGENTS + 1)   # ineligible agents sink to the end
    )
    sorted_agent_indices = jnp.argsort(sort_keys)  # (M,) — eligible first, by loss count

    # Pre-slice the fought matrix for this coop; closed over in scan body
    fought_in_this_coop = tournament_fought_matrix[:, :, coop_index]  # (M, M)

    def scan_body(carry, scan_position):
        taken_mask, pairs_array, pair_count = carry

        current_agent = sorted_agent_indices[scan_position]  # dynamic gather → int32 scalar

        is_eligible_and_free = eligible_to_battle[current_agent] & ~taken_mask[current_agent]

        already_fought_current_agent = fought_in_this_coop[:, current_agent]  # (M,) dynamic gather
        valid_partner_mask = (
            eligible_to_battle
            & ~taken_mask
            & (already_fought_current_agent == 0)
            & (jnp.arange(NUM_AGENTS, dtype=jnp.int32) != current_agent)
        )

        do_pair = is_eligible_and_free & valid_partner_mask.any()

        # Lowest-index valid partner; falls back to 0 when none exist (gated by do_pair)
        partner_search_cost = jnp.where(
            valid_partner_mask,
            jnp.arange(NUM_AGENTS, dtype=jnp.int32),
            jnp.int32(NUM_AGENTS)
        )
        first_valid_partner = jnp.argmin(partner_search_cost).astype(jnp.int32)

        safe_write_position = jnp.minimum(pair_count, jnp.int32(MAX_PAIRS_PER_COOP - 1))
        current_pair_row    = jnp.stack([current_agent.astype(jnp.int32), first_valid_partner])
        pairs_array = pairs_array.at[safe_write_position].set(
            jnp.where(do_pair, current_pair_row, pairs_array[safe_write_position])
        )

        taken_mask = taken_mask.at[current_agent].set(taken_mask[current_agent] | do_pair)
        taken_mask = taken_mask.at[first_valid_partner].set(
            taken_mask[first_valid_partner] | do_pair
        )

        return (taken_mask, pairs_array, pair_count + do_pair.astype(jnp.int32)), jnp.int32(0)

    (final_taken_mask, final_pairs_array, final_pair_count), _ = lax.scan(
        scan_body,
        (
            jnp.zeros(NUM_AGENTS, dtype=jnp.bool_),
            jnp.full((MAX_PAIRS_PER_COOP, 2), PAIR_SENTINEL, dtype=jnp.int32),
            jnp.int32(0),
        ),
        jnp.arange(NUM_AGENTS, dtype=jnp.int32),
    )

    unpaired_eligible_mask = eligible_to_battle & ~final_taken_mask
    return final_pairs_array, final_pair_count, unpaired_eligible_mask


def assign_agents_to_cage(state, rng_subkeys_per_coop):
    """
    For each coop: pair eligible BATTLE agents, then move unpaired eligible → SPECTATOR.

    Returns (updated_state, all_coop_pairs, all_pair_counts) where
        all_coop_pairs   (NUM_COOPS, MAX_PAIRS_PER_COOP, 2)  int32
        all_pair_counts  (NUM_COOPS,)                         int32
    """
    zone_array       = state.zone_array
    pairs_list       = []
    pair_counts_list = []

    for coop_index in range(NUM_COOPS):
        pairs_array, pair_count, unpaired_eligible_mask = _build_pairs_for_coop(
            zone_array,
            state.tournament_losses_per_coop,
            state.tournament_fought_matrix,
            coop_index,
            rng_subkeys_per_coop[coop_index],
        )
        pairs_list.append(pairs_array)
        pair_counts_list.append(pair_count)

        unpaired_mask_int8 = unpaired_eligible_mask.astype(jnp.int8)
        zone_array = (
            zone_array
            .at[:, coop_index, ZONE_BATTLE].add(-unpaired_mask_int8)
            .at[:, coop_index, ZONE_SPECTATOR].add(unpaired_mask_int8)
        )

    all_coop_pairs  = jnp.stack(pairs_list,       axis=0)   # (N, MAX_PAIRS_PER_COOP, 2)
    all_pair_counts = jnp.stack(pair_counts_list, axis=0)   # (N,)
    return state._replace(zone_array=zone_array), all_coop_pairs, all_pair_counts


# ─── Event: DominanceBattle ────────────────────────────────────────────────────

def dominance_battle(state, all_coop_pairs, all_pair_counts, battle_rng_subkeys):
    """
    Resolve all battles across all coops simultaneously.

    Higher ability wins outright; equal abilities resolved by coin flip.
    Sentinel pairs (clipped to index NUM_AGENTS-1) contribute zero to every
    scatter because valid_pair_mask_int8 is 0 for them.
    """
    tournament_outcomes        = state.tournament_outcomes
    tournament_losses_per_coop = state.tournament_losses_per_coop
    tournament_fought_matrix   = state.tournament_fought_matrix

    for coop_index in range(NUM_COOPS):
        pairs_this_coop      = all_coop_pairs[coop_index]   # (MAX_PAIRS_PER_COOP, 2)
        pair_count_this_coop = all_pair_counts[coop_index]

        valid_pair_mask_bool = jnp.arange(MAX_PAIRS_PER_COOP) < pair_count_this_coop
        valid_pair_mask_int8 = valid_pair_mask_bool.astype(jnp.int8)

        safe_left_agents  = jnp.clip(pairs_this_coop[:, 0], 0, NUM_AGENTS - 1)
        safe_right_agents = jnp.clip(pairs_this_coop[:, 1], 0, NUM_AGENTS - 1)

        left_ability_values  = state.agent_abilities[safe_left_agents,  coop_index]
        right_ability_values = state.agent_abilities[safe_right_agents, coop_index]

        left_wins_outright = left_ability_values > right_ability_values
        abilities_are_tied = left_ability_values == right_ability_values
        tie_breaking_flips = jax.random.bernoulli(
            battle_rng_subkeys[coop_index], 0.5, (MAX_PAIRS_PER_COOP,)
        )
        left_agent_wins = left_wins_outright | (abilities_are_tied & tie_breaking_flips)

        # +1 where left won, -1 where right won, 0 for sentinel pairs
        signed_left_perspective = (
            jnp.where(left_agent_wins, jnp.int8(1), jnp.int8(-1)) * valid_pair_mask_int8
        )

        tournament_outcomes = (
            tournament_outcomes
            .at[safe_left_agents,  safe_right_agents, coop_index].add( signed_left_perspective)
            .at[safe_right_agents, safe_left_agents,  coop_index].add(-signed_left_perspective)
        )

        left_agent_lost  = (~left_agent_wins) & valid_pair_mask_bool
        right_agent_lost =   left_agent_wins  & valid_pair_mask_bool

        tournament_losses_per_coop = (
            tournament_losses_per_coop
            .at[safe_left_agents,  coop_index].add(left_agent_lost.astype(jnp.int32))
            .at[safe_right_agents, coop_index].add(right_agent_lost.astype(jnp.int32))
        )

        # Use .add rather than .set — sentinel pairs add 0, no collision risk
        tournament_fought_matrix = (
            tournament_fought_matrix
            .at[safe_left_agents,  safe_right_agents, coop_index].add(valid_pair_mask_int8)
            .at[safe_right_agents, safe_left_agents,  coop_index].add(valid_pair_mask_int8)
        )

    return state._replace(
        tournament_outcomes=tournament_outcomes,
        tournament_losses_per_coop=tournament_losses_per_coop,
        tournament_fought_matrix=tournament_fought_matrix,
    )


# ─── Event: RegionView ─────────────────────────────────────────────────────────

def region_view(zone_array, action_array):
    """
    Determine observation coverage for this step (does not accumulate; caller
    adds the result to tournament_obs_coverage).

    Battle-zone agents observe their coop automatically.
    Spectators with a watch action observe their chosen coop.
    Spectators with a move action observe nothing — they are leaving this round.
    """
    battle_zone_coverage = zone_array[:, :, ZONE_BATTLE].astype(jnp.int8)   # (M, N)

    is_watching_action   = (action_array >= ACTION_WATCH_BASE) & (action_array < ACTION_MOVE_BASE)
    watched_coop_index   = jnp.clip(action_array - ACTION_WATCH_BASE, 0, NUM_COOPS - 1)
    watched_coop_onehot  = jax.nn.one_hot(watched_coop_index, NUM_COOPS, dtype=jnp.int8)  # (M, N)

    is_in_spectator_zone = (zone_array[:, :, ZONE_SPECTATOR].sum(axis=1) > 0)[:, None]   # (M, 1)
    spectator_coverage   = (
        watched_coop_onehot
        * (is_in_spectator_zone & is_watching_action[:, None]).astype(jnp.int8)
    )

    return jnp.clip(battle_zone_coverage + spectator_coverage, 0, 1).astype(jnp.int8)


# ─── Event: RelocateRegion ─────────────────────────────────────────────────────

def relocate_region(zone_array, action_array, transit_destination_coop):
    """
    Spectating agents who chose a move action leave their SpectatorZone and enter
    the TransitZone of their current coop.  They will arrive at their destination
    BattleZone at the start of the next step (Phase 0 of env_step).
    """
    is_move_action           = action_array >= ACTION_MOVE_BASE
    destination_coop_indices = jnp.clip(action_array - ACTION_MOVE_BASE, 0, NUM_COOPS - 1)

    is_in_spectator_zone  = zone_array[:, :, ZONE_SPECTATOR].sum(axis=1) > 0   # (M,)
    should_enter_transit  = is_in_spectator_zone & is_move_action               # (M,)

    current_coop_per_agent = jnp.argmax(zone_array.sum(axis=2), axis=1)         # (M,)
    departure_coop_onehot  = jax.nn.one_hot(current_coop_per_agent, NUM_COOPS, dtype=jnp.int8)
    departure_mask         = departure_coop_onehot * should_enter_transit[:, None].astype(jnp.int8)

    updated_zone_array = (
        zone_array
        .at[:, :, ZONE_SPECTATOR].add(-departure_mask)
        .at[:, :, ZONE_TRANSIT].add(departure_mask)
    )
    updated_transit_destination = jnp.where(
        should_enter_transit, destination_coop_indices, transit_destination_coop
    )
    return updated_zone_array, updated_transit_destination


# ─── Action policy (vectorized over all agents) ────────────────────────────────

def compute_all_agent_actions(zone_array, agent_abilities, king_beliefs):
    """
    Every agent scores each coop as:
        0.65 × (own_ability / NUM_ABILITY_LEVELS)  +  0.35 × P(self is king there)

    and either watches or moves to the highest-scoring coop.
    Battle-zone and transit agents receive ACTION_NO_OP.
    """
    agent_index_range = jnp.arange(NUM_AGENTS)

    # Diagonal extraction: king_beliefs[m, m, k, 1] = agent m's belief it will be king in coop k
    self_king_probability_per_coop = king_beliefs[agent_index_range, agent_index_range, :, 1]  # (M, N)

    coop_combined_score = (
        ABILITY_SCORE_POLICY_WEIGHT * agent_abilities.astype(jnp.float32) / NUM_ABILITY_LEVELS
        + KING_BELIEF_POLICY_WEIGHT  * self_king_probability_per_coop
    )  # (M, N)

    best_coop_per_agent    = jnp.argmax(coop_combined_score, axis=1)             # (M,)
    current_coop_per_agent = jnp.argmax(zone_array.sum(axis=2), axis=1)          # (M,)
    already_in_best_coop   = best_coop_per_agent == current_coop_per_agent

    spectator_action = jnp.where(
        already_in_best_coop,
        (ACTION_WATCH_BASE + best_coop_per_agent).astype(jnp.int32),
        (ACTION_MOVE_BASE  + best_coop_per_agent).astype(jnp.int32),
    )

    is_in_spectator_zone = zone_array[:, :, ZONE_SPECTATOR].sum(axis=1) > 0     # (M,)
    return jnp.where(is_in_spectator_zone, spectator_action, jnp.int32(ACTION_NO_OP))


# ─── Environment step ──────────────────────────────────────────────────────────

def env_step(state):
    """
    One simulation step, five ordered phases:

      Phase 0   Transit arrivals   — agents completing transit enter ZONE_BATTLE.
      Phase 1   Cage assignment    — eligible BATTLE agents paired; unpaired → SPECTATOR.
      Phase 2   Action selection   — spectators choose watch or move target.
      Phase 3a  Battles            — paired agents fight; outcomes and losses recorded.
      Phase 3b  Region view        — observation coverage updated before anyone moves.
      Phase 3c  Relocation         — move-action spectators enter ZONE_TRANSIT.
    """
    rng_key, step_specific_key           = jax.random.split(state.rng_key)
    assign_phase_key, battle_phase_key   = jax.random.split(step_specific_key)
    assign_rng_subkeys = jax.random.split(assign_phase_key, NUM_COOPS)
    battle_rng_subkeys = jax.random.split(battle_phase_key, NUM_COOPS)

    # ── Phase 0: transit arrivals ─────────────────────────────────────────────
    is_in_transit_per_agent  = state.zone_array[:, :, ZONE_TRANSIT].sum(axis=1) > 0   # (M,)
    departure_coop_per_agent = jnp.argmax(state.zone_array[:, :, ZONE_TRANSIT], axis=1)
    safe_destination_coop    = jnp.clip(state.transit_destination_coop, 0, NUM_COOPS - 1)

    transit_agent_mask = is_in_transit_per_agent[:, None].astype(jnp.int8)
    departure_onehot   = jax.nn.one_hot(departure_coop_per_agent, NUM_COOPS, dtype=jnp.int8)
    arrival_onehot     = jax.nn.one_hot(safe_destination_coop,    NUM_COOPS, dtype=jnp.int8)

    zone_after_transit_resolved = (
        state.zone_array
        .at[:, :, ZONE_TRANSIT].add(-departure_onehot * transit_agent_mask)
        .at[:, :, ZONE_BATTLE].add(   arrival_onehot  * transit_agent_mask)
    )
    transit_destination_after_arrivals = jnp.where(
        is_in_transit_per_agent, jnp.int32(-1), state.transit_destination_coop
    )

    state_after_arrivals = state._replace(
        zone_array=zone_after_transit_resolved,
        transit_destination_coop=transit_destination_after_arrivals,
        rng_key=rng_key,
    )

    # ── Phase 1: cage assignment ──────────────────────────────────────────────
    state_after_assignment, all_coop_pairs, all_pair_counts = assign_agents_to_cage(
        state_after_arrivals, assign_rng_subkeys
    )

    # ── Phase 2: action selection ─────────────────────────────────────────────
    new_action_array = compute_all_agent_actions(
        state_after_assignment.zone_array,
        state_after_assignment.agent_abilities,
        state_after_assignment.king_beliefs,
    )
    state_after_actions = state_after_assignment._replace(action_array=new_action_array)

    # ── Phase 3a: battles ─────────────────────────────────────────────────────
    state_after_battles = dominance_battle(
        state_after_actions, all_coop_pairs, all_pair_counts, battle_rng_subkeys
    )

    # ── Phase 3b: observation update (before any movement) ───────────────────
    new_step_obs_coverage = region_view(
        state_after_battles.zone_array,
        state_after_battles.action_array,
    )
    accumulated_obs_coverage = jnp.clip(
        state_after_battles.tournament_obs_coverage + new_step_obs_coverage, 0, 1
    ).astype(jnp.int8)

    # ── Phase 3c: movers enter transit ───────────────────────────────────────
    zone_after_movement, updated_transit_destination = relocate_region(
        state_after_battles.zone_array,
        state_after_battles.action_array,
        state_after_battles.transit_destination_coop,
    )

    return state_after_battles._replace(
        zone_array=zone_after_movement,
        transit_destination_coop=updated_transit_destination,
        tournament_obs_coverage=accumulated_obs_coverage,
    )


# ─── Tournament termination check ──────────────────────────────────────────────

def can_any_coop_pair(state):
    """True iff at least one coop still has two eligible agents who have not yet fought."""
    eligible_in_coop = (
        (state.zone_array[:, :, ZONE_BATTLE] == 1)
        & (state.tournament_losses_per_coop < MAX_LOSSES_BEFORE_ELIMINATION)
    )  # (M, N)

    valid_pairing_exists = (
        eligible_in_coop[:, None, :]                   # (M, 1, N)
        & eligible_in_coop[None, :, :]                 # (1, M, N)
        & (state.tournament_fought_matrix == 0)        # (M, M, N)
        & ~jnp.eye(NUM_AGENTS, dtype=jnp.bool_)[:, :, None]  # (M, M, 1)
    ).any()

    return valid_pairing_exists


# ─── Bayesian belief updates ───────────────────────────────────────────────────

def update_ability_beliefs(state):
    """
    Update every observer's posterior over every agent's ability using the
    battles witnessed this tournament.

    For observer a watching agent i face opponent j in coop k:

        P(ability_i = v | observations)
            ∝ prior(v)
            × ∏_{j : i beat j}  P(ability v beats j | a's beliefs about j)
            × ∏_{j : j beat i}  P(ability v loses to j | a's beliefs about j)

    Win probability for hypothetical ability v against opponent j:
        P(v beats j) = P(ability_j < v) + 0.5 × P(ability_j = v)
                     = CDF_j(v−1) + 0.5 × belief_j(v)

    Explaining away: defeating a high-inferred-ability opponent contributes a
    steeper log-likelihood gradient across v than defeating a weak opponent,
    because win_prob[a, strong_j, v] rises sharply with v.  Once i is inferred
    to be strong from one informative win, subsequent wins against weak opponents
    contribute comparatively little — they are explained away by the existing
    strong-ability hypothesis.
    """
    ability_beliefs = state.ability_beliefs
    is_self_pair    = jnp.eye(NUM_AGENTS, dtype=jnp.bool_)  # (M, M)

    for coop_index in range(NUM_COOPS):
        outcomes_this_coop = state.tournament_outcomes[:, :, coop_index].astype(jnp.float32)
        wins_this_coop     = jnp.maximum( outcomes_this_coop, 0.0)   # [i,j]=1 if i beat j
        losses_this_coop   = jnp.maximum(-outcomes_this_coop, 0.0)   # [i,j]=1 if j beat i

        current_beliefs_this_coop = ability_beliefs[:, :, coop_index, :]  # (M, M, L)

        # CDF_j(v−1): probability that j's ability is strictly less than v, per observer a
        cumulative_belief = jnp.cumsum(current_beliefs_this_coop, axis=-1)  # (M, M, L)
        cdf_strictly_less = jnp.concatenate(
            [jnp.zeros_like(cumulative_belief[:, :, :1]), cumulative_belief[:, :, :-1]],
            axis=-1,
        )  # (M, M, L) — cdf_strictly_less[a, j, v] = P(ability_j < v | observer a)

        win_probability  = cdf_strictly_less + 0.5 * current_beliefs_this_coop   # (M, M, L)
        loss_probability = 1.0 - win_probability

        log_win_probability  = jnp.maximum(jnp.log(win_probability  + 1e-30), LOG_PROBABILITY_FLOOR)
        log_loss_probability = jnp.maximum(jnp.log(loss_probability + 1e-30), LOG_PROBABILITY_FLOOR)

        # einsum 'ij,ajv->aiv':
        #   for each (observer a, agent i, hypothetical ability v):
        #   accumulate log P(ability v beats/loses to each specific opponent j that i beat/lost to)
        log_likelihood_from_wins   = jnp.einsum('ij,ajv->aiv', wins_this_coop,   log_win_probability)
        log_likelihood_from_losses = jnp.einsum('ij,ajv->aiv', losses_this_coop, log_loss_probability)
        total_log_likelihood       = log_likelihood_from_wins + log_likelihood_from_losses  # (M, M, L)

        log_prior_this_coop        = jnp.maximum(jnp.log(current_beliefs_this_coop + 1e-30), LOG_PROBABILITY_FLOOR)
        log_unnormalized_posterior = log_prior_this_coop + total_log_likelihood
        log_partition              = jax.nn.logsumexp(log_unnormalized_posterior, axis=-1, keepdims=True)
        normalized_posterior       = jnp.exp(log_unnormalized_posterior - log_partition)

        # Apply only to observers who witnessed this coop this tournament
        observed_this_coop = state.tournament_obs_coverage[:, coop_index].astype(jnp.bool_)  # (M,)
        updated_beliefs    = jnp.where(
            observed_this_coop[:, None, None],
            normalized_posterior,
            current_beliefs_this_coop,
        )

        # Restore exact self-knowledge: each agent knows its own ability precisely.
        # The log-prior being -∞ at wrong values naturally preserves this, but
        # we enforce it explicitly for numerical exactness.
        self_ability_one_hot = jax.nn.one_hot(
            state.agent_abilities[:, coop_index], NUM_ABILITY_LEVELS
        )  # (M, L)
        updated_beliefs = jnp.where(
            is_self_pair[:, :, None],
            self_ability_one_hot[:, None, :],   # (M, 1, L) → broadcasts to (M, M, L)
            updated_beliefs,
        )

        ability_beliefs = ability_beliefs.at[:, :, coop_index, :].set(updated_beliefs)

    return state._replace(ability_beliefs=ability_beliefs)


def update_king_beliefs(state):
    """
    Derive king-belief posteriors from current ability beliefs and zone assignments.
    Called after close_tournament, so zone_array reflects the next tournament's layout.

    For observer a and candidate j in coop k:
        P(j is king) ≈ P(j has maximum ability in coop k)
            = Σ_v  belief[a,j,k,v]  ×  ∏_{i≠j, i∈coop k}  CDF[a,i,k,v]

    Explaining away: if another agent i in the same coop is already believed
    strong, CDF[a,i,k,v] is small at moderate v, suppressing j's king
    probability.  Increasing i's inferred ability directly decreases j's
    king probability — competing candidates explain each other away.
    """
    king_beliefs = state.king_beliefs

    for coop_index in range(NUM_COOPS):
        in_coop_indicator = state.zone_array[:, coop_index, ZONE_BATTLE].astype(jnp.float32)  # (M,)
        beliefs_this_coop = state.ability_beliefs[:, :, coop_index, :]                        # (M, M, L)

        cumulative_distribution = jnp.cumsum(beliefs_this_coop, axis=-1)  # (M, M, L)
        log_cdf = jnp.maximum(jnp.log(cumulative_distribution + 1e-30), LOG_PROBABILITY_FLOOR)

        # Sum of log-CDFs for all coop members, per observer and ability level
        log_total_cdf_sum = jnp.einsum('j,ajv->av', in_coop_indicator, log_cdf)  # (M, L)

        # Subtract target j's contribution → log-product of all *other* coop members' CDFs
        log_product_of_other_cdfs = (
            log_total_cdf_sum[:, None, :]                               # (M, 1, L)
            - in_coop_indicator[None, :, None] * log_cdf               # (M, M, L)
        )  # (M, M, L)

        # Marginalize over j's own ability: P(j = maximum in coop | observer a)
        king_probability = jnp.sum(
            beliefs_this_coop * jnp.exp(log_product_of_other_cdfs), axis=-1
        )  # (M, M)
        king_probability = jnp.clip(king_probability, 0.0, 1.0)

        king_beliefs = king_beliefs.at[:, :, coop_index, :].set(
            jnp.stack([1.0 - king_probability, king_probability], axis=-1)
        )

    return state._replace(king_beliefs=king_beliefs)


# ─── Tournament close and reset (CloseTournament + NewTournament events) ────────

def close_tournament(state, rng_subkey):
    """
    1. Write this tournament's outcomes and observation mask to history arrays.
    2. Crown assignment: undefeated participants first; fallback to max-wins participant.
    3. Redistribute: crowned agents → most-battled coop; uncrowned → uniform random.
    4. Reset all per-tournament tracking fields.
    5. Increment tournament_index.
    """
    tournament_idx = state.tournament_index

    battle_history = state.battle_history.at[:, :, :, tournament_idx].set(state.tournament_outcomes)
    obs_mask       = state.obs_mask.at[:, :, tournament_idx].set(state.tournament_obs_coverage)

    wins_per_agent_per_coop    = jnp.maximum( state.tournament_outcomes, 0).sum(axis=1)   # (M, N)
    losses_per_agent_per_coop  = jnp.maximum(-state.tournament_outcomes, 0).sum(axis=1)   # (M, N)
    battles_per_agent_per_coop = wins_per_agent_per_coop + losses_per_agent_per_coop       # (M, N)
    participated_in_coop       = battles_per_agent_per_coop > 0                            # (M, N)

    undefeated_in_coop     = participated_in_coop & (losses_per_agent_per_coop == 0)
    any_undefeated_in_coop = undefeated_in_coop.any(axis=0)                                # (N,)
    max_wins_per_coop      = wins_per_agent_per_coop.max(axis=0, keepdims=True)            # (1, N)
    most_wins_in_coop      = participated_in_coop & (wins_per_agent_per_coop == max_wins_per_coop)

    crown_this_tournament = jnp.where(
        any_undefeated_in_coop[None, :], undefeated_in_coop, most_wins_in_coop
    )  # (M, N)
    new_crown_counts = state.crown_counts + crown_this_tournament.astype(jnp.int32)

    noise_rng, random_assignment_rng = jax.random.split(rng_subkey)
    tie_breaking_noise = jax.random.uniform(noise_rng, (NUM_AGENTS, NUM_COOPS)) * 0.1

    preferred_coop_per_agent = jnp.argmax(
        battles_per_agent_per_coop.astype(jnp.float32) + tie_breaking_noise, axis=1
    )  # (M,) — most-battled coop, ties broken stochastically
    random_coop_per_agent = jax.random.randint(
        random_assignment_rng, (NUM_AGENTS,), 0, NUM_COOPS
    )

    has_any_crown      = crown_this_tournament.any(axis=1)                                 # (M,)
    new_coop_per_agent = jnp.where(has_any_crown, preferred_coop_per_agent, random_coop_per_agent)

    new_zone_array = jnp.zeros((NUM_AGENTS, NUM_COOPS, 3), dtype=jnp.int8).at[
        :, :, ZONE_BATTLE
    ].set(jax.nn.one_hot(new_coop_per_agent, NUM_COOPS, dtype=jnp.int8))

    return state._replace(
        zone_array=new_zone_array,
        transit_destination_coop=jnp.full((NUM_AGENTS,),                   -1, jnp.int32),
        action_array=jnp.zeros((NUM_AGENTS,),                                  jnp.int32),
        tournament_losses_per_coop=jnp.zeros((NUM_AGENTS, NUM_COOPS),          jnp.int32),
        tournament_fought_matrix=jnp.zeros((NUM_AGENTS, NUM_AGENTS, NUM_COOPS), jnp.int8),
        tournament_outcomes=jnp.zeros((NUM_AGENTS, NUM_AGENTS, NUM_COOPS),     jnp.int8),
        tournament_obs_coverage=jnp.zeros((NUM_AGENTS, NUM_COOPS),             jnp.int8),
        battle_history=battle_history,
        obs_mask=obs_mask,
        crown_counts=new_crown_counts,
        tournament_index=tournament_idx + 1,
        step_done_flag=jnp.bool_(False),
    )


# ─── Tournament and simulation loops ───────────────────────────────────────────

def _run_one_step_body(step_index, state):
    """
    Inner lax.fori_loop body.  Guards on step_done_flag first to avoid
    redundant pairing checks after the tournament has already terminated.
    """
    def run_env_step(_):
        return env_step(state)

    def mark_tournament_done(_):
        return state._replace(step_done_flag=jnp.bool_(True))

    def check_pairing_and_step(_):
        return lax.cond(can_any_coop_pair(state), run_env_step, mark_tournament_done, None)

    def no_op(_):
        return state

    return lax.cond(state.step_done_flag, no_op, check_pairing_and_step, None)


def _run_one_tournament_body(tournament_loop_index, state):
    """
    Outer lax.fori_loop body.

    Ordering matters:
      1. Run steps — accumulate tournament_outcomes and tournament_obs_coverage.
      2. Update ability beliefs — must happen before close_tournament zeroes those fields.
      3. Close tournament — write history, crown agents, redistribute, reset fields.
      4. Update king beliefs — must happen after close_tournament sets new zone_array.
    """
    state_with_fresh_step_flag = state._replace(step_done_flag=jnp.bool_(False))

    state_after_steps = lax.fori_loop(
        0, MAX_STEPS_PER_TOURNAMENT, _run_one_step_body, state_with_fresh_step_flag
    )

    state_after_ability_update = update_ability_beliefs(state_after_steps)

    rng_key, close_rng_subkey = jax.random.split(state_after_ability_update.rng_key)
    state_after_close = close_tournament(
        state_after_ability_update._replace(rng_key=rng_key), close_rng_subkey
    )

    return update_king_beliefs(state_after_close)


@jax.jit
def run_simulation(initial_state):
    return lax.fori_loop(0, NUM_TOURNAMENTS, _run_one_tournament_body, initial_state)


# ─── Initialization ────────────────────────────────────────────────────────────

def initialize_pec_king_environment(seed=0):
    """
    Build the initial SimulationState and Python infrastructure objects.
    Not JIT-compiled; called once before run_simulation.

    Returns (initial_state, cages, monitors).
    """
    base_rng_key = jax.random.PRNGKey(seed)
    ability_rng_key, coop_rng_key, simulation_rng_key = jax.random.split(base_rng_key, 3)

    # Sample 0-indexed abilities from the truncated Poisson prior
    agent_abilities = jax.random.categorical(
        ability_rng_key,
        jnp.tile(ABILITY_PRIOR_LOG_PROBS[None, None, :], (NUM_AGENTS, NUM_COOPS, 1)),
        axis=-1,
    ).astype(jnp.int32)  # (NUM_AGENTS, NUM_COOPS)

    initial_coop_assignments = jax.random.randint(coop_rng_key, (NUM_AGENTS,), 0, NUM_COOPS)
    initial_zone_array = jnp.zeros(
        (NUM_AGENTS, NUM_COOPS, 3), dtype=jnp.int8
    ).at[:, :, ZONE_BATTLE].set(
        jax.nn.one_hot(initial_coop_assignments, NUM_COOPS, dtype=jnp.int8)
    )

    # Prior for all observer-target pairs; diagonal overwritten with exact self-knowledge
    initial_ability_beliefs = jnp.tile(
        jax.nn.softmax(ABILITY_PRIOR_LOG_PROBS)[None, None, None, :],
        (NUM_AGENTS, NUM_AGENTS, NUM_COOPS, 1),
    )  # (M, M, N, L)

    self_ability_one_hot = jax.nn.one_hot(agent_abilities, NUM_ABILITY_LEVELS)  # (M, N, L)
    is_self_pair         = jnp.eye(NUM_AGENTS, dtype=jnp.bool_)
    initial_ability_beliefs = jnp.where(
        is_self_pair[:, :, None, None],
        self_ability_one_hot[:, None, :, :],   # (M, 1, N, L) → (M, M, N, L)
        initial_ability_beliefs,
    )

    initial_state = SimulationState(
        zone_array=initial_zone_array,
        transit_destination_coop=jnp.full((NUM_AGENTS,), -1, jnp.int32),
        action_array=jnp.zeros((NUM_AGENTS,), jnp.int32),
        agent_abilities=agent_abilities,
        tournament_losses_per_coop=jnp.zeros((NUM_AGENTS, NUM_COOPS), jnp.int32),
        tournament_fought_matrix=jnp.zeros((NUM_AGENTS, NUM_AGENTS, NUM_COOPS), jnp.int8),
        tournament_outcomes=jnp.zeros((NUM_AGENTS, NUM_AGENTS, NUM_COOPS), jnp.int8),
        tournament_obs_coverage=jnp.zeros((NUM_AGENTS, NUM_COOPS), jnp.int8),
        battle_history=jnp.zeros((NUM_AGENTS, NUM_AGENTS, NUM_COOPS, NUM_TOURNAMENTS), jnp.int8),
        obs_mask=jnp.zeros((NUM_AGENTS, NUM_COOPS, NUM_TOURNAMENTS), jnp.int8),
        ability_beliefs=initial_ability_beliefs,
        king_beliefs=jnp.full((NUM_AGENTS, NUM_AGENTS, NUM_COOPS, 2), 0.5, jnp.float32),
        crown_counts=jnp.zeros((NUM_AGENTS, NUM_COOPS), jnp.int32),
        tournament_index=jnp.int32(0),
        step_done_flag=jnp.bool_(False),
        rng_key=simulation_rng_key,
    )

    cages = [
        Cage(
            cage_id=coop_index * (NUM_AGENTS // 2) + cage_slot,
            home_coop_index=coop_index,
        )
        for coop_index in range(NUM_COOPS)
        for cage_slot in range(NUM_AGENTS // 2)
    ]

    monitors = [
        Monitor(
            monitor_id=coop_index * NUM_COOPS + viewing_coop_index,
            installed_in_coop_index=coop_index,
            viewing_coop_index=viewing_coop_index,
        )
        for coop_index in range(NUM_COOPS)
        for viewing_coop_index in range(NUM_COOPS)
    ]

    return initial_state, cages, monitors
