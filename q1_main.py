import time
from dataclasses import dataclass
from typing import NamedTuple, Tuple

import jax
import jax.lax as lax
import jax.nn
import jax.numpy as jnp
import jax.random
import numpy as np
import numpy.typing as npt


NUM_AGENTS = 64
NUM_COOPS = 4
MAX_LOSSES_BEFORE_ELIMINATION = 4
NUM_TOURNAMENTS = 1000
MAX_STEPS_PER_TOURNAMENT = 50
NUM_ABILITY_LEVELS = 4
MAX_PAIRS_PER_COOP = 32

ZONE_BATTLE = 0
ZONE_SPECTATOR = 1
ZONE_TRANSIT = 2

ACTION_NO_OP = 0
ACTION_WATCH_BASE = 1
ACTION_MOVE_BASE = ACTION_WATCH_BASE + NUM_COOPS

ABILITY_SCORE_POLICY_WEIGHT = 0.65
KING_BELIEF_POLICY_WEIGHT = 0.35
PAIR_SENTINEL = NUM_AGENTS


def clip_open_probability(probability_tensor: jnp.ndarray) -> jnp.ndarray:
    # Clamp probabilities.
    return jnp.clip(probability_tensor, 1e-38, 1.0 - 1e-7)


def compute_empirical_crown_frequency(
    crown_history: jnp.ndarray, completed_tournaments: jnp.ndarray
) -> jnp.ndarray:
    # Compute crown frequencies.
    crown_counts = crown_history.astype(jnp.float32).sum(axis=-1)
    return (crown_counts + 1.0) / (completed_tournaments.astype(jnp.float32) + 2.0)


def compute_ability_prior_log_probs() -> jnp.ndarray:
    # Compute the ability prior.
    ability_level_values = jnp.arange(1, NUM_ABILITY_LEVELS + 1, dtype=jnp.float32)
    poisson_rate_parameter = float(NUM_COOPS) / 3.0
    unnormalized_log_weights = -ability_level_values / poisson_rate_parameter
    return unnormalized_log_weights - jax.nn.logsumexp(unnormalized_log_weights)


ABILITY_PRIOR_LOG_PROBS: jnp.ndarray = compute_ability_prior_log_probs()


@dataclass
class Location:
    # Entity location.

    coop_index: int
    zone_index: int
    transit_target_coop: int = -1


@dataclass
class Agent:
    # Chicken state.

    agent_index: int
    observation_memory: np.ndarray
    ability_belief: np.ndarray
    my_ability: np.ndarray
    my_location: Location
    last_observation: np.ndarray
    king_belief: np.ndarray

    def action_policy(self) -> Tuple[str, int]:
        # Choose an action.
        if self.my_location.zone_index != ZONE_SPECTATOR:
            return ("no_op", -1)

        self_king_probability_per_coop = self.king_belief[self.agent_index, :, 1]
        normalized_ability_per_coop = self.my_ability.astype(np.float32) / NUM_ABILITY_LEVELS
        policy_scores = (
            ABILITY_SCORE_POLICY_WEIGHT * normalized_ability_per_coop
            + KING_BELIEF_POLICY_WEIGHT * self_king_probability_per_coop
        )
        best_coop_index = int(policy_scores.argmax())
        if best_coop_index == self.my_location.coop_index:
            return ("watch", best_coop_index)
        return ("move", best_coop_index)


@dataclass
class Cage:
    # Cage state.

    cage_index: int
    coop_index: int
    occupants: Tuple[int, int] | None = None


@dataclass
class Monitor:
    # Monitor state.

    monitor_index: int
    coop_index: int
    viewed_coop_index: int


class SimulationState(NamedTuple):
    zone_array: jnp.ndarray
    transit_destination_coop: jnp.ndarray
    action_array: jnp.ndarray
    agent_abilities: jnp.ndarray
    tournament_losses_per_coop: jnp.ndarray
    tournament_fought_matrix: jnp.ndarray
    tournament_outcomes: jnp.ndarray
    tournament_step_obs_coverage: jnp.ndarray
    tournament_step_outcomes: jnp.ndarray
    battle_outcome: jnp.ndarray
    battle_history: jnp.ndarray
    obs_mask: jnp.ndarray
    last_tournament_observed_outcomes: jnp.ndarray
    ability_beliefs: jnp.ndarray
    king_beliefs: jnp.ndarray
    crown_history: jnp.ndarray
    tournament_index: jnp.ndarray
    tournament_step_index: jnp.ndarray
    step_done_flag: jnp.ndarray
    rng_key: jnp.ndarray


def _build_pairs_for_coop(
    zone_array: jnp.ndarray,
    tournament_losses_per_coop: jnp.ndarray,
    tournament_fought_matrix: jnp.ndarray,
    coop_index: int,
    rng_subkey: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Build pairings for one coop.
    in_battle_zone_this_coop = zone_array[:, coop_index, ZONE_BATTLE].astype(jnp.bool_)
    under_loss_limit_this_coop = (
        tournament_losses_per_coop[:, coop_index] < MAX_LOSSES_BEFORE_ELIMINATION
    )
    eligible_to_battle = in_battle_zone_this_coop & under_loss_limit_this_coop

    tie_breaking_noise = jax.random.uniform(rng_subkey, (NUM_AGENTS,)) * 0.1
    sort_keys = jnp.where(
        eligible_to_battle,
        tournament_losses_per_coop[:, coop_index].astype(jnp.float32) + tie_breaking_noise,
        jnp.float32(NUM_AGENTS + 1),
    )
    sorted_agent_indices = jnp.argsort(sort_keys)
    fought_in_this_coop = tournament_fought_matrix[:, :, coop_index]

    def scan_body(
        carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        scan_position: jnp.ndarray,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        taken_mask, pairs_array, pair_count = carry
        current_agent = sorted_agent_indices[scan_position]
        is_eligible_and_free = eligible_to_battle[current_agent] & ~taken_mask[current_agent]

        already_fought_current_agent = fought_in_this_coop[:, current_agent]
        valid_partner_mask = (
            eligible_to_battle
            & ~taken_mask
            & (already_fought_current_agent == 0)
            & (jnp.arange(NUM_AGENTS, dtype=jnp.int32) != current_agent)
        )
        do_pair = is_eligible_and_free & valid_partner_mask.any()

        partner_search_cost = jnp.where(
            valid_partner_mask,
            jnp.arange(NUM_AGENTS, dtype=jnp.int32),
            jnp.int32(NUM_AGENTS),
        )
        first_valid_partner = jnp.argmin(partner_search_cost).astype(jnp.int32)

        safe_write_position = jnp.minimum(pair_count, jnp.int32(MAX_PAIRS_PER_COOP - 1))
        current_pair_row = jnp.stack([current_agent.astype(jnp.int32), first_valid_partner])
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


def assign_agents_to_cage(
    state: SimulationState,
    rng_subkeys_per_coop: jnp.ndarray,
) -> Tuple[SimulationState, jnp.ndarray, jnp.ndarray]:
    # Assign agents to cages.
    zone_array = state.zone_array
    pairs_list = []
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
            zone_array.at[:, coop_index, ZONE_BATTLE].add(-unpaired_mask_int8).at[
                :, coop_index, ZONE_SPECTATOR
            ].add(unpaired_mask_int8)
        )

    all_coop_pairs = jnp.stack(pairs_list, axis=0)
    all_pair_counts = jnp.stack(pair_counts_list, axis=0)
    return state._replace(zone_array=zone_array), all_coop_pairs, all_pair_counts


def dominance_battle(
    state: SimulationState,
    all_coop_pairs: jnp.ndarray,
    all_pair_counts: jnp.ndarray,
    battle_rng_subkeys: jnp.ndarray,
) -> Tuple[SimulationState, jnp.ndarray]:
    # Resolve battles.
    tournament_outcomes = state.tournament_outcomes
    tournament_losses_per_coop = state.tournament_losses_per_coop
    tournament_fought_matrix = state.tournament_fought_matrix
    step_outcomes = jnp.zeros((NUM_AGENTS, NUM_AGENTS, NUM_COOPS), dtype=jnp.int8)

    for coop_index in range(NUM_COOPS):
        pairs_this_coop = all_coop_pairs[coop_index]
        pair_count_this_coop = all_pair_counts[coop_index]

        valid_pair_mask_bool = jnp.arange(MAX_PAIRS_PER_COOP) < pair_count_this_coop
        valid_pair_mask_int8 = valid_pair_mask_bool.astype(jnp.int8)

        safe_left_agents = jnp.clip(pairs_this_coop[:, 0], 0, NUM_AGENTS - 1)
        safe_right_agents = jnp.clip(pairs_this_coop[:, 1], 0, NUM_AGENTS - 1)

        left_ability_values = state.agent_abilities[safe_left_agents, coop_index]
        right_ability_values = state.agent_abilities[safe_right_agents, coop_index]

        left_wins_outright = left_ability_values > right_ability_values
        abilities_are_tied = left_ability_values == right_ability_values
        tie_breaking_flips = jax.random.bernoulli(
            battle_rng_subkeys[coop_index], 0.5, (MAX_PAIRS_PER_COOP,)
        )
        left_agent_wins = left_wins_outright | (abilities_are_tied & tie_breaking_flips)

        signed_left_perspective = (
            jnp.where(left_agent_wins, jnp.int8(1), jnp.int8(-1)) * valid_pair_mask_int8
        )

        tournament_outcomes = (
            tournament_outcomes.at[safe_left_agents, safe_right_agents, coop_index].add(
                signed_left_perspective
            ).at[safe_right_agents, safe_left_agents, coop_index].add(-signed_left_perspective)
        )
        step_outcomes = (
            step_outcomes.at[safe_left_agents, safe_right_agents, coop_index].add(
                signed_left_perspective
            ).at[safe_right_agents, safe_left_agents, coop_index].add(-signed_left_perspective)
        )

        left_agent_lost = (~left_agent_wins) & valid_pair_mask_bool
        right_agent_lost = left_agent_wins & valid_pair_mask_bool

        tournament_losses_per_coop = (
            tournament_losses_per_coop.at[safe_left_agents, coop_index].add(
                left_agent_lost.astype(jnp.int32)
            ).at[safe_right_agents, coop_index].add(right_agent_lost.astype(jnp.int32))
        )

        tournament_fought_matrix = (
            tournament_fought_matrix.at[safe_left_agents, safe_right_agents, coop_index].add(
                valid_pair_mask_int8
            ).at[safe_right_agents, safe_left_agents, coop_index].add(valid_pair_mask_int8)
        )

    return (
        state._replace(
            tournament_outcomes=tournament_outcomes,
            tournament_losses_per_coop=tournament_losses_per_coop,
            tournament_fought_matrix=tournament_fought_matrix,
        ),
        step_outcomes,
    )


def compute_observation_coverage(
    zone_array: jnp.ndarray, action_array: jnp.ndarray
) -> jnp.ndarray:
    # Compute observation coverage.
    action_codes = action_array[:, 0]
    battle_zone_coverage = zone_array[:, :, ZONE_BATTLE].astype(jnp.int8)
    is_watching_action = (action_codes >= ACTION_WATCH_BASE) & (action_codes < ACTION_MOVE_BASE)
    watched_coop_index = jnp.clip(action_codes - ACTION_WATCH_BASE, 0, NUM_COOPS - 1)
    watched_coop_onehot = jax.nn.one_hot(watched_coop_index, NUM_COOPS, dtype=jnp.int8)
    is_in_spectator_zone = (zone_array[:, :, ZONE_SPECTATOR].sum(axis=1) > 0)[:, None]
    spectator_coverage = watched_coop_onehot * (
        is_in_spectator_zone & is_watching_action[:, None]
    ).astype(jnp.int8)
    return jnp.clip(battle_zone_coverage + spectator_coverage, 0, 1).astype(jnp.int8)


def initiate_transit(
    zone_array: jnp.ndarray,
    action_array: jnp.ndarray,
    transit_destination_coop: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Start transit moves.
    action_codes = action_array[:, 0]
    is_move_action = action_codes >= ACTION_MOVE_BASE
    destination_coop_indices = jnp.clip(action_codes - ACTION_MOVE_BASE, 0, NUM_COOPS - 1)
    is_in_spectator_zone = zone_array[:, :, ZONE_SPECTATOR].sum(axis=1) > 0
    should_enter_transit = is_in_spectator_zone & is_move_action

    current_coop_per_agent = jnp.argmax(zone_array.sum(axis=2), axis=1)
    departure_coop_onehot = jax.nn.one_hot(current_coop_per_agent, NUM_COOPS, dtype=jnp.int8)
    departure_mask = departure_coop_onehot * should_enter_transit[:, None].astype(jnp.int8)

    updated_zone_array = (
        zone_array.at[:, :, ZONE_SPECTATOR].add(-departure_mask).at[:, :, ZONE_TRANSIT].add(
            departure_mask
        )
    )
    updated_transit_destination = jnp.where(
        should_enter_transit, destination_coop_indices, transit_destination_coop
    )
    return updated_zone_array, updated_transit_destination


def compute_policy_scores(
    agent_abilities: jnp.ndarray,
    king_beliefs: jnp.ndarray,
) -> jnp.ndarray:
    # Compute policy scores.
    self_king_probability_per_coop = king_beliefs[:, :, 1]
    normalized_ability_per_coop = agent_abilities.astype(jnp.float32) / NUM_ABILITY_LEVELS
    return (
        ABILITY_SCORE_POLICY_WEIGHT * normalized_ability_per_coop
        + KING_BELIEF_POLICY_WEIGHT * self_king_probability_per_coop
    )


def compute_all_agent_actions(
    zone_array: jnp.ndarray,
    agent_abilities: jnp.ndarray,
    king_beliefs: jnp.ndarray,
) -> jnp.ndarray:
    # Compute actions.
    best_coop_per_agent = jnp.argmax(
        compute_policy_scores(agent_abilities, king_beliefs), axis=1
    )
    current_coop_per_agent = jnp.argmax(zone_array.sum(axis=2), axis=1)
    already_in_best_coop = best_coop_per_agent == current_coop_per_agent

    spectator_action = jnp.where(
        already_in_best_coop,
        (ACTION_WATCH_BASE + best_coop_per_agent).astype(jnp.int32),
        (ACTION_MOVE_BASE + best_coop_per_agent).astype(jnp.int32),
    )

    is_in_spectator_zone = zone_array[:, :, ZONE_SPECTATOR].sum(axis=1) > 0
    action_codes = jnp.where(is_in_spectator_zone, spectator_action, jnp.int32(ACTION_NO_OP))
    return action_codes[:, None]


def env_step(state: SimulationState) -> SimulationState:
    # Run one step.
    rng_key, step_specific_key = jax.random.split(state.rng_key)
    assign_phase_key, battle_phase_key = jax.random.split(step_specific_key)
    assign_rng_subkeys = jax.random.split(assign_phase_key, NUM_COOPS)
    battle_rng_subkeys = jax.random.split(battle_phase_key, NUM_COOPS)

    is_in_transit_per_agent = state.zone_array[:, :, ZONE_TRANSIT].sum(axis=1) > 0
    departure_coop_per_agent = jnp.argmax(state.zone_array[:, :, ZONE_TRANSIT], axis=1)
    safe_destination_coop = jnp.clip(state.transit_destination_coop, 0, NUM_COOPS - 1)

    transit_agent_mask = is_in_transit_per_agent[:, None].astype(jnp.int8)
    departure_onehot = jax.nn.one_hot(departure_coop_per_agent, NUM_COOPS, dtype=jnp.int8)
    arrival_onehot = jax.nn.one_hot(safe_destination_coop, NUM_COOPS, dtype=jnp.int8)

    zone_after_transit_resolved = (
        state.zone_array.at[:, :, ZONE_TRANSIT].add(-departure_onehot * transit_agent_mask).at[
            :, :, ZONE_BATTLE
        ].add(arrival_onehot * transit_agent_mask)
    )
    transit_destination_after_arrivals = jnp.where(
        is_in_transit_per_agent, jnp.int32(-1), state.transit_destination_coop
    )

    state_after_arrivals = state._replace(
        zone_array=zone_after_transit_resolved,
        transit_destination_coop=transit_destination_after_arrivals,
        rng_key=rng_key,
    )
    state_after_assignment, all_coop_pairs, all_pair_counts = assign_agents_to_cage(
        state_after_arrivals, assign_rng_subkeys
    )

    new_action_array = compute_all_agent_actions(
        state_after_assignment.zone_array,
        state_after_assignment.agent_abilities,
        state_after_assignment.king_beliefs,
    )
    state_after_actions = state_after_assignment._replace(action_array=new_action_array)
    state_after_battles, step_outcomes = dominance_battle(
        state_after_actions, all_coop_pairs, all_pair_counts, battle_rng_subkeys
    )

    new_step_obs_coverage = compute_observation_coverage(
        state_after_battles.zone_array, state_after_battles.action_array
    )
    safe_step_index = jnp.minimum(
        state_after_battles.tournament_step_index,
        jnp.int32(MAX_STEPS_PER_TOURNAMENT - 1),
    )
    updated_step_obs_coverage = state_after_battles.tournament_step_obs_coverage.at[
        safe_step_index
    ].set(new_step_obs_coverage)
    updated_step_outcomes = state_after_battles.tournament_step_outcomes.at[
        safe_step_index
    ].set(step_outcomes)

    zone_after_movement, updated_transit_destination = initiate_transit(
        state_after_battles.zone_array,
        state_after_battles.action_array,
        state_after_battles.transit_destination_coop,
    )
    return state_after_battles._replace(
        zone_array=zone_after_movement,
        transit_destination_coop=updated_transit_destination,
        tournament_step_obs_coverage=updated_step_obs_coverage,
        tournament_step_outcomes=updated_step_outcomes,
        battle_outcome=step_outcomes,
        tournament_step_index=state_after_battles.tournament_step_index + 1,
    )


def can_any_coop_pair(state: SimulationState) -> jnp.ndarray:
    # Check whether any coop can pair.
    eligible_in_coop = (
        (state.zone_array[:, :, ZONE_BATTLE] == 1)
        & (state.tournament_losses_per_coop < MAX_LOSSES_BEFORE_ELIMINATION)
    )
    valid_pairing_exists = (
        eligible_in_coop[:, None, :]
        & eligible_in_coop[None, :, :]
        & (state.tournament_fought_matrix == 0)
        & ~jnp.eye(NUM_AGENTS, dtype=jnp.bool_)[:, :, None]
    ).any()
    return valid_pairing_exists


def summarize_tournament_observations(
    state: SimulationState,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Summarize tournament observations.
    tournament_obs_coverage = jnp.clip(
        state.tournament_step_obs_coverage.sum(axis=0), 0, 1
    ).astype(jnp.int8)
    tournament_observed_outcomes = (
        state.tournament_step_obs_coverage[:, :, None, None, :].astype(jnp.int16)
        * state.tournament_step_outcomes[:, None, :, :, :].astype(jnp.int16)
    ).sum(axis=0).astype(jnp.int8)
    return tournament_obs_coverage, tournament_observed_outcomes


def update_ability_beliefs(state: SimulationState) -> SimulationState:
    # Update ability beliefs.
    ability_beliefs = state.ability_beliefs
    is_self_pair = jnp.eye(NUM_AGENTS, dtype=jnp.bool_)
    _, tournament_observed_outcomes = summarize_tournament_observations(state)

    for coop_index in range(NUM_COOPS):
        observed_outcomes_this_coop = tournament_observed_outcomes[:, :, :, coop_index].astype(
            jnp.float32
        )
        wins_this_coop = jnp.maximum(observed_outcomes_this_coop, 0.0)
        losses_this_coop = jnp.maximum(-observed_outcomes_this_coop, 0.0)
        current_beliefs_this_coop = ability_beliefs[:, :, coop_index, :]

        cumulative_belief = jnp.clip(jnp.cumsum(current_beliefs_this_coop, axis=-1), 0.0, 1.0)
        cdf_strictly_less = jnp.concatenate(
            [jnp.zeros_like(cumulative_belief[:, :, :1]), cumulative_belief[:, :, :-1]],
            axis=-1,
        )

        win_probability = clip_open_probability(
            jnp.clip(cdf_strictly_less + 0.5 * current_beliefs_this_coop, 0.0, 1.0)
        )
        log_win_probability = jnp.log(win_probability)
        log_loss_probability = jnp.log1p(-win_probability)

        win_event_mask = wins_this_coop[:, :, :, None].astype(jnp.bool_)
        loss_event_mask = losses_this_coop[:, :, :, None].astype(jnp.bool_)

        log_likelihood_from_wins = jnp.where(
            win_event_mask, log_win_probability[:, None, :, :], 0.0
        ).sum(axis=2)
        log_likelihood_from_losses = jnp.where(
            loss_event_mask, log_loss_probability[:, None, :, :], 0.0
        ).sum(axis=2)
        total_log_likelihood = log_likelihood_from_wins + log_likelihood_from_losses

        log_prior_this_coop = jnp.log(clip_open_probability(current_beliefs_this_coop))
        log_unnormalized_posterior = log_prior_this_coop + total_log_likelihood
        log_partition = jax.nn.logsumexp(
            log_unnormalized_posterior, axis=-1, keepdims=True
        )
        normalized_posterior = jnp.exp(log_unnormalized_posterior - log_partition)

        observed_this_coop = (jnp.abs(observed_outcomes_this_coop).sum(axis=(1, 2)) > 0).astype(
            jnp.bool_
        )
        updated_beliefs = jnp.where(
            observed_this_coop[:, None, None],
            normalized_posterior,
            current_beliefs_this_coop,
        )

        self_ability_one_hot = jax.nn.one_hot(
            state.agent_abilities[:, coop_index], NUM_ABILITY_LEVELS
        )
        updated_beliefs = jnp.where(
            is_self_pair[:, :, None],
            self_ability_one_hot[:, None, :],
            updated_beliefs,
        )

        ability_beliefs = ability_beliefs.at[:, :, coop_index, :].set(updated_beliefs)

    return state._replace(ability_beliefs=ability_beliefs)


def update_king_beliefs(state: SimulationState) -> SimulationState:
    # Update king beliefs.
    posterior_predictive_king_probability = compute_empirical_crown_frequency(
        state.crown_history, state.tournament_index
    )
    king_beliefs = jnp.stack(
        [1.0 - posterior_predictive_king_probability, posterior_predictive_king_probability],
        axis=-1,
    )
    return state._replace(king_beliefs=king_beliefs)


def compute_win_beliefs(ability_beliefs: jnp.ndarray) -> jnp.ndarray:
    # Compute win beliefs.
    target_win_beliefs = ability_beliefs[:, :, None, :, :]
    target_loss_beliefs = ability_beliefs[:, None, :, :, :]
    cumulative_loss_beliefs = jnp.cumsum(target_loss_beliefs, axis=-1)
    strict_less_loss_beliefs = jnp.concatenate(
        [
            jnp.zeros_like(cumulative_loss_beliefs[..., :1]),
            cumulative_loss_beliefs[..., :-1],
        ],
        axis=-1,
    )
    return (
        target_win_beliefs * (strict_less_loss_beliefs + 0.5 * target_loss_beliefs)
    ).sum(axis=-1)


def close_tournament(state: SimulationState, rng_subkey: jnp.ndarray) -> SimulationState:
    # Close a tournament.
    tournament_idx = state.tournament_index
    tournament_obs_coverage, tournament_observed_outcomes = summarize_tournament_observations(
        state
    )
    battle_history = state.battle_history.at[:, :, :, tournament_idx].set(
        state.tournament_outcomes
    )
    obs_mask = state.obs_mask.at[:, :, tournament_idx].set(tournament_obs_coverage)
    last_tournament_observed_outcomes = tournament_observed_outcomes

    wins_per_agent_per_coop = jnp.maximum(state.tournament_outcomes, 0).sum(axis=1)
    losses_per_agent_per_coop = jnp.maximum(-state.tournament_outcomes, 0).sum(axis=1)
    battles_per_agent_per_coop = wins_per_agent_per_coop + losses_per_agent_per_coop
    participated_in_coop = battles_per_agent_per_coop > 0

    undefeated_in_coop = participated_in_coop & (losses_per_agent_per_coop == 0)
    any_undefeated_in_coop = undefeated_in_coop.any(axis=0)
    max_wins_per_coop = wins_per_agent_per_coop.max(axis=0, keepdims=True)
    most_wins_in_coop = participated_in_coop & (wins_per_agent_per_coop == max_wins_per_coop)
    crown_this_tournament = jnp.where(
        any_undefeated_in_coop[None, :], undefeated_in_coop, most_wins_in_coop
    )
    crown_history = state.crown_history.at[:, :, tournament_idx].set(
        crown_this_tournament.astype(jnp.int8)
    )

    noise_rng, random_assignment_rng = jax.random.split(rng_subkey)
    tie_breaking_noise = jax.random.uniform(noise_rng, (NUM_AGENTS, NUM_COOPS)) * 0.1
    preferred_coop_per_agent = jnp.argmax(
        battles_per_agent_per_coop.astype(jnp.float32) + tie_breaking_noise, axis=1
    )
    random_coop_per_agent = jax.random.randint(
        random_assignment_rng, (NUM_AGENTS,), 0, NUM_COOPS
    )

    has_any_crown = crown_this_tournament.any(axis=1)
    new_coop_per_agent = jnp.where(
        has_any_crown, preferred_coop_per_agent, random_coop_per_agent
    )
    new_zone_array = jnp.zeros((NUM_AGENTS, NUM_COOPS, 3), dtype=jnp.int8).at[
        :, :, ZONE_BATTLE
    ].set(jax.nn.one_hot(new_coop_per_agent, NUM_COOPS, dtype=jnp.int8))

    return state._replace(
        zone_array=new_zone_array,
        transit_destination_coop=jnp.full((NUM_AGENTS,), -1, jnp.int32),
        action_array=jnp.zeros((NUM_AGENTS, 1), jnp.int32),
        tournament_losses_per_coop=jnp.zeros((NUM_AGENTS, NUM_COOPS), jnp.int32),
        tournament_fought_matrix=jnp.zeros((NUM_AGENTS, NUM_AGENTS, NUM_COOPS), jnp.int8),
        tournament_outcomes=jnp.zeros((NUM_AGENTS, NUM_AGENTS, NUM_COOPS), jnp.int8),
        tournament_step_obs_coverage=jnp.zeros(
            (MAX_STEPS_PER_TOURNAMENT, NUM_AGENTS, NUM_COOPS), jnp.int8
        ),
        tournament_step_outcomes=jnp.zeros(
            (MAX_STEPS_PER_TOURNAMENT, NUM_AGENTS, NUM_AGENTS, NUM_COOPS), jnp.int8
        ),
        battle_outcome=state.battle_outcome,
        battle_history=battle_history,
        obs_mask=obs_mask,
        last_tournament_observed_outcomes=last_tournament_observed_outcomes,
        crown_history=crown_history,
        tournament_index=tournament_idx + 1,
        tournament_step_index=jnp.int32(0),
        step_done_flag=jnp.bool_(False),
    )


def tournament_step_body(_: int, state: SimulationState) -> SimulationState:
    # Tournament step body.

    def run_env_step(_unused: None) -> SimulationState:
        return env_step(state)

    def mark_tournament_done(_unused: None) -> SimulationState:
        return state._replace(step_done_flag=jnp.bool_(True))

    def check_pairing_and_step(_unused: None) -> SimulationState:
        return lax.cond(can_any_coop_pair(state), run_env_step, mark_tournament_done, None)

    def no_op(_unused: None) -> SimulationState:
        return state

    return lax.cond(state.step_done_flag, no_op, check_pairing_and_step, None)


def tournament_loop_body(_: int, state: SimulationState) -> SimulationState:
    # Tournament loop body.
    state_with_fresh_step_flag = state._replace(
        tournament_step_index=jnp.int32(0),
        step_done_flag=jnp.bool_(False),
    )
    state_after_steps = lax.fori_loop(
        0, MAX_STEPS_PER_TOURNAMENT, tournament_step_body, state_with_fresh_step_flag
    )
    state_after_ability_update = update_ability_beliefs(state_after_steps)
    rng_key, close_rng_subkey = jax.random.split(state_after_ability_update.rng_key)
    state_after_close = close_tournament(
        state_after_ability_update._replace(rng_key=rng_key), close_rng_subkey
    )
    return update_king_beliefs(state_after_close)


@jax.jit
def run_one_tournament(state: SimulationState) -> SimulationState:
    # Run one tournament.
    return tournament_loop_body(0, state)


def run_simulation(initial_state: SimulationState) -> Tuple[SimulationState, jnp.ndarray]:
    # Run the simulation.
    observation_history_slices = []
    state = initial_state
    for _ in range(NUM_TOURNAMENTS):
        state = run_one_tournament(state)
        observation_history_slices.append(state.last_tournament_observed_outcomes.sum(axis=-1))
    observation_history = jnp.stack(observation_history_slices, axis=-1).astype(jnp.int8)
    return state, observation_history


def initialize_pec_king_environment(
    seed: int = 0,
) -> Tuple[SimulationState, list[Cage], list[Monitor]]:
    # Initialize the environment.
    base_rng_key = jax.random.PRNGKey(seed)
    ability_rng_key, coop_rng_key, simulation_rng_key = jax.random.split(base_rng_key, 3)

    agent_abilities = jax.random.categorical(
        ability_rng_key,
        jnp.tile(ABILITY_PRIOR_LOG_PROBS[None, None, :], (NUM_AGENTS, NUM_COOPS, 1)),
        axis=-1,
    ).astype(jnp.int32)
    initial_coop_assignments = jax.random.randint(coop_rng_key, (NUM_AGENTS,), 0, NUM_COOPS)
    initial_zone_array = jnp.zeros((NUM_AGENTS, NUM_COOPS, 3), dtype=jnp.int8).at[
        :, :, ZONE_BATTLE
    ].set(jax.nn.one_hot(initial_coop_assignments, NUM_COOPS, dtype=jnp.int8))

    initial_ability_beliefs = jnp.tile(
        jax.nn.softmax(ABILITY_PRIOR_LOG_PROBS)[None, None, None, :],
        (NUM_AGENTS, NUM_AGENTS, NUM_COOPS, 1),
    )

    self_ability_one_hot = jax.nn.one_hot(agent_abilities, NUM_ABILITY_LEVELS)
    is_self_pair = jnp.eye(NUM_AGENTS, dtype=jnp.bool_)
    initial_ability_beliefs = jnp.where(
        is_self_pair[:, :, None, None],
        self_ability_one_hot[:, None, :, :],
        initial_ability_beliefs,
    )

    initial_state = SimulationState(
        zone_array=initial_zone_array,
        transit_destination_coop=jnp.full((NUM_AGENTS,), -1, jnp.int32),
        action_array=jnp.zeros((NUM_AGENTS, 1), jnp.int32),
        agent_abilities=agent_abilities,
        tournament_losses_per_coop=jnp.zeros((NUM_AGENTS, NUM_COOPS), jnp.int32),
        tournament_fought_matrix=jnp.zeros((NUM_AGENTS, NUM_AGENTS, NUM_COOPS), jnp.int8),
        tournament_outcomes=jnp.zeros((NUM_AGENTS, NUM_AGENTS, NUM_COOPS), jnp.int8),
        tournament_step_obs_coverage=jnp.zeros(
            (MAX_STEPS_PER_TOURNAMENT, NUM_AGENTS, NUM_COOPS), jnp.int8
        ),
        tournament_step_outcomes=jnp.zeros(
            (MAX_STEPS_PER_TOURNAMENT, NUM_AGENTS, NUM_AGENTS, NUM_COOPS), jnp.int8
        ),
        battle_outcome=jnp.zeros((NUM_AGENTS, NUM_AGENTS, NUM_COOPS), jnp.int8),
        battle_history=jnp.zeros((NUM_AGENTS, NUM_AGENTS, NUM_COOPS, NUM_TOURNAMENTS), jnp.int8),
        obs_mask=jnp.zeros((NUM_AGENTS, NUM_COOPS, NUM_TOURNAMENTS), jnp.int8),
        last_tournament_observed_outcomes=jnp.zeros(
            (NUM_AGENTS, NUM_AGENTS, NUM_AGENTS, NUM_COOPS), jnp.int8
        ),
        ability_beliefs=initial_ability_beliefs,
        king_beliefs=jnp.full((NUM_AGENTS, NUM_COOPS, 2), 0.5, jnp.float32),
        crown_history=jnp.zeros((NUM_AGENTS, NUM_COOPS, NUM_TOURNAMENTS), jnp.int8),
        tournament_index=jnp.int32(0),
        tournament_step_index=jnp.int32(0),
        step_done_flag=jnp.bool_(False),
        rng_key=simulation_rng_key,
    )

    cages = [
        Cage(cage_index=coop_index * (NUM_AGENTS // 2) + cage_slot, coop_index=coop_index)
        for coop_index in range(NUM_COOPS)
        for cage_slot in range(NUM_AGENTS // 2)
    ]
    monitors = [
        Monitor(
            monitor_index=coop_index * NUM_COOPS + viewed_coop_index,
            coop_index=coop_index,
            viewed_coop_index=viewed_coop_index,
        )
        for coop_index in range(NUM_COOPS)
        for viewed_coop_index in range(NUM_COOPS)
    ]
    return initial_state, cages, monitors


def build_agents(final_state: SimulationState, observation_history: jnp.ndarray) -> list[Agent]:
    # Build agent objects.
    ability_beliefs_np = np.array(final_state.ability_beliefs)
    king_beliefs_np = np.array(final_state.king_beliefs)
    agent_abilities_np = np.array(final_state.agent_abilities)
    observation_history_np = np.array(observation_history)
    last_tournament_observed_outcomes_np = np.array(final_state.last_tournament_observed_outcomes)
    zone_array_np = np.array(final_state.zone_array)
    transit_dest_np = np.array(final_state.transit_destination_coop)

    agents = []
    for agent_idx in range(NUM_AGENTS):
        observation_memory = observation_history_np[agent_idx]
        zone_slice = zone_array_np[agent_idx]
        coop_index = int(zone_slice.sum(axis=1).argmax())
        zone_index = int(zone_slice[coop_index].argmax())
        last_observation = last_tournament_observed_outcomes_np[agent_idx]

        agents.append(
            Agent(
                agent_index=agent_idx,
                observation_memory=observation_memory,
                ability_belief=ability_beliefs_np[agent_idx],
                my_ability=agent_abilities_np[agent_idx],
                my_location=Location(
                    coop_index=coop_index,
                    zone_index=zone_index,
                    transit_target_coop=int(transit_dest_np[agent_idx]),
                ),
                last_observation=last_observation,
                king_belief=king_beliefs_np,
            )
        )

    return agents


def to_jax_arrays(state: SimulationState) -> dict[str, jnp.ndarray]:
    # Build the JAX array dict.
    return {
        "zone_array": state.zone_array,
        "action_array": state.action_array,
        "battle_outcome": state.battle_outcome,
        "battle_history": state.battle_history,
    }


def compute_convergence_metrics(final_state: SimulationState) -> dict[str, float | int]:
    # Compute convergence metrics.
    abilities_np = np.array(final_state.agent_abilities)
    ability_beliefs_np = np.array(final_state.ability_beliefs)
    king_beliefs_np = np.array(final_state.king_beliefs)
    crown_history_np = np.array(final_state.crown_history)
    obs_mask_np = np.array(final_state.obs_mask)
    completed_tournaments = int(final_state.tournament_index)

    observed_per_coop: npt.NDArray[np.bool_] = np.asarray(obs_mask_np.any(axis=-1), dtype=np.bool_)
    correct_count = 0
    total_count = 0

    for coop_index in range(NUM_COOPS):
        observer_mask: npt.NDArray[np.bool_] = observed_per_coop[:, coop_index]
        if not observer_mask.any():
            continue

        pooled_ability_beliefs = ability_beliefs_np[observer_mask, :, coop_index, :].mean(axis=0)
        ability_map_estimates = pooled_ability_beliefs.argmax(axis=-1)
        correct_count += int((ability_map_estimates == abilities_np[:, coop_index]).sum())
        total_count += NUM_AGENTS

    ability_map_accuracy = float(correct_count / max(total_count, 1))

    true_win_prob_per_coop = (
        (abilities_np[:, np.newaxis, :] > abilities_np[np.newaxis, :, :]).astype(np.float32)
        + 0.5
        * (abilities_np[:, np.newaxis, :] == abilities_np[np.newaxis, :, :]).astype(np.float32)
    )
    win_beliefs_np = np.array(compute_win_beliefs(final_state.ability_beliefs))
    pooled_win_beliefs = np.full((NUM_AGENTS, NUM_AGENTS, NUM_COOPS), 0.5, dtype=np.float32)
    for coop_index in range(NUM_COOPS):
        observer_mask: npt.NDArray[np.bool_] = observed_per_coop[:, coop_index]
        if observer_mask.any():
            pooled_win_beliefs[:, :, coop_index] = win_beliefs_np[
                observer_mask, :, :, coop_index
            ].mean(axis=0)
    off_diagonal_mask = ~np.eye(NUM_AGENTS, dtype=np.bool_)
    win_belief_mae = float(
        np.abs(pooled_win_beliefs - true_win_prob_per_coop)[off_diagonal_mask, :].mean()
    )

    empirical_king_frequency = crown_history_np.sum(axis=-1) / max(completed_tournaments, 1)
    predicted_king_probability = king_beliefs_np[:, :, 1]
    king_belief_mae = float(
        np.abs(predicted_king_probability - empirical_king_frequency).mean()
    )

    return {
        "ability_map_accuracy": ability_map_accuracy,
        "win_belief_mae": win_belief_mae,
        "king_belief_mae": king_belief_mae,
        "completed_tournaments": completed_tournaments,
    }


def main() -> None:
    # Run the program.
    prior_probabilities = np.array(jax.nn.softmax(ABILITY_PRIOR_LOG_PROBS))
    print("\nAbility prior:")
    for ability_level, prior_probability in zip(
        range(1, NUM_ABILITY_LEVELS + 1), prior_probabilities, strict=True
    ):
        print(f"  level {ability_level}: {prior_probability:.4f}")

    initial_state, cages, monitors = initialize_pec_king_environment(seed=0)
    print("\nEnvironment:")
    print(f"  agents: {NUM_AGENTS}")
    print(f"  coops: {NUM_COOPS}")
    print(f"  cages: {len(cages)} total, {NUM_AGENTS // 2} per coop")
    print(f"  monitors: {len(monitors)} total, {NUM_COOPS} per coop")
    print(f"  loss limit k: {MAX_LOSSES_BEFORE_ELIMINATION}")

    simulation_start_time = time.time()
    final_state, observation_history = run_simulation(initial_state)
    final_state.tournament_index.block_until_ready()
    simulation_elapsed_seconds = time.time() - simulation_start_time

    metrics = compute_convergence_metrics(final_state)
    print(f"Done in {simulation_elapsed_seconds:.1f}s")
    print(f"Completed tournaments: {int(final_state.tournament_index)}")
    print(f"Ability MAP accuracy: {metrics['ability_map_accuracy']:.4f}")
    print(f"Win belief MAE: {metrics['win_belief_mae']:.4f}")
    print(f"King posterior MAE: {metrics['king_belief_mae']:.4f}")

    agents = build_agents(final_state, observation_history)
    jax_array_dict = to_jax_arrays(final_state)
    print("\nJAX tensors:")
    for name, array in jax_array_dict.items():
        print(f"  {name}: shape={tuple(array.shape)} dtype={array.dtype}")

    abilities_np = np.array(final_state.agent_abilities)
    win_beliefs_np = np.array(compute_win_beliefs(final_state.ability_beliefs))
    battle_history_np = np.array(final_state.battle_history)
    observed_per_coop: npt.NDArray[np.bool_] = np.asarray(
        np.array(final_state.obs_mask).any(axis=-1),
        dtype=np.bool_,
    )
    sample_pairs = [(0, 1, 0), (0, 7, 1), (3, 12, 2), (10, 25, 3), (5, 41, 0)]
    print("\nSample pairwise win beliefs:")
    for agent_i, agent_j, coop_index in sample_pairs:
        ability_i = int(abilities_np[agent_i, coop_index])
        ability_j = int(abilities_np[agent_j, coop_index])
        true_win_prob = 1.0 if ability_i > ability_j else (0.5 if ability_i == ability_j else 0.0)
        observer_mask = observed_per_coop[:, coop_index]
        if observer_mask.any():
            predicted_win_belief = float(
                win_beliefs_np[observer_mask, agent_i, agent_j, coop_index].mean()
            )
        else:
            predicted_win_belief = 0.5
        wins_in_coop = np.maximum(battle_history_np[agent_i, agent_j, coop_index, :], 0).sum()
        battles_in_coop = np.abs(battle_history_np[agent_i, agent_j, coop_index, :]).sum()
        empirical_rate = float(wins_in_coop / battles_in_coop) if battles_in_coop > 0 else 0.5
        print(
            "  "
            f"agents ({agent_i}, {agent_j}) coop {coop_index}: "
            f"abilities=({ability_i + 1}, {ability_j + 1}) "
            f"belief={predicted_win_belief:.4f} "
            f"true={true_win_prob:.1f} empirical={empirical_rate:.4f}"
        )

    crown_history_np = np.array(final_state.crown_history)
    crown_counts_np = crown_history_np.sum(axis=-1)
    completed_tournaments = int(final_state.tournament_index)
    empirical_king_frequency = crown_counts_np / max(completed_tournaments, 1)
    king_beliefs_np = np.array(final_state.king_beliefs)
    predicted_king_probability = king_beliefs_np[:, :, 1]
    all_ability_beliefs_np = np.array(final_state.ability_beliefs)
    pooled_ability_beliefs = all_ability_beliefs_np.mean(axis=0)
    for coop_index in range(NUM_COOPS):
        observer_mask: npt.NDArray[np.bool_] = observed_per_coop[:, coop_index]
        if observer_mask.any():
            pooled_ability_beliefs[:, coop_index, :] = all_ability_beliefs_np[
                observer_mask, :, coop_index, :
            ].mean(axis=0)

    top_five_agents = np.argsort(-crown_counts_np.sum(axis=1))[:5]
    print("\nKing beliefs:")
    for agent_idx in top_five_agents:
        for coop_index in range(NUM_COOPS):
            ability_map_estimate = int(pooled_ability_beliefs[agent_idx, coop_index].argmax()) + 1
            print(
                "  "
                f"agent {agent_idx} coop {coop_index}: "
                f"true_ability={int(abilities_np[agent_idx, coop_index]) + 1} "
                f"map={ability_map_estimate} "
                f"empirical={float(empirical_king_frequency[agent_idx, coop_index]):.4f} "
                f"posterior={float(predicted_king_probability[agent_idx, coop_index]):.4f}"
            )

    sample_agent = agents[int(top_five_agents[0])]
    sample_policy_scores = (
        ABILITY_SCORE_POLICY_WEIGHT * sample_agent.my_ability.astype(np.float32) / NUM_ABILITY_LEVELS
        + KING_BELIEF_POLICY_WEIGHT * sample_agent.king_belief[sample_agent.agent_index, :, 1]
    )
    sample_action, sample_action_coop = sample_agent.action_policy()
    zone_name = (
        "BATTLE"
        if sample_agent.my_location.zone_index == ZONE_BATTLE
        else "SPECTATOR"
        if sample_agent.my_location.zone_index == ZONE_SPECTATOR
        else "TRANSIT"
    )
    print(f"\nSample agent: {sample_agent.agent_index}")
    print(
        f"  location: coop {sample_agent.my_location.coop_index}, "
        f"zone {sample_agent.my_location.zone_index} ({zone_name})"
    )
    print(f"  abilities: {(sample_agent.my_ability + 1).tolist()}")
    print(f"  policy scores: {np.round(sample_policy_scores, 4).tolist()}")
    print(f"  action_policy: {sample_action} {sample_action_coop}")


if __name__ == "__main__":
    main()
