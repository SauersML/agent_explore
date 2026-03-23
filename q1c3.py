# ─── build_agents ──────────────────────────────────────────────────────────────

def build_agents(final_state):
    """
    Reconstruct one Agent object per chicken from the final SimulationState.
    All JAX arrays are transferred to numpy once at the top to avoid repeated
    device-to-host copies inside the loop.
    """
    battle_history_np  = np.array(final_state.battle_history)          # (M, M, N, T) int8
    obs_mask_np        = np.array(final_state.obs_mask)                 # (M, N, T)    int8
    ability_beliefs_np = np.array(final_state.ability_beliefs)          # (M, M, N, L) float32
    king_beliefs_np    = np.array(final_state.king_beliefs)             # (M, M, N, 2) float32
    agent_abilities_np = np.array(final_state.agent_abilities)          # (M, N)       int32
    zone_array_np      = np.array(final_state.zone_array)               # (M, N, 3)    int8
    transit_dest_np    = np.array(final_state.transit_destination_coop) # (M,)         int32

    completed_tournaments = int(final_state.tournament_index)
    last_tournament_index = max(completed_tournaments - 1, 0)

    agents = []
    for agent_idx in range(NUM_AGENTS):
        # observation_memory[i, j, t] = sum_k battle_history[i,j,k,t] * obs_mask[agent_idx,k,t]
        # Each pair fights at most once per coop per tournament, so the sum over k
        # produces at most one non-zero term — a clean ±1 or 0.
        observation_memory = np.einsum(
            'ijkt,kt->ijt',
            battle_history_np,
            obs_mask_np[agent_idx],
        ).astype(np.int8)  # (M, M, T)

        zone_slice = zone_array_np[agent_idx]                          # (N, 3)
        coop_index = int(zone_slice.sum(axis=1).argmax())
        zone_index = int(zone_slice[coop_index].argmax())

        last_tournament_obs_mask = obs_mask_np[agent_idx, :, last_tournament_index]  # (N,)
        last_observation = (
            battle_history_np[:, :, :, last_tournament_index]
            * last_tournament_obs_mask[np.newaxis, np.newaxis, :]
        ).astype(np.int8)  # (M, M, N)

        agents.append(Agent(
            agent_index=agent_idx,
            observation_memory=observation_memory,
            ability_belief=ability_beliefs_np[agent_idx],   # (M, N, L)
            my_ability=agent_abilities_np[agent_idx],        # (N,)
            my_location=Location(
                coop_index=coop_index,
                zone_index=zone_index,
                transit_target_coop=int(transit_dest_np[agent_idx]),
            ),
            last_observation=last_observation,
            king_belief=king_beliefs_np[agent_idx],          # (M, N, 2)
        ))

    return agents


# ─── to_jax_arrays ─────────────────────────────────────────────────────────────

def to_jax_arrays(state):
    """
    Return the four spec-required environment tensors as a labelled dict.
    battle_outcome is the current tournament's accumulated outcome matrix;
    battle_history is the full cross-tournament record.
    """
    return {
        'zone_array':     state.zone_array,
        'action_array':   state.action_array,
        'battle_outcome': state.tournament_outcomes,
        'battle_history': state.battle_history,
    }


# ─── Convergence metrics ───────────────────────────────────────────────────────

def compute_convergence_metrics(final_state):
    """
    Three metrics measuring how well agents' beliefs converged to ground truth.

    ability_map_accuracy   — fraction of (agent, coop) pairs where the MAP
                             estimate of the pooled ability belief matches true ability.

    win_belief_mae         — mean absolute error between empirical win rates
                             (from battle_history) and true pairwise win probabilities
                             derived from ground-truth abilities.

    king_belief_mae        — mean absolute error between the predicted king
                             probabilities (pooled over observers) and the empirical
                             crown frequency across all completed tournaments.
    """
    abilities_np       = np.array(final_state.agent_abilities)    # (M, N) 0-indexed
    ability_beliefs_np = np.array(final_state.ability_beliefs)    # (M, M, N, L)
    king_beliefs_np    = np.array(final_state.king_beliefs)       # (M, M, N, 2)
    crown_counts_np    = np.array(final_state.crown_counts)       # (M, N)
    battle_history_np  = np.array(final_state.battle_history)     # (M, M, N, T) int8
    completed_tournaments = int(final_state.tournament_index)

    # ── Ability MAP accuracy ──────────────────────────────────────────────────
    pooled_ability_beliefs  = ability_beliefs_np.mean(axis=0)        # (M, N, L)
    ability_map_estimates   = pooled_ability_beliefs.argmax(axis=-1) # (M, N)  0-indexed
    ability_map_accuracy    = float((ability_map_estimates == abilities_np).mean())

    # ── Win belief MAE ────────────────────────────────────────────────────────
    # True win probability: 1.0 if i strictly stronger, 0.5 if tied, 0.0 if weaker
    true_win_prob_per_coop = (
        (abilities_np[:, np.newaxis, :] > abilities_np[np.newaxis, :, :]).astype(np.float32)
        + 0.5 * (abilities_np[:, np.newaxis, :] == abilities_np[np.newaxis, :, :]).astype(np.float32)
    )  # (M, M, N)
    mean_true_win_prob = true_win_prob_per_coop.mean(axis=-1)        # (M, M)

    total_wins_observed    = np.maximum(battle_history_np, 0).sum(axis=(2, 3))   # (M, M)
    total_battles_observed = np.abs(battle_history_np).sum(axis=(2, 3))          # (M, M)
    empirical_win_rate = np.where(
        total_battles_observed > 0,
        total_wins_observed.astype(np.float32) / np.maximum(total_battles_observed, 1),
        0.5,
    )  # (M, M)

    off_diagonal_mask = ~np.eye(NUM_AGENTS, dtype=np.bool_)
    win_belief_mae = float(
        np.abs(empirical_win_rate - mean_true_win_prob)[off_diagonal_mask].mean()
    )

    # ── King belief MAE ───────────────────────────────────────────────────────
    empirical_king_frequency   = crown_counts_np / max(completed_tournaments, 1)  # (M, N)
    predicted_king_probability = king_beliefs_np[:, :, :, 1].mean(axis=0)         # (M, N)
    king_belief_mae = float(
        np.abs(predicted_king_probability - empirical_king_frequency).mean()
    )

    return {
        'ability_map_accuracy': ability_map_accuracy,
        'win_belief_mae':       win_belief_mae,
        'king_belief_mae':      king_belief_mae,
        'completed_tournaments': completed_tournaments,
    }


# ─── Execution ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("PEC-KING ORDER  —  CSCi 5512 Problem 1")
print("=" * 60)

# ── Ability prior ─────────────────────────────────────────────────────────────
prior_probabilities = np.array(jax.nn.softmax(ABILITY_PRIOR_LOG_PROBS))
print("\nTruncated Poisson ability prior  (rate = NUM_COOPS / 3.0 ≈ 1.333):\n")
print(pd.DataFrame({
    'ability_level':    range(1, NUM_ABILITY_LEVELS + 1),
    'prior_probability': np.round(prior_probabilities, 4),
}).to_string(index=False))

# ── Initialize ────────────────────────────────────────────────────────────────
initial_state, cages, monitors = initialize_pec_king_environment(seed=0)

print(f"\nEnvironment initialized:")
print(f"  {NUM_AGENTS} agents  ×  {NUM_COOPS} coops")
print(f"  {len(cages)} cages total  ({NUM_AGENTS // 2} per BattleZone)")
print(f"  {len(monitors)} monitors total  ({NUM_COOPS} per SpectatorZone)")
print(f"  Loss limit k = {MAX_LOSSES_BEFORE_ELIMINATION}")

# ── Run simulation ────────────────────────────────────────────────────────────
# First call to run_simulation triggers JIT compilation; timing includes that cost.
print(f"\nCompiling and running {NUM_TOURNAMENTS} tournaments ...")
simulation_start_time = time.time()
final_state = run_simulation(initial_state)
final_state.tournament_index.block_until_ready()   # ensure device work is done before stopping clock
simulation_elapsed_seconds = time.time() - simulation_start_time

print(f"Done in {simulation_elapsed_seconds:.1f} s  "
      f"(includes one-time JIT compilation)")
print(f"Completed tournaments: {int(final_state.tournament_index)}")

# ── Convergence metrics ───────────────────────────────────────────────────────
metrics = compute_convergence_metrics(final_state)

print(f"\nConvergence metrics after {metrics['completed_tournaments']} tournaments:")
print(f"  Ability MAP accuracy             {metrics['ability_map_accuracy']:.4f}")
print(f"  Win belief MAE                   {metrics['win_belief_mae']:.4f}")
print(f"  King belief MAE                  {metrics['king_belief_mae']:.4f}")

# ── Build agent objects ───────────────────────────────────────────────────────
agents = build_agents(final_state)

# ── Spec-required array shapes ────────────────────────────────────────────────
jax_array_dict = to_jax_arrays(final_state)

state_shapes_table = pd.DataFrame([
    {'tensor': name, 'shape': str(tuple(array.shape)), 'dtype': str(array.dtype)}
    for name, array in jax_array_dict.items()
])
print("\nSpec-required JAX environment tensors:\n")
print(state_shapes_table.to_string(index=False))

# ── Sample pairwise win beliefs ───────────────────────────────────────────────
abilities_np         = np.array(final_state.agent_abilities)   # (M, N) 0-indexed
battle_history_np    = np.array(final_state.battle_history)
total_wins_observed  = np.maximum(battle_history_np, 0).sum(axis=(2, 3))
total_battles_np     = np.abs(battle_history_np).sum(axis=(2, 3))
empirical_win_rate   = np.where(
    total_battles_np > 0,
    total_wins_observed.astype(np.float32) / np.maximum(total_battles_np, 1),
    0.5,
)

sample_pairs = [(0, 1, 0), (0, 7, 1), (3, 12, 2), (10, 25, 3), (5, 41, 0)]
win_table_rows = []
for agent_i, agent_j, coop_index in sample_pairs:
    ability_i = int(abilities_np[agent_i, coop_index])
    ability_j = int(abilities_np[agent_j, coop_index])
    true_win_prob = 1.0 if ability_i > ability_j else (0.5 if ability_i == ability_j else 0.0)
    win_table_rows.append({
        'agent_i':              agent_i,
        'agent_j':              agent_j,
        'coop':                 coop_index,
        'true_ability_i':       ability_i + 1,   # display as 1-indexed
        'true_ability_j':       ability_j + 1,
        'true_win_prob':        round(true_win_prob, 1),
        'empirical_win_rate':   round(float(empirical_win_rate[agent_i, agent_j]), 4),
    })

print("\nSample pairwise win beliefs (empirical rate from battle history):\n")
print(pd.DataFrame(win_table_rows).to_string(index=False))

# ── Sample king beliefs for the most-crowned agents ───────────────────────────
crown_counts_np           = np.array(final_state.crown_counts)          # (M, N)
completed_tournaments     = int(final_state.tournament_index)
empirical_king_frequency  = crown_counts_np / max(completed_tournaments, 1)
king_beliefs_np           = np.array(final_state.king_beliefs)
predicted_king_probability = king_beliefs_np[:, :, :, 1].mean(axis=0)   # (M, N)
pooled_ability_beliefs    = np.array(final_state.ability_beliefs).mean(axis=0)

top_five_agents = np.argsort(-crown_counts_np.sum(axis=1))[:5]
king_table_rows = []
for agent_idx in top_five_agents:
    for coop_index in range(NUM_COOPS):
        ability_map_estimate = int(pooled_ability_beliefs[agent_idx, coop_index].argmax()) + 1
        king_table_rows.append({
            'agent':                agent_idx,
            'coop':                 coop_index,
            'true_ability':         int(abilities_np[agent_idx, coop_index]) + 1,
            'ability_MAP_estimate': ability_map_estimate,
            'empirical_P_king':     round(float(empirical_king_frequency[agent_idx, coop_index]), 4),
            'predicted_P_king':     round(float(predicted_king_probability[agent_idx, coop_index]), 4),
        })

print("\nKing beliefs for the 5 most-crowned agents:\n")
print(pd.DataFrame(king_table_rows).to_string(index=False))

# ── Sample agent policy ───────────────────────────────────────────────────────
sample_agent = agents[top_five_agents[0]]
policy_action, policy_target = sample_agent.action_policy()

print(f"\nSample agent (index {sample_agent.agent_index}):")
print(f"  Location       coop {sample_agent.my_location.coop_index}, "
      f"zone {sample_agent.my_location.zone_index} "
      f"({'BATTLE' if sample_agent.my_location.zone_index == ZONE_BATTLE else 'SPECTATOR' if sample_agent.my_location.zone_index == ZONE_SPECTATOR else 'TRANSIT'})")
print(f"  Own abilities  {(sample_agent.my_ability + 1).tolist()}  (1-indexed, one per coop)")
print(f"  Policy output  ('{policy_action}', coop={policy_target})")
