# Rolling replay recording and analysis protocol

Accepted current algorithm: capacity 50000, append 2048 horizon-1 synthetic
transitions per refit, FIFO overwrite when full, no clear on refit. Refits use
real-only snapshots every2000 real transitions starting at4000. This cleanup
changes neither these settings nor policy updates/sampling/RNG.

Future processes write metrics/synthetic_generations.jsonl with:
- refit ID, generation real_env_steps, generated count;
- capacity, before/after occupied size and insertion position;
- number evicted, retained counts per refit generation;
- age of the oldest retained generation measured in real transitions.

Generation bookkeeping uses only deterministic integer/deque operations. It does
not sample replay, consume RNG or modify stored transitions. Records describe
retained generations, not per-SAC-batch sampled ages. `model/synthetic_state_diagnostics_raw` describes newly generated, pre-normalization
predictions. `model_pool/*` inspects up to 256 occupied physical slots after each
refit, deterministically (not a random or population-wide estimate). It describes
stored, quaternion-normalized predictions. `model_sampled_batch/*` describes the
synthetic subset of the first actual policy batch on each crossed log interval.
It reuses the already sampled batch; it never draws additional samples. Neither
pool nor sampled validity has real next-state ground truth: these are consistency
checks, not prediction accuracy or synthetic utility. Per-batch sampled generation
ages are not logged; FIFO age statistics describe retained data only.

The audit reconstructs FIFO generation counts and checks insertion-position
arithmetic, refit/generated counters and replay size at every transition record.
It checks episode row counts, per-step environment identity (when available),
finished_episodes and return/length/success means against the local W&B ledger.
These checks do not prove the content of every historical replay slot. They verify
recorded bookkeeping, and are not a synthetic-utility or learning-quality test.

At this capacity/generation size the pool fills at the25th refit (52000 real
transitions). A full pool contains24 complete recent batches plus848 records from
the preceding generation. This is intentional rolling retention, not an error.

Existing runs remain unchanged. Runs lacking generation records return
rolling_provenance=UNVERIFIED; missing env_index limits identity checking but does
not prevent recomputing the recorded means. Runtime code changes do not update
already-running Python processes. Do not restart a valid run merely to add these
fields; use them for subsequent processes and preserve the running result.

For later comparisons record each run's git_commit, resolved_config, actual real
transitions and policy_gradient_steps, FIFO vs clear protocol and capacity, and
evaluation episode seeds/termination policy. Match x-axis and episode aggregation;
report different actual update counts explicitly. Compare frozen evaluation
success/return/length together. A higher training return alone does not identify
why two algorithms differ.
