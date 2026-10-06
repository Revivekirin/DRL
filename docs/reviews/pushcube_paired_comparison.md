# PushCube SAC paired evaluation and source selection

Inputs: `outputs/pushcube_reeval_100.2RM5Z1/{step_100000,final}/{episodes.csv,summary.json}`.
Join key: integer episode seed. Each input has exactly 100 unique seeds, 0–99; no duplicates or missing seeds. Both actual evaluation contracts are identical (PushCube-v1/state/panda/pd_ee_delta_pos/normalized_dense, CPU num_envs=1, horizon 50, terminate_on_success, deterministic CUDA learner). The recorded training policy is legacy metadata, not newly verified historical evidence.

Group means below; delta = 500k minus 100k. Return is cumulative and early success stops accumulation.

| Group | Count | Length 100k | Length 500k | Δ length | Return 100k | Return 500k | Δ return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| both_success | 50 | 6.4000 | 8.5000 | 2.1000 | 1.9750 | 2.1650 | 0.1900 |
| only_100k | 46 | 6.4783 | 50.0000 | 43.5217 | 1.9758 | 5.6680 | 3.6922 |
| only_500k | 2 | 50.0000 | 10.5000 | -39.5000 | 7.3819 | 2.6980 | -4.6839 |
| both_failure | 2 | 50.0000 | 50.0000 | 0.0000 | 8.1695 | 5.8658 | -2.3037 |

100k is designated the source candidate, not an established optimum across all possible checkpoints.
Candidate: `20261006T083356_f30a4046/checkpoints/step_100000.pt`; SHA256 `9048950a50e2d95e8a7df09af55b4199cf08eef11e843c9e614a6d97451851fd`.
The 46 lost successes and 2 gained successes describe this paired evaluation set; they do not identify a cause of degradation. Only one training seed was used. Full group seed lists are in `pushcube_paired_results.json`.

## Selection and final evaluation separation
- Selection evidence: historical 20-episode evaluation stream initialized with seed 0, plus the now-inspected explicit episode seeds 0–99. Treat all as development/selection evidence.
- Reserved future final evaluation: explicit episode reset seeds 30000–30099. Do not inspect these results for checkpoint or hyperparameter selection; no final evaluation is run or requested here. This is a prospective reservation, not proof of historical non-use outside the supplied records.
- MBPO smoke development evaluation: three episodes in a reset stream initialized with seed 20000. This is diagnostic only, separate from both selection and reserved final evaluation.
- Training seed remains 0; changing evaluation seeds does not create independent training replicates.
- Source candidate designation does not initialize this MBPO smoke. The smoke initializes both SAC and ensemble from scratch, with empty replay. Future SAC learner initialization would be a separate warm start, not exact training resume.
