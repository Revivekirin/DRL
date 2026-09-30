# Execution policy

- Training runs belong on the user's remote server. Provide commands instead of
  executing training locally, unless the user explicitly changes this instruction.
- This includes smoke training and tests that invoke training or learner updates.
  Do not run the entire pytest suite locally: it contains these tests.
- Local static checks and inspections are allowed. State clearly which checks
  were performed and which commands are provided for remote execution.

# Development scope

- Continue development with seed 0. Do not launch or recommend additional
  million-transition or multi-seed training runs unless explicitly requested.
