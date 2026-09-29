# Execution policy

- Training runs belong on the user's remote server. Provide commands instead of
  executing training locally, unless the user explicitly changes this instruction.
- This includes smoke training and tests that invoke training or learner updates.
  Do not run the entire pytest suite locally: it contains these tests.
- Local static checks and inspections are allowed. State clearly which checks
  were performed and which commands are provided for remote execution.
