# Validation-output status

Any seed-42 files generated before the commitment-block telemetry correction
must not be used as paper evidence. The earlier raw count was inflated by
duplicated log entries created during parallel state merges. Rerun the
seed-42 validation with the current runner; it reports unique temporary block
reasons, affected applicants, and terminal blocks separately.
