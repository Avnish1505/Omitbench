# Superseded

`pilot.py` (v1) and `pilot2.py` (v2) are kept for provenance, not use.

- **v1** injected omissions by deleting hunks. Vacuous: the omitted code was
  literally absent from the diff, so `grep` scored F1 1.00.
- **v2** introduced the ABSENT/UNWIRED/STUB mutation classes (kept in
  `omitbench/mutate.py`) but had a 75% positive base rate, at which
  `flag-everything` scored F1 0.82 and beat every real detector. It also
  matched bare symbol names and required internal call sites for all mutations.

Live code: `omitbench/detectors.py` and `omitbench/experiment.py`.
