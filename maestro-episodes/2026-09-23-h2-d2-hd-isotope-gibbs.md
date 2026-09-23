---
task: unsupported
engine: none
error_class: support_gap
failure_class: setup_error
outcome: support_gap
method: B3LYP
basis: 6-31G(d)
mode: slurm
---
Symptom: The requested reaction Gibbs free energy for H2 + D2 -> 2 HD requires isotope-specific thermochemistry for H2, D2, and HD.

Attempts: Queried the live MAESTRO catalog for free energy, thermochemistry, Gibbs, reaction, and isotope capabilities. ThermoTask produces gibbs_free_energy for a molecule; IsotopeShiftTask produces isotope_frequencies from a Hessian after mass substitution.

Result: No supported MAESTRO task produces isotope-specific Gibbs free energies or the isotope-exchange reaction free energy. Direct engine input is required for this unsupported calculation.

Context: User requested a Python workflow at B3LYP/6-31G(d), submitted through Slurm to a free compute node rather than the login node. No calculation was run.
