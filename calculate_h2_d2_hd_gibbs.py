#!/usr/bin/env python3
"""Standard-state Gibbs free energy for H2 + D2 -> 2 HD.

The electronic potential is calculated once with B3LYP/6-31G(d); within the
Born--Oppenheimer approximation, it is identical for H2, D2, and HD.  The
isotope-specific Gibbs energies are then obtained from the common optimized
geometry and Cartesian Hessian using the corresponding nuclear masses.

Results are reported at 298.15 K and 1 atm, including the rotational symmetry
number (2 for H2/D2 and 1 for HD).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyscf import dft, gto
from pyscf.data import nist
from pyscf.hessian import thermo
from scipy.optimize import minimize_scalar

TEMPERATURE_K = 298.15
PRESSURE_PA = 101_325.0  # 1 atm
BASIS = "6-31G(d)"
FUNCTIONAL = "B3LYP"
H_MASS_U = 1.00782503223
D_MASS_U = 2.01410177812
HARTREE_TO_KJMOL = nist.HARTREE2J * nist.AVOGADRO / 1000.0


def make_molecule(distance_angstrom: float) -> gto.Mole:
    """Make the electronic H--H molecule used for all isotopologues."""
    return gto.M(
        atom=f"H 0 0 {-distance_angstrom / 2:.12f}; H 0 0 {distance_angstrom / 2:.12f}",
        basis=BASIS,
        charge=0,
        spin=0,
        unit="Angstrom",
        verbose=0,
    )


def electronic_energy(distance_angstrom: float) -> float:
    mol = make_molecule(distance_angstrom)
    mf = dft.RKS(mol)
    mf.xc = FUNCTIONAL
    mf.grids.level = 4
    return float(mf.kernel())


def optimize_bond() -> tuple[float, float, gto.Mole, dft.RKS]:
    """Optimize the isotope-independent Born--Oppenheimer H--H distance."""
    optimum = minimize_scalar(
        electronic_energy,
        bounds=(0.4, 1.2),
        method="bounded",
        options={"xatol": 1.0e-7},
    )
    if not optimum.success:
        raise RuntimeError(f"Bond optimization failed: {optimum.message}")

    mol = make_molecule(float(optimum.x))
    mf = dft.RKS(mol)
    mf.xc = FUNCTIONAL
    mf.grids.level = 4
    energy = float(mf.kernel())
    return float(optimum.x), energy, mol, mf


def gibbs_free_energy(
    mol: gto.Mole,
    electronic_energy_hartree: float,
    hessian: np.ndarray,
    masses_u: np.ndarray,
    symmetry_number: int,
) -> dict[str, float]:
    """Rigid-rotor/harmonic-oscillator thermochemistry for specified masses."""
    vibrations = thermo.harmonic_analysis(mol, hessian, mass=masses_u)
    frequencies_au = np.asarray(vibrations["freq_au"]).real
    if np.any(frequencies_au <= 0.0):
        raise RuntimeError(f"Non-positive vibrational frequency: {frequencies_au}")

    coordinates = mol.atom_coords()
    center_of_mass = np.einsum("z,zx->x", masses_u, coordinates) / masses_u.sum()
    centered = coordinates - center_of_mass
    rotational_constants_ghz = thermo.rotation_const(masses_u, centered, "GHz")
    rotational_constant_hz = float(rotational_constants_ghz[1] * 1.0e9)

    k_b = nist.BOLTZMANN
    h = nist.PLANCK
    gas_constant_hartree = k_b / nist.HARTREE2J
    total_mass_kg = masses_u.sum() * nist.ATOMIC_MASS

    q_trans = (
        (2.0 * np.pi * total_mass_kg * k_b * TEMPERATURE_K / h**2) ** 1.5
        * k_b
        * TEMPERATURE_K
        / PRESSURE_PA
    )
    s_trans = gas_constant_hartree * (2.5 + np.log(q_trans))
    h_trans = 2.5 * gas_constant_hartree * TEMPERATURE_K

    q_rot = k_b * TEMPERATURE_K / (symmetry_number * h * rotational_constant_hz)
    s_rot = gas_constant_hartree * (1.0 + np.log(q_rot))
    h_rot = gas_constant_hartree * TEMPERATURE_K

    au_to_hz = (
        nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2)
    ) ** 0.5 / (2.0 * np.pi)
    vibrational_temperatures = frequencies_au * au_to_hz * h / k_b
    reduced_temperatures = vibrational_temperatures / TEMPERATURE_K
    boltzmann = np.exp(-reduced_temperatures)
    zpe = 0.5 * gas_constant_hartree * vibrational_temperatures.sum()
    s_vib = gas_constant_hartree * (
        reduced_temperatures * boltzmann / (1.0 - boltzmann)
        - np.log(1.0 - boltzmann)
    ).sum()
    h_vib = zpe + gas_constant_hartree * TEMPERATURE_K * (
        reduced_temperatures * boltzmann / (1.0 - boltzmann)
    ).sum()

    entropy = s_trans + s_rot + s_vib
    enthalpy = electronic_energy_hartree + h_trans + h_rot + h_vib
    gibbs = enthalpy - TEMPERATURE_K * entropy
    return {
        "gibbs_hartree": float(gibbs),
        "gibbs_kj_mol": float(gibbs * HARTREE_TO_KJMOL),
        "frequency_cm-1": float(np.asarray(vibrations["freq_wavenumber"]).real[0]),
        "entropy_hartree_per_k": float(entropy),
    }


def main() -> None:
    bond_length, electronic_energy_hartree, mol, mf = optimize_bond()
    hessian = mf.Hessian().kernel()

    species = {
        "H2": (np.array([H_MASS_U, H_MASS_U]), 2),
        "D2": (np.array([D_MASS_U, D_MASS_U]), 2),
        "HD": (np.array([H_MASS_U, D_MASS_U]), 1),
    }
    results = {
        name: gibbs_free_energy(mol, electronic_energy_hartree, hessian, masses, sigma)
        for name, (masses, sigma) in species.items()
    }
    delta_g_hartree = 2.0 * results["HD"]["gibbs_hartree"] - results["H2"]["gibbs_hartree"] - results["D2"]["gibbs_hartree"]
    delta_g_kj_mol = delta_g_hartree * HARTREE_TO_KJMOL

    report = {
        "method": f"{FUNCTIONAL}/{BASIS}",
        "temperature_K": TEMPERATURE_K,
        "pressure_Pa": PRESSURE_PA,
        "optimized_HH_distance_angstrom": bond_length,
        "electronic_energy_hartree": electronic_energy_hartree,
        "species": results,
        "reaction": "H2 + D2 -> 2 HD",
        "delta_g_hartree": delta_g_hartree,
        "delta_g_kj_mol": delta_g_kj_mol,
    }
    Path("h2_d2_hd_gibbs_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
