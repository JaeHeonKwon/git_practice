#!/bin/bash
#SBATCH --job-name=h2-d2-hd-gibbs
#SBATCH --nodelist=node7
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=h2_d2_hd_gibbs-%j.out
#SBATCH --error=h2_d2_hd_gibbs-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
MAESTRO_ALLOW_RAW=1 maestro-python calculate_h2_d2_hd_gibbs.py
