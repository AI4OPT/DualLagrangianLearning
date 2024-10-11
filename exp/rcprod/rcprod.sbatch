#!/bin/bash
#SBATCH --job-name=RCProd                   # Job name
#SBATCH --account=<account_name>             # charge account
#SBATCH --mail-type=NONE                    # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=<account_mail>    # Where to send mail
#SBATCH --gres=gpu:V100:1
#SBATCH --nodes=2                           # Use multiple nodes
#SBATCH --ntasks-per-node=1                 # Run a single task per node
#SBATCH --mem=64G                           # Use 64GB of memory (should be ample enough while fitting on smaller nodes)
#SBATCH --time=01:00:00                     # Time limit hrs:min:sec
#SBATCH -o exp/rcprod/slurm/rcprod_%A\_%a.out         # Combined output and error messages file
#SBATCH -q-q<queue_name>

# The index of the array job indicates the number of nodes in the underlying graph
#SBATCH --array=10,50,100,500,1000,20,200

module load julia/1.9.2
cd <path_to_project_root>
srun exp/rcprod/rcprod.sh
