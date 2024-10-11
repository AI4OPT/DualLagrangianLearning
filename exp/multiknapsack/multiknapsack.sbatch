#!/bin/bash
#SBATCH --job-name=MDK                      # Job name
#SBATCH --account=<account_name>             # charge account
#SBATCH --mail-type=NONE                    # Mail events (NONE, BEGIN, END, FAIL, ARRAY_TASKS, ALL)
#SBATCH --mail-user=<user_mail>    # Where to send mail
#SBATCH --gres=gpu:V100:1
#SBATCH --nodes=1                           # Use multiple nodes
#SBATCH --ntasks-per-node=1                 # Run a single task per node
# #SBATCH --cpus-per-task=24                  # Give 24 CPUs to each task
#SBATCH --mem=168G                          # Use all memory on that node
#SBATCH --time=04:00:00                     # Time limit hrs:min:sec
#SBATCH -o exp/multiknapsack/slurm/MDK_%A\_%a.out         # Combined output and error messages file
#SBATCH -q<queue_name>

# The index of the array job indicates the number of nodes in the underlying graph
#SBATCH --array=5,10,15,20,25,30

cd <path_to_project_root>
module load julia/1.9.2  # make sure we use the right julia version

echo "Running with 100 items"
julia --project=. -t12 exp/multiknapsack/run.jl ${SLURM_ARRAY_TASK_ID} 100 42 false data/multiknapsack/ > exp/multiknapsack/logs/MDK_m${SLURM_ARRAY_TASK_ID}_n100_s42_DLL.log 2>&1
julia --project=. -t12 exp/multiknapsack/run.jl ${SLURM_ARRAY_TASK_ID} 100 42 true  data/multiknapsack/ > exp/multiknapsack/logs/MDK_m${SLURM_ARRAY_TASK_ID}_n100_s42_DC3.log 2>&1

echo "Running with 200 items"
julia --project=. -t12 exp/multiknapsack/run.jl ${SLURM_ARRAY_TASK_ID} 200 42 false data/multiknapsack/ > exp/multiknapsack/logs/MDK_m${SLURM_ARRAY_TASK_ID}_n200_s42_DLL.log 2>&1
julia --project=. -t12 exp/multiknapsack/run.jl ${SLURM_ARRAY_TASK_ID} 200 42 true  data/multiknapsack/ > exp/multiknapsack/logs/MDK_m${SLURM_ARRAY_TASK_ID}_n200_s42_DC3.log 2>&1

echo "Running with 500 items"
julia --project=. -t12 exp/multiknapsack/run.jl ${SLURM_ARRAY_TASK_ID} 500 42 false data/multiknapsack/ > exp/multiknapsack/logs/MDK_m${SLURM_ARRAY_TASK_ID}_n500_s42_DLL.log 2>&1
julia --project=. -t12 exp/multiknapsack/run.jl ${SLURM_ARRAY_TASK_ID} 500 42 true  data/multiknapsack/ > exp/multiknapsack/logs/MDK_m${SLURM_ARRAY_TASK_ID}_n500_s42_DC3.log 2>&1
