#!/bin/bash
#SBATCH --job-name=no_attn         # Name of the job
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Each node runs 1 task that manages all GPUs
#SBATCH --gpus-per-task=2          # Number of GPUs to allocate per task
#SBATCH --cpus-per-task=2          # Must match >= GPUs on the task
#SBATCH --mem=48G                  # Total memory for job
#SBATCH --time=15:59:00            # Max time limit
#SBATCH --partition=pli            # Partition name
#SBATCH --account=hazan_intern     # Account to charge resources

#SBATCH --error=tensorized_stu_no_attn_%j.err
#SBATCH --output=tensorized_stu_no_attn_%j.out

# Logging
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Error handling
set -e
trap 'log_info "Error on line $LINENO"; exit 1' ERR

# Activate your virtual environment accordingly
module purge
module load anaconda3/2024.6
module load cudatoolkit/12.3
conda activate hazan-lab

# Get the first node (master node) from the SLURM_JOB_NODELIST
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

# Get the IP address of the master node
MASTER_NODE_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname --ip-address)

# Find an available port
RDZV_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# Log start of training run
log_info "Starting training run..."

# Run the script using torchrun
srun torchrun \
--nnodes 1 \
--nproc_per_node 2 \
--rdzv_endpoint $MASTER_NODE_ADDR:$RDZV_PORT \
--max-restarts 16 \
train.py

##################### HELPFUL SLURM COMMANDS #####################

# Disk usage and quota info; request additional space via link.
# `checkquota`

# Operating system details.
# `cat /etc/os-release`

# CPU specifications on the current node.
# `lscpu`

# Compute node information (standard and easy-to-read formats).
# `snodes`
# `shownodes`

# Cluster nodes usage overview; check status (idle, down, busy).
# `sinfo`
    # GPU usage specifics.
    # `sinfo -p gpu`

# Quality of Service insights: job partitions and limits.
# `qos`

# Current processor activity; exit with 'q'.
# `top`
# `htop`

# Overview of group-assigned cluster shares.
# `sshare`

# Job priority mechanics: factors and weights.
# `sprio -w`

# Performance of a completed job by job ID.
# `seff <jobid>`

# Your historical job records.
# `shistory`

# Detailed job statistics (memory, CPU, GPU).
# `jobstats <jobid>`

# Additional commands from your list for GPU details:

# Details about GPUs on the cluster.
# `snodes`

# Number of available GPUs.
# `shownodes -p gpu,mig`

# GPU utilization, refreshed every 10 min.
# `gpudash`
    # Specific to your jobs.
    # `gpudash -u $USER`

 # Real-time GPU status on active jobs.
# `nvidia-smi` OR `watch nvidia-smi`

 # Your queue status.
# `squeue -u $USER`
