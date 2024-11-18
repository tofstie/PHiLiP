#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=rrg-nadaraja-ac
#SBATCH --job-name=ESROM_TGV_P3_CFL0p0005_20
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8                          ## <-- refer to https://docs.computecanada.ca/wiki/Advanced_MPI_scheduling
#SBATCH --mem=0                                       ## <-- total shared memory; --mem=0 means to reserve all the available memory on each node assigned to the job
#SBATCH --mail-user=tyson.ofstie@mail.mcgill.ca ## <-- for receiving job updates via email
#SBATCH --mail-type=ALL                               ## <-- what kind of updates to receive by email

SLURM_USER="tofstie"                    ## <-- Enter compute canada username here
PARAMETERS_FILE="ESROM_TGV_P3_CFL0p0005_15.prm" ## <-- Enter .prm filename here
PHiLiP_DIMENSIONS="3"                    ## WARNING: must correspond to the DIM in the .prm file
NUM_PROCS="8"                           ## WARNING: must correspond to nodes*(ntasks-per-node) above
RUN_ON_TMPDIR=false                      ## Set as true for fast write speeds, however, output files will only be copied to your job submit directory once mpirun has completed.

PHiLiP_EXECUTABLE="/home/${SLURM_USER}/scratch/PHiLiP_${PHiLiP_DIMENSIONS}D"

## Below are the modules needed to run the executable
module --force purge # not needed?
module load StdEnv/2020 # not needed?
##module load intel/2020.1.217
module load gcc/9.3.0 # not needed?
module load openmpi/4.0.3 # required

if [ ${RUN_ON_TMPDIR} = true ]; then
        cd ${SLURM_TMPDIR};      
fi

mpirun -n ${NUM_PROCS} "${PHiLiP_EXECUTABLE}" -i "${SLURM_SUBMIT_DIR}/${PARAMETERS_FILE}"

if [ ${RUN_ON_TMPDIR} = true ]; then
        # Get output files, exclude subdirectories
        rsync -axvH --no-g --no-p --exclude='*/' ${SLURM_TMPDIR}/* ${SLURM_SUBMIT_DIR};
fi