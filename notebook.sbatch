#!/bin/bash
#SBATCH --job-name=notebook_%A                                           
#SBATCH --output=../logs/notebook_%A.out                      
#SBATCH --error=../logs/notebook_%A.err                            
#SBATCH --account=theory
#SBATCH --ntasks-per-node=4
#SBATCH --mem 60gb
#SBATCH --time=11:59:00

cd $HOME/assignment_errors/
XDG_RUNTIME_DIR=""
jupyter-notebook --no-browser --ip=$(hostname) --port=9999
