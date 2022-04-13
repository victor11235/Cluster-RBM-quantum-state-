#!/bin/bash
#SBATCH --account=def-coish
#SBATCH --time=08:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out
#SBATCH --mem-per-cpu=1024M
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=1106194118v@gmail.com

python excited_hist.py