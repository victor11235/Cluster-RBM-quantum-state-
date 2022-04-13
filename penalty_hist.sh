#!/bin/bash
#SBATCH --account=def-coish
#SBATCH --time=05:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out
#SBATCH --mem-per-cpu=2048M
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=1106194118v@gmail.com

python penalty_hist_ex3.py