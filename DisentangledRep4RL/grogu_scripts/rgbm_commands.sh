sbatch -J "13011-rgbm-adroit" --export=args=" task=adroit_hammer-human-v1 agent=drqv2RGBM experiment_id=12000" slurm_wrapper.sh &
sbatch -J "13010-rgbm-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=drqv2RGBM experiment_id=12001" slurm_wrapper.sh &
