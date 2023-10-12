sbatch -J "11000-noisy-adroit" --export=args=" task=adroit_hammer-human-v1 agent=V1 experiment_id=11000 noisy_mask_drop_prob=0.00" slurm_wrapper.sh &
sbatch -J "11001-noisy-adroit" --export=args=" task=adroit_hammer-human-v1 agent=V1 experiment_id=11001 noisy_mask_drop_prob=0.05" slurm_wrapper.sh &
sbatch -J "11002-noisy-adroit" --export=args=" task=adroit_hammer-human-v1 agent=V1 experiment_id=11002 noisy_mask_drop_prob=0.10" slurm_wrapper.sh &
sbatch -J "11003-noisy-adroit" --export=args=" task=adroit_hammer-human-v1 agent=V1 experiment_id=11003 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "11004-noisy-adroit" --export=args=" task=adroit_hammer-human-v1 agent=V1 experiment_id=11004 noisy_mask_drop_prob=0.50" slurm_wrapper.sh &

sbatch -J "11005-noisy-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=11005 noisy_mask_drop_prob=0.00" slurm_wrapper.sh &
sbatch -J "11006-noisy-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=11006 noisy_mask_drop_prob=0.05" slurm_wrapper.sh &
sbatch -J "11007-noisy-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=11007 noisy_mask_drop_prob=0.10" slurm_wrapper.sh &
sbatch -J "11008-noisy-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=11008 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "11009-noisy-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=11009 noisy_mask_drop_prob=0.50" slurm_wrapper.sh &
