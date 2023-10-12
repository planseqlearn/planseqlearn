sbatch -J "15000-detach-mask-adroit" --export=args=" task=adroit_hammer-human-v1 agent=V1 experiment_id=15000 agent.detach_mask_decoder=true" slurm_wrapper_2.sh &
sbatch -J "15001-detach-recon-adroit" --export=args=" task=adroit_hammer-human-v1 agent=V1 experiment_id=15001 agent.detach_reconstruction_decoder=true" slurm_wrapper_2.sh &
sbatch -J "15002-detach-mask-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=15002 agent.detach_mask_decoder=true" slurm_wrapper_2.sh &
sbatch -J "15003-detach-recon-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=15003 agent.detach_reconstruction_decoder=true" slurm_wrapper_2.sh &

sbatch -J "13017-noisy-adroit" --export=args=" task=adroit_hammer-human-v1 agent=V1 experiment_id=13017 noisy_mask_drop_prob=0.005" slurm_wrapper_2.sh &
sbatch -J "13018-noisy-adroit" --export=args=" task=adroit_hammer-human-v1 agent=V1 experiment_id=13018 noisy_mask_drop_prob=0.05" slurm_wrapper_2.sh &
sbatch -J "13019-noisy-adroit" --export=args=" task=adroit_hammer-human-v1 agent=V1 experiment_id=13019 noisy_mask_drop_prob=0.10" slurm_wrapper_2.sh &
sbatch -J "13020-noisy-adroit" --export=args=" task=adroit_hammer-human-v1 agent=V1 experiment_id=13020 noisy_mask_drop_prob=0.25" slurm_wrapper_2.sh &
sbatch -J "13021-noisy-adroit" --export=args=" task=adroit_hammer-human-v1 agent=V1 experiment_id=13021 noisy_mask_drop_prob=0.50" slurm_wrapper_2.sh &
sbatch -J "13022-noisy-adroit" --export=args=" task=adroit_hammer-human-v1 agent=V1 experiment_id=13022 noisy_mask_drop_prob=0.15" slurm_wrapper_2.sh &
sbatch -J "13023-noisy-adroit" --export=args=" task=adroit_hammer-human-v1 agent=V1 experiment_id=13023 noisy_mask_drop_prob=0.30" slurm_wrapper_2.sh &

sbatch -J "13005-noisy-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=13005 noisy_mask_drop_prob=0.00" slurm_wrapper.sh &
sbatch -J "13006-noisy-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=13006 noisy_mask_drop_prob=0.05" slurm_wrapper.sh &
sbatch -J "13007-noisy-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=13007 noisy_mask_drop_prob=0.10" slurm_wrapper.sh &
sbatch -J "13008-noisy-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=13008 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "13009-noisy-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=13009 noisy_mask_drop_prob=0.50" slurm_wrapper.sh &


sbatch -J "13012-noisy-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=13012 noisy_mask_drop_prob=0.15" slurm_wrapper_2.sh &
sbatch -J "13013-noisy-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=13013 noisy_mask_drop_prob=0.30" slurm_wrapper_2.sh &
sbatch -J "13014-noisy-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=13014 noisy_mask_drop_prob=0.005" slurm_wrapper_2.sh &
sbatch -J "13015-noisy-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=13015 noisy_mask_drop_prob=0.01" slurm_wrapper_2.sh &
sbatch -J "13016-noisy-metaworld" --export=args=" task=metaworld_bin-picking-v2 agent=V1 experiment_id=13016 noisy_mask_drop_prob=0.02" slurm_wrapper_2.sh &