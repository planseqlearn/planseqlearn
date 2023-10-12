export bg_dataset_path=/home/sbahl2/research/DisentangledRep4RL/DAVIS/JPEGImages/480p/

export set_v1_hyperparams="agent.latent_dim=4096 agent.reconstruction_loss_coeff=1 agent.mask_loss_coeff=0.25"
sbatch -J "50000-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50000 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50001-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50001 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50002-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50002 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50003-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50003 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50004-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50004 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50005-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50005 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50006-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50006 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50007-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50007 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50008-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50008 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50009-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50009 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50010-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50010 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50011-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50011 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50012-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50012 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50013-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50013 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50014-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50014 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50015-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50015 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50016-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=50016 noisy_mask_drop_prob=0.25 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "50017-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=50017 noisy_mask_drop_prob=0.25 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &

sbatch -J "50018-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50018 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50019-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50019 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50020-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50020 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50021-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50021 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50022-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50022 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50023-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50023 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50024-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50024 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50025-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50025 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50026-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50026 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50027-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50027 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50028-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50028 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50029-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50029 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50030-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50030 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50031-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50031 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50032-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50032 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50033-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50033 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50034-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=50034 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "50035-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=50035 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &



export set_v1_hyperparams="agent.latent_dim=4096 agent.reconstruction_loss_coeff=0.5 agent.mask_loss_coeff=0.125"
sbatch -J "50036-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50036 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50037-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50037 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50038-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50038 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50039-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50039 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50040-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50040 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50041-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50041 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50042-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50042 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50043-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50043 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50044-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50044 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50045-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50045 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50046-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50046 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50047-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50047 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50048-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50048 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50049-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50049 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50050-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50050 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50051-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50051 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50052-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=50052 noisy_mask_drop_prob=0.25 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "50053-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=50053 noisy_mask_drop_prob=0.25 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &

sbatch -J "50054-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50054 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50055-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50055 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50056-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50056 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50057-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50057 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50058-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50058 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50059-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50059 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50060-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50060 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50061-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50061 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50062-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50062 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50063-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50063 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50064-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50064 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50065-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50065 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50066-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50066 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50067-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50067 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50068-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50068 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50069-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50069 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50070-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=50070 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "50071-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=50071 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &



export set_v1_hyperparams="agent.latent_dim=4096 agent.reconstruction_loss_coeff=0.1 agent.mask_loss_coeff=0.025"
sbatch -J "50072-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50072 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50073-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50073 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50074-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50074 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50075-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50075 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50076-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50076 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50077-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50077 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50078-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50078 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50079-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50079 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50080-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50080 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50081-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50081 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50082-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50082 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50083-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50083 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50084-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50084 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50085-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50085 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50086-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50086 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50087-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50087 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50088-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=50088 noisy_mask_drop_prob=0.25 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "50089-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=50089 noisy_mask_drop_prob=0.25 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &

sbatch -J "50090-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50090 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50091-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50091 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50092-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50092 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50093-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50093 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50094-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50094 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50095-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50095 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50096-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50096 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50097-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50097 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50098-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50098 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50099-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50099 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50100-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50100 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50101-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50101 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50102-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50102 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50103-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50103 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50104-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50104 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50105-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50105 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50106-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=50106 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "50107-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=50107 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &



export set_v1_hyperparams="agent.latent_dim=4096 agent.reconstruction_loss_coeff=0.05 agent.mask_loss_coeff=0.0125"
sbatch -J "50108-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50108 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50109-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50109 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50110-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50110 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50111-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50111 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50112-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50112 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50113-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50113 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50114-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50114 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50115-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50115 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50116-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50116 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50117-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50117 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50118-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50118 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50119-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50119 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50120-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50120 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50121-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50121 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50122-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50122 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50123-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50123 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50124-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=50124 noisy_mask_drop_prob=0.25 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "50125-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=50125 noisy_mask_drop_prob=0.25 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &

sbatch -J "50126-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50126 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50127-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50127 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50128-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50128 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50129-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50129 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50130-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50130 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50131-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50131 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50132-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50132 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50133-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50133 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50134-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50134 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50135-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50135 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50136-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50136 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50137-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50137 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50138-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50138 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50139-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50139 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50140-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50140 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50141-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50141 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50142-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=50142 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "50143-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=50143 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &



export set_v1_hyperparams="agent.latent_dim=4096 agent.reconstruction_loss_coeff=0.01 agent.mask_loss_coeff=0.0025"
sbatch -J "50144-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50144 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50145-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50145 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50146-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50146 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50147-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50147 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50148-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50148 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50149-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50149 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50150-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50150 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50151-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50151 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50152-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50152 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50153-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50153 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50154-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50154 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50155-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50155 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50156-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50156 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50157-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50157 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50158-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50158 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50159-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50159 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50160-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=50160 noisy_mask_drop_prob=0.25 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "50161-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=50161 noisy_mask_drop_prob=0.25 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &

sbatch -J "50162-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50162 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50163-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50163 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50164-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50164 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50165-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50165 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50166-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50166 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50167-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50167 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50168-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50168 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50169-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50169 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50170-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50170 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50171-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50171 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50172-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50172 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50173-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50173 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50174-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50174 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50175-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50175 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50176-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50176 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50177-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50177 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50178-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=50178 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "50179-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=50179 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &



export set_v1_hyperparams="agent.latent_dim=4096 agent.reconstruction_loss_coeff=0.001 agent.mask_loss_coeff=0.00025"
sbatch -J "50180-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50180 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50181-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50181 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50182-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50182 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50183-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50183 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50184-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50184 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50185-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50185 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50186-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50186 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50187-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50187 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50188-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50188 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50189-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50189 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50190-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50190 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50191-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50191 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50192-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50192 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50193-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50193 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50194-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50194 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50195-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50195 noisy_mask_drop_prob=0.25" slurm_wrapper.sh &
sbatch -J "50196-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=50196 noisy_mask_drop_prob=0.25 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "50197-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=50197 noisy_mask_drop_prob=0.25 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &

sbatch -J "50198-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50198 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50199-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50199 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50200-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50200 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50201-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50201 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50202-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50202 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50203-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50203 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50204-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50204 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50205-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50205 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50206-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50206 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50207-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=50207 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50208-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50208 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50209-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50209 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50210-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=50210 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50211-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50211 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50212-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50212 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50213-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=50213 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5" slurm_wrapper.sh &
sbatch -J "50214-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=50214 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "50215-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=50215 slim_mask_cfg.use_slim_mask=true slim_mask_cfg.scale=3 slim_mask_cfg.sigma=0.5 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &




# RGBM Experiments
sbatch -J "50216-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=drqv2RGBM experiment_id=50216" slurm_wrapper.sh &
sbatch -J "50217-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=drqv2RGBM experiment_id=50217" slurm_wrapper.sh &
sbatch -J "50218-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=drqv2RGBM experiment_id=50218" slurm_wrapper.sh &
sbatch -J "50219-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=drqv2RGBM experiment_id=50219" slurm_wrapper.sh &
sbatch -J "50220-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=drqv2RGBM experiment_id=50220" slurm_wrapper.sh &
sbatch -J "50221-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=drqv2RGBM experiment_id=50221" slurm_wrapper.sh &
sbatch -J "50222-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=drqv2RGBM experiment_id=50222" slurm_wrapper.sh &
sbatch -J "50223-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=drqv2RGBM experiment_id=50223" slurm_wrapper.sh &
sbatch -J "50224-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=drqv2RGBM experiment_id=50224" slurm_wrapper.sh &
sbatch -J "50225-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=drqv2RGBM experiment_id=50225" slurm_wrapper.sh &
sbatch -J "50226-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=drqv2RGBM experiment_id=50226" slurm_wrapper.sh &
sbatch -J "50227-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=drqv2RGBM experiment_id=50227" slurm_wrapper.sh &
sbatch -J "50228-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=drqv2RGBM experiment_id=50228" slurm_wrapper.sh &
sbatch -J "50229-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=drqv2RGBM experiment_id=50229" slurm_wrapper.sh &
sbatch -J "50230-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=drqv2RGBM experiment_id=50230" slurm_wrapper.sh &
sbatch -J "50231-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=drqv2RGBM experiment_id=50231" slurm_wrapper.sh &
sbatch -J "50232-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=drqv2RGBM experiment_id=50232 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "50233-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=drqv2RGBM experiment_id=50233 distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
