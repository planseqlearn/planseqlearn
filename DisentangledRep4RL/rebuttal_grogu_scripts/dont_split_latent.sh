export bg_dataset_path=/home/sbahl2/research/DisentangledRep4RL/DAVIS/JPEGImages/480p/

export set_v1_hyperparams="agent.latent_dim=4096 agent.reconstruction_loss_coeff=1 agent.mask_loss_coeff=0.25"
sbatch -J "40000-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40000 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40001-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40001 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40002-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40002 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40003-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40003 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40004-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40004 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40005-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40005 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40006-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40006 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40007-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40007 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40008-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40008 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40009-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40009 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40010-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40010 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40011-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40011 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40012-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40012 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40013-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40013 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40014-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40014 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40015-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40015 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40016-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=40016 agent.split_latent=false distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "40017-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=40017 agent.split_latent=false distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &

export set_v1_hyperparams="agent.latent_dim=4096 agent.reconstruction_loss_coeff=0.5 agent.mask_loss_coeff=0.125"
sbatch -J "40018-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40018 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40019-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40019 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40020-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40020 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40021-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40021 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40022-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40022 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40023-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40023 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40024-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40024 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40025-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40025 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40026-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40026 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40027-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40027 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40028-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40028 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40029-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40029 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40030-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40030 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40031-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40031 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40032-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40032 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40033-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40033 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40034-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=40034 agent.split_latent=false distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "40035-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=40035 agent.split_latent=false distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &

export set_v1_hyperparams="agent.latent_dim=4096 agent.reconstruction_loss_coeff=0.1 agent.mask_loss_coeff=0.025"
sbatch -J "40036-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40036 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40037-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40037 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40038-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40038 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40039-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40039 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40040-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40040 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40041-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40041 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40042-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40042 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40043-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40043 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40044-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40044 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40045-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40045 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40046-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40046 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40047-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40047 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40048-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40048 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40049-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40049 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40050-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40050 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40051-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40051 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40052-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=40052 agent.split_latent=false distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "40053-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=40053 agent.split_latent=false distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &

export set_v1_hyperparams="agent.latent_dim=4096 agent.reconstruction_loss_coeff=0.05 agent.mask_loss_coeff=0.0125"
sbatch -J "40054-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40054 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40055-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40055 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40056-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40056 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40057-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40057 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40058-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40058 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40059-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40059 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40060-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40060 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40061-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40061 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40062-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40062 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40063-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40063 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40064-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40064 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40065-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40065 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40066-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40066 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40067-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40067 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40068-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40068 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40069-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40069 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40070-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=40070 agent.split_latent=false distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "40071-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=40071 agent.split_latent=false distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &

export set_v1_hyperparams="agent.latent_dim=4096 agent.reconstruction_loss_coeff=0.01 agent.mask_loss_coeff=0.0025"
sbatch -J "40072-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40072 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40073-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40073 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40074-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40074 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40075-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40075 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40076-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40076 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40077-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40077 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40078-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40078 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40079-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40079 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40080-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40080 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40081-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40081 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40082-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40082 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40083-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40083 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40084-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40084 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40085-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40085 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40086-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40086 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40087-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40087 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40088-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=40088 agent.split_latent=false distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "40089-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=40089 agent.split_latent=false distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &

export set_v1_hyperparams="agent.latent_dim=4096 agent.reconstruction_loss_coeff=0.001 agent.mask_loss_coeff=0.00025"
sbatch -J "40090-metaworld_box-close-v2" --export=args=" task=metaworld_box-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40090 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40091-metaworld_bin-picking-v2" --export=args=" task=metaworld_bin-picking-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40091 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40092-metaworld_assembly-v2" --export=args=" task=metaworld_assembly-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40092 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40093-metaworld_button-press-topdown-wall-v2" --export=args=" task=metaworld_button-press-topdown-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40093 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40094-metaworld_button-press-wall-v2" --export=args=" task=metaworld_button-press-wall-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40094 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40095-metaworld_door-close-v2" --export=args=" task=metaworld_door-close-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40095 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40096-metaworld_door-lock-v2" --export=args=" task=metaworld_door-lock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40096 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40097-metaworld_door-open-v2" --export=args=" task=metaworld_door-open-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40097 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40098-metaworld_door-unlock-v2" --export=args=" task=metaworld_door-unlock-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40098 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40099-metaworld_hammer-v2" --export=args=" task=metaworld_hammer-v2 agent=V1 ${set_v1_hyperparams} experiment_id=40099 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40100-kitchen_kitchen-kettle-v0" --export=args=" task=kitchen_kitchen-kettle-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40100 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40101-kitchen_kitchen-light-v0" --export=args=" task=kitchen_kitchen-light-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40101 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40102-kitchen_kitchen-slider-v0" --export=args=" task=kitchen_kitchen-slider-v0 agent=V1 ${set_v1_hyperparams} experiment_id=40102 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40103-adroit_hammer-human-v1" --export=args=" task=adroit_hammer-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40103 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40104-adroit_door-human-v1" --export=args=" task=adroit_door-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40104 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40105-adroit_pen-human-v1" --export=args=" task=adroit_pen-human-v1 agent=V1 ${set_v1_hyperparams} experiment_id=40105 agent.split_latent=false" slurm_wrapper.sh &
sbatch -J "40106-distracting_walker_walk" --export=args=" task=distracting_walker_walk agent=V1 ${set_v1_hyperparams} experiment_id=40106 agent.split_latent=false distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
sbatch -J "40107-distracting_cup_catch" --export=args=" task=distracting_cup_catch agent=V1 ${set_v1_hyperparams} experiment_id=40107 agent.split_latent=false distraction.types=\[background\] distraction.dataset_path=bg_dataset_path" slurm_wrapper.sh &
