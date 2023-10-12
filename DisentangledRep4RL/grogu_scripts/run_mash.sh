

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_bin-picking-v2 seed=10 experiment_id=8000 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_bin-picking-v2 seed=11 experiment_id=8000 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_bin-picking-v2 seed=12 experiment_id=8000 camera_name=corner agent=V1_random_mask &

# V1_random_mask, Adroit
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 seed=10 experiment_id=8003 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 seed=11 experiment_id=8003 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 seed=12 experiment_id=8003 camera_name=fixed agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_pen-human-v1 seed=10 experiment_id=8004 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_pen-human-v1 seed=11 experiment_id=8004 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_pen-human-v1 seed=12 experiment_id=8004 camera_name=fixed agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_door-human-v1 seed=10 experiment_id=8005 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_door-human-v1 seed=11 experiment_id=8005 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_door-human-v1 seed=12 experiment_id=8005 camera_name=fixed agent=V1_random_mask &

# V1_random_mask, Kitchen, random camera
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 seed=10 experiment_id=8006 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 seed=11 experiment_id=8006 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 seed=12 experiment_id=8006 camera_name=random agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 seed=10 experiment_id=8007 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 seed=11 experiment_id=8007 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 seed=12 experiment_id=8007 camera_name=random agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 seed=10 experiment_id=8008 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 seed=11 experiment_id=8008 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 seed=12 experiment_id=8008 camera_name=random agent=V1_random_mask &


# V1_random_mask, MT1
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 seed=10 experiment_id=8009 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 seed=11 experiment_id=8009 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 seed=12 experiment_id=8009 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_box-close-v2 seed=10 experiment_id=8010 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_box-close-v2 seed=11 experiment_id=8010 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_box-close-v2 seed=12 experiment_id=8010 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 seed=10 experiment_id=8011 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 seed=11 experiment_id=8011 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 seed=12 experiment_id=8011 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-wall-v2 seed=10 experiment_id=8012 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-wall-v2 seed=11 experiment_id=8012 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-wall-v2 seed=12 experiment_id=8012 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-close-v2 seed=10 experiment_id=8013 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-close-v2 seed=11 experiment_id=8013 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-close-v2 seed=12 experiment_id=8013 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-lock-v2 seed=10 experiment_id=8014 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-lock-v2 seed=11 experiment_id=8014 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-lock-v2 seed=12 experiment_id=8014 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-open-v2 seed=10 experiment_id=8015 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-open-v2 seed=11 experiment_id=8015 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-open-v2 seed=12 experiment_id=8015 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-unlock-v2 seed=10 experiment_id=8016 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-unlock-v2 seed=11 experiment_id=8016 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-unlock-v2 seed=12 experiment_id=8016 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_hammer-v2 seed=10 experiment_id=8016 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_hammer-v2 seed=11 experiment_id=8016 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_hammer-v2 seed=12 experiment_id=8016 camera_name=corner agent=V1_random_mask &