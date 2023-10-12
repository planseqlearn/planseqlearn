# V1_random_mask, MT10
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 seed=10 experiment_id=8000 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 seed=11 experiment_id=8000 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10 seed=12 experiment_id=8000 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10-customized seed=10 experiment_id=8001 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10-customized seed=11 experiment_id=8001 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_mt10-customized seed=12 experiment_id=8001 camera_name=corner agent=V1_random_mask &

# V1_random_mask, Adroit
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 seed=10 experiment_id=8002 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 seed=11 experiment_id=8002 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_hammer-human-v1 seed=12 experiment_id=8002 camera_name=fixed agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_pen-human-v1 seed=10 experiment_id=8003 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_pen-human-v1 seed=11 experiment_id=8003 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_pen-human-v1 seed=12 experiment_id=8003 camera_name=fixed agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_relocate-human-v1 seed=10 experiment_id=8004 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_relocate-human-v1 seed=11 experiment_id=8004 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_relocate-human-v1 seed=12 experiment_id=8004 camera_name=fixed agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_door-human-v1 seed=10 experiment_id=8005 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_door-human-v1 seed=11 experiment_id=8005 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=adroit_door-human-v1 seed=12 experiment_id=8005 camera_name=fixed agent=V1_random_mask &

# V1_random_mask, Kitchen, fixed camera
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 seed=10 experiment_id=8006 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 seed=11 experiment_id=8006 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 seed=12 experiment_id=8006 camera_name=fixed agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 seed=10 experiment_id=8007 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 seed=11 experiment_id=8007 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 seed=12 experiment_id=8007 camera_name=fixed agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 seed=10 experiment_id=8008 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 seed=11 experiment_id=8008 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 seed=12 experiment_id=8008 camera_name=fixed agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 seed=10 experiment_id=8009 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 seed=11 experiment_id=8009 camera_name=fixed agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 seed=12 experiment_id=8009 camera_name=fixed agent=V1_random_mask &

# V1_random_mask, Kitchen, random camera
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 seed=10 experiment_id=8010 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 seed=11 experiment_id=8010 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 seed=12 experiment_id=8010 camera_name=random agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 seed=10 experiment_id=8011 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 seed=11 experiment_id=8011 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 seed=12 experiment_id=8011 camera_name=random agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 seed=10 experiment_id=8012 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 seed=11 experiment_id=8012 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 seed=12 experiment_id=8012 camera_name=random agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 seed=10 experiment_id=8013 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 seed=11 experiment_id=8013 camera_name=random agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 seed=12 experiment_id=8013 camera_name=random agent=V1_random_mask &


# V1_random_mask, MT1
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 seed=10 experiment_id=8014 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 seed=11 experiment_id=8014 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_assembly-v2 seed=12 experiment_id=8014 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_basketball-v2 seed=10 experiment_id=8015 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_basketball-v2 seed=11 experiment_id=8015 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_basketball-v2 seed=12 experiment_id=8015 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_bin-picking-v2 seed=10 experiment_id=8016 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_bin-picking-v2 seed=11 experiment_id=8016 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_bin-picking-v2 seed=12 experiment_id=8016 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_box-close-v2 seed=10 experiment_id=8017 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_box-close-v2 seed=11 experiment_id=8017 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_box-close-v2 seed=12 experiment_id=8017 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-topdown-v2 seed=10 experiment_id=8018 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-topdown-v2 seed=11 experiment_id=8018 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-topdown-v2 seed=12 experiment_id=8018 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 seed=10 experiment_id=8019 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 seed=11 experiment_id=8019 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-topdown-wall-v2 seed=12 experiment_id=8019 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-v2 seed=10 experiment_id=8020 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-v2 seed=11 experiment_id=8020 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-v2 seed=12 experiment_id=8020 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-wall-v2 seed=10 experiment_id=8021 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-wall-v2 seed=11 experiment_id=8021 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_button-press-wall-v2 seed=12 experiment_id=8021 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_coffee-button-v2 seed=10 experiment_id=8022 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_coffee-button-v2 seed=11 experiment_id=8022 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_coffee-button-v2 seed=12 experiment_id=8022 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_coffee-pull-v2 seed=10 experiment_id=8023 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_coffee-pull-v2 seed=11 experiment_id=8023 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_coffee-pull-v2 seed=12 experiment_id=8023 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_coffee-push-v2 seed=10 experiment_id=8024 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_coffee-push-v2 seed=11 experiment_id=8024 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_coffee-push-v2 seed=12 experiment_id=8024 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_dial-turn-v2 seed=10 experiment_id=8025 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_dial-turn-v2 seed=11 experiment_id=8025 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_dial-turn-v2 seed=12 experiment_id=8025 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_disassemble-v2 seed=10 experiment_id=8026 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_disassemble-v2 seed=11 experiment_id=8026 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_disassemble-v2 seed=12 experiment_id=8026 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-close-v2 seed=10 experiment_id=8027 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-close-v2 seed=11 experiment_id=8027 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-close-v2 seed=12 experiment_id=8027 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-lock-v2 seed=10 experiment_id=8028 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-lock-v2 seed=11 experiment_id=8028 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-lock-v2 seed=12 experiment_id=8028 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-open-v2 seed=10 experiment_id=8029 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-open-v2 seed=11 experiment_id=8029 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-open-v2 seed=12 experiment_id=8029 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-unlock-v2 seed=10 experiment_id=8030 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-unlock-v2 seed=11 experiment_id=8030 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_door-unlock-v2 seed=12 experiment_id=8030 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_hand-insert-v2 seed=10 experiment_id=8031 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_hand-insert-v2 seed=11 experiment_id=8031 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_hand-insert-v2 seed=12 experiment_id=8031 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_drawer-close-v2 seed=10 experiment_id=8032 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_drawer-close-v2 seed=11 experiment_id=8032 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_drawer-close-v2 seed=12 experiment_id=8032 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_drawer-open-v2 seed=10 experiment_id=8033 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_drawer-open-v2 seed=11 experiment_id=8033 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_drawer-open-v2 seed=12 experiment_id=8033 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_faucet-open-v2 seed=10 experiment_id=8034 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_faucet-open-v2 seed=11 experiment_id=8034 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_faucet-open-v2 seed=12 experiment_id=8034 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_faucet-close-v2 seed=10 experiment_id=8035 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_faucet-close-v2 seed=11 experiment_id=8035 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_faucet-close-v2 seed=12 experiment_id=8035 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_hammer-v2 seed=10 experiment_id=8036 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_hammer-v2 seed=11 experiment_id=8036 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_hammer-v2 seed=12 experiment_id=8036 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_handle-press-side-v2 seed=10 experiment_id=8037 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_handle-press-side-v2 seed=11 experiment_id=8037 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_handle-press-side-v2 seed=12 experiment_id=8037 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_handle-press-v2 seed=10 experiment_id=8038 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_handle-press-v2 seed=11 experiment_id=8038 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_handle-press-v2 seed=12 experiment_id=8038 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_handle-pull-side-v2 seed=10 experiment_id=8039 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_handle-pull-side-v2 seed=11 experiment_id=8039 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_handle-pull-side-v2 seed=12 experiment_id=8039 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_handle-pull-v2 seed=10 experiment_id=8040 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_handle-pull-v2 seed=11 experiment_id=8040 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_handle-pull-v2 seed=12 experiment_id=8040 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_lever-pull-v2 seed=10 experiment_id=8041 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_lever-pull-v2 seed=11 experiment_id=8041 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_lever-pull-v2 seed=12 experiment_id=8041 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_peg-insert-side-v2 seed=10 experiment_id=8042 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_peg-insert-side-v2 seed=11 experiment_id=8042 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_peg-insert-side-v2 seed=12 experiment_id=8042 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_pick-place-wall-v2 seed=10 experiment_id=8043 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_pick-place-wall-v2 seed=11 experiment_id=8043 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_pick-place-wall-v2 seed=12 experiment_id=8043 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_pick-out-of-hole-v2 seed=10 experiment_id=8044 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_pick-out-of-hole-v2 seed=11 experiment_id=8044 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_pick-out-of-hole-v2 seed=12 experiment_id=8044 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_reach-v2 seed=10 experiment_id=8045 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_reach-v2 seed=11 experiment_id=8045 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_reach-v2 seed=12 experiment_id=8045 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_push-back-v2 seed=10 experiment_id=8046 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_push-back-v2 seed=11 experiment_id=8046 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_push-back-v2 seed=12 experiment_id=8046 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_push-v2 seed=10 experiment_id=8047 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_push-v2 seed=11 experiment_id=8047 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_push-v2 seed=12 experiment_id=8047 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_pick-place-v2 seed=10 experiment_id=8048 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_pick-place-v2 seed=11 experiment_id=8048 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_pick-place-v2 seed=12 experiment_id=8048 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_plate-slide-v2 seed=10 experiment_id=8049 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_plate-slide-v2 seed=11 experiment_id=8049 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_plate-slide-v2 seed=12 experiment_id=8049 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_plate-slide-side-v2 seed=10 experiment_id=8050 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_plate-slide-side-v2 seed=11 experiment_id=8050 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_plate-slide-side-v2 seed=12 experiment_id=8050 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_plate-slide-back-v2 seed=10 experiment_id=8051 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_plate-slide-back-v2 seed=11 experiment_id=8051 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_plate-slide-back-v2 seed=12 experiment_id=8051 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_plate-slide-back-side-v2 seed=10 experiment_id=8052 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_plate-slide-back-side-v2 seed=11 experiment_id=8052 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_plate-slide-back-side-v2 seed=12 experiment_id=8052 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_peg-unplug-side-v2 seed=10 experiment_id=8053 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_peg-unplug-side-v2 seed=11 experiment_id=8053 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_peg-unplug-side-v2 seed=12 experiment_id=8053 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_soccer-v2 seed=10 experiment_id=8054 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_soccer-v2 seed=11 experiment_id=8054 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_soccer-v2 seed=12 experiment_id=8054 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_stick-push-v2 seed=10 experiment_id=8055 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_stick-push-v2 seed=11 experiment_id=8055 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_stick-push-v2 seed=12 experiment_id=8055 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_stick-pull-v2 seed=10 experiment_id=8056 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_stick-pull-v2 seed=11 experiment_id=8056 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_stick-pull-v2 seed=12 experiment_id=8056 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_push-wall-v2 seed=10 experiment_id=8057 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_push-wall-v2 seed=11 experiment_id=8057 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_push-wall-v2 seed=12 experiment_id=8057 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_reach-wall-v2 seed=10 experiment_id=8058 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_reach-wall-v2 seed=11 experiment_id=8058 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_reach-wall-v2 seed=12 experiment_id=8058 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_shelf-place-v2 seed=10 experiment_id=8059 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_shelf-place-v2 seed=11 experiment_id=8059 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_shelf-place-v2 seed=12 experiment_id=8059 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_sweep-into-v2 seed=10 experiment_id=8060 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_sweep-into-v2 seed=11 experiment_id=8060 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_sweep-into-v2 seed=12 experiment_id=8060 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_sweep-v2 seed=10 experiment_id=8061 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_sweep-v2 seed=11 experiment_id=8061 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_sweep-v2 seed=12 experiment_id=8061 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_window-open-v2 seed=10 experiment_id=8062 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_window-open-v2 seed=11 experiment_id=8062 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_window-open-v2 seed=12 experiment_id=8062 camera_name=corner agent=V1_random_mask &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_window-close-v2 seed=10 experiment_id=8063 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_window-close-v2 seed=11 experiment_id=8063 camera_name=corner agent=V1_random_mask &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=metaworld_window-close-v2 seed=12 experiment_id=8063 camera_name=corner agent=V1_random_mask &
