# DrQv2, fixed camera
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2 seed=10 experiment_id=402 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2 seed=11 experiment_id=402 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2 seed=12 experiment_id=402 camera_name=fixed &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2 seed=10 experiment_id=403 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2 seed=11 experiment_id=403 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2 seed=12 experiment_id=403 camera_name=fixed &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2 seed=10 experiment_id=404 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2 seed=11 experiment_id=404 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2 seed=12 experiment_id=404 camera_name=fixed &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2 seed=10 experiment_id=405 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2 seed=11 experiment_id=405 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2 seed=12 experiment_id=405 camera_name=fixed &

# DrQv2, random camera
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2 seed=10 experiment_id=406 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2 seed=11 experiment_id=406 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2 seed=12 experiment_id=406 camera_name=random &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2 seed=10 experiment_id=407 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2 seed=11 experiment_id=407 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2 seed=12 experiment_id=407 camera_name=random &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2 seed=10 experiment_id=408 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2 seed=11 experiment_id=408 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2 seed=12 experiment_id=408 camera_name=random &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2 seed=10 experiment_id=409 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2 seed=11 experiment_id=409 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2 seed=12 experiment_id=409 camera_name=random &

# DrQv2AE, fixed camera
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2AE seed=10 experiment_id=410 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2AE seed=11 experiment_id=410 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2AE seed=12 experiment_id=410 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2AE seed=10 experiment_id=411 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2AE seed=11 experiment_id=411 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2AE seed=12 experiment_id=411 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2AE seed=10 experiment_id=412 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2AE seed=11 experiment_id=412 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2AE seed=12 experiment_id=412 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2AE seed=10 experiment_id=413 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2AE seed=11 experiment_id=413 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2AE seed=12 experiment_id=413 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=fixed &

# DrQv2AE, random camera
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2AE seed=10 experiment_id=414 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2AE seed=11 experiment_id=414 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=drqv2AE seed=12 experiment_id=414 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2AE seed=10 experiment_id=415 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2AE seed=11 experiment_id=415 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=drqv2AE seed=12 experiment_id=415 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2AE seed=10 experiment_id=416 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2AE seed=11 experiment_id=416 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=drqv2AE seed=12 experiment_id=416 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2AE seed=10 experiment_id=417 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2AE seed=11 experiment_id=417 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=drqv2AE seed=12 experiment_id=417 agent.reconstruction_loss_coeff=2 latent_dim=4096 camera_name=random &

# V1, fixed camera
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=V1 seed=10 experiment_id=418 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=V1 seed=11 experiment_id=418 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=V1 seed=12 experiment_id=418 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=fixed &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=V1 seed=10 experiment_id=419 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=V1 seed=11 experiment_id=419 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=V1 seed=12 experiment_id=419 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=fixed &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=V1 seed=10 experiment_id=420 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=V1 seed=11 experiment_id=420 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=V1 seed=12 experiment_id=420 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=fixed &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=V1 seed=10 experiment_id=421 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=V1 seed=11 experiment_id=421 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=fixed &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=V1 seed=12 experiment_id=421 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=fixed &

# V1, random camera
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=V1 seed=10 experiment_id=422 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=V1 seed=11 experiment_id=422 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-kettle-v0 agent=V1 seed=12 experiment_id=422 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=random &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=V1 seed=10 experiment_id=423 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=V1 seed=11 experiment_id=423 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-light-v0 agent=V1 seed=12 experiment_id=423 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=random &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=V1 seed=10 experiment_id=424 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=V1 seed=11 experiment_id=424 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-microwave-v0 agent=V1 seed=12 experiment_id=424 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=random &

export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=V1 seed=10 experiment_id=425 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=V1 seed=11 experiment_id=425 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=random &
export CUDA_VISIBLE_DEVICES=0; python disrep4rl/train.py task=kitchen_kitchen-slider-v0 agent=V1 seed=12 experiment_id=425 agent.mask_loss_coeff=2.5e-1 agent.reconstruction_loss_coeff=1 latent_dim=4096 camera_name=random &
