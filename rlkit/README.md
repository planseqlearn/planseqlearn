# RLkit
Reinforcement learning framework and algorithms implemented in PyTorch.

## Installation

Setup Dependencies:
```
sudo apt-get update
sudo apt-get install curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev
sudo apt-get install libglfw3-dev libgles2-mesa-dev patchelf
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz
tar -xvf mujoco-2.1.1-linux-x86_64.tar.gz
```

Add the following to your bashrc:
```
export MUJOCO_GL='egl'
export MKL_THREADING_LAYER=GNU
export D4RL_SUPPRESS_IMPORT_ERROR='1'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mdalal/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
```

Setup Directories:
```
mkdir ~/research/<project_name>/
cd ~/research/<project_name>/
git clone git@github.com:mihdalal/doodad.git
git clone git@github.com:mihdalal/metaworld.git
git clone -b mprl git@github.com:mihdalal/rlkit.git
git clone -b mprl git@github.com:mihdalal/robosuite.git
git clone git@github.com:mihdalal/viskit.git
```

Install Ananconda environment
```
conda create -n mprl python=3.8.13
source activate mprl
```

Install packages
```
conda activate mprl
./setup_script.sh mprl # arg is python env name
```

3. (Optional) Copy `conf.py` to `conf_private.py` and edit to override defaults:
```
cp rlkit/launchers/conf.py rlkit/launchers/conf_private.py
```

## Example Commands:
Debug:

`python /path/to/experiment.py --debug`

Replicate SAC Lift results:
```
python experiments/mprl/lift/sac.py
```
Replicate SAC PickPlace results:
```
python experiments/mprl/pick_place/sac.py
```

Replicate MPRL Lift results:
```
python experiments/mprl/lift/sac_mprl.py
```

Replicate MPRL PickPlace results:
```
python experiments/mprl/pick_place/sac_mprl.py
```

## How to run sweeps from command line
`python /path/to/experiment.py -sk key1 key2 key3 -sv v11 v12 ... -sv v21 v22 ... -sv v31 v32 ... -st k1type k2type k3type`

Example:

`python experiments/ll_raps/dreamer_v2_single_task_low_level_primitives_raps_params.py -sk env_name num_low_level_actions_per_primitive -sv assembly-v2 disassemble-v2 soccer-v2 sweep-into-v2 -sv 5 10 -st str int`

Basically, for each key, add `-sv <list of values>`

## Visualizing results
During training, the results will be saved to a file called under
```
LOCAL_LOG_DIR/<exp_prefix>/<foldername>
```
 - `LOCAL_LOG_DIR` is the directory set by `rlkit.launchers.config.LOCAL_LOG_DIR`. Default name is 'results/'.
 - `<exp_prefix>` is given either to `setup_logger`.
 - `<foldername>` is auto-generated and based off of `exp_prefix`.

To visualize graphs of the results:
```
python viskit/viskit/frontend.py LOCAL_LOG_DIR/<exp_prefix>/<foldername>
```

you can add an alias of the form to your `~/.aliases` file:
```
alias vis='python viskit/viskit/frontend.py'
```

## Launching jobs with `doodad`
The `run_experiment` function makes it easy to run Python code on Amazon Web Services (AWS) or Google Cloud Platform (GCP) or Slurm by using
[this fork of doodad](git@github.com:mihdalal/doodad.git).

It's as easy as:
```
from rlkit.launchers.launcher_util import run_experiment

def function_to_run(variant):
    learning_rate = variant['learning_rate']
    ...

run_experiment(
    function_to_run,
    exp_prefix="my-experiment-name",
    mode='ssm',  # or 'ec2'
    variant={'learning_rate': 1e-3},
)
```

## Using ML-Runlog to track results:
- Follow instructions from https://pyshark.com/google-sheets-api-using-python/ to make a google cloud console account (non CMU id)
- Use these instructions to make a service account + get credentials: https://docs.gspread.org/en/latest/oauth2.html#for-bots-using-service-account
- Make the spreadsheet and share it to the client email in the service account credentials
- `pip install ml_runlog`
- Run ml_runlog
