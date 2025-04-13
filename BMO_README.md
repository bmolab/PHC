# How PHC is set up - RL Games

The main file that starts simulation is phc/run_hydra.py. Few nice introductions to how PHC is set up is are:
- https://github.com/Denys88/rl_games/blob/master/docs/ISAAC_GYM.md.
- https://github.com/Denys88/rl_games/blob/master/docs/HOW_TO_RL_GAMES.md

The main idea is split as follows:

- phc/run_hydra.py: Starting file that runs simulations 
- phc/data/cfg/: Configuration files
- -  env: imitation environment parameters. Determins which task to run.
- -  learning: neural network setups and parameters
- phc/env/tasks: Imiation Environment set up (similar to gym environments) with base environment being HumanoidIm.py and HumanoidImMCP.py. These two files provides control similar to openai gym environment with the step function.

# Computing Observations

PHC's way of computing observations is quite complicated with various versions. For now it seems like the version 7 is generally used. All compute observation functions can be found in /phc/env/tasks/humanoid_im.py. For the live imitation demo, the compute observation is overridden in /phc/env/tasks/humanoid_im_mcp_demo.

# Converting Files To PHC Readables

PHC converts files prior to loading it (refer to the READ.ME). AMASS files need to be converted by using scripts/data_process/convert_amass_data.py to pkl files before loading into running for run_hydra.py.

A module could be created to be created in dpg system to load and convert to the pkl format and preprocess.


# Sample Command For Runing PHC

```
python phc/run_hydra.py learning=im_mcp_big learning.params.network.ending_act=False exp_name=phc_comp_kp_2 env.obs_v=7 env=env_im_getup_mcp robot=smpl_humanoid robot.real_weight_porpotion_boxes=False env.motion_file=sample_data/accad.pkl env.models=['output/HumanoidIm/phc_kp_2/Humanoid.pth'] env.num_prim=3 env.num_envs=1  headless=False epoch=-1 test=True
```

# Integrating with DPG System

Pulling out data for dpg system will require modifying the phc/run_hydra.py and files related. Specifically the main workings of the runner.run in phc/run_hydra.py is in phc/learingn/im_amp_players.py def run function. This seems to be qutie difficult as it would mean taking functions apart in PHC. 

Another method would be working with communication between processes. PHC has a demo for live motion capture with camera. This simply runs two applications, one being a live pose capture, another being the isaac gym environment. The Isaac gym envirnoment then queries the live motion capture application for pose information each time. A similar paradigm could be done with dpg system where dpg system queries or receives data from the Isaac gym environment through requests.

