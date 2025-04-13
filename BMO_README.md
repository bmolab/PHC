# How PHC Works

The main file that starts simulation is phc/run_hydra.py. A nice introduction to the file is https://github.com/Denys88/rl_games/blob/master/docs/ISAAC_GYM.md.

# Converting Files To PHC Readables

PHC converts files prior to loading it (refer to the READ.ME). AMASS files need to be converted by using scripts/data_process/convert_amass_data.py to pkl files before loading into running for run_hydra.py.

A module could be created to be created in dpg system to load and convert to the pkl format and preprocess.


# Sample Command

```
python phc/run_hydra.py learning=im_mcp_big learning.params.network.ending_act=False exp_name=phc_comp_kp_2 env.obs_v=7 env=env_im_getup_mcp robot=smpl_humanoid robot.real_weight_porpotion_boxes=False env.motion_file=sample_data/accad.pkl env.models=['output/HumanoidIm/phc_kp_2/Humanoid.pth'] env.num_prim=3 env.num_envs=1  headless=False epoch=-1 test=True
```

# Pulling Data For DPG System
Pulling out data for dpg system will require modifying the phc/run_hydra.py and files related. Specifically the main workings of the runner.run in phc/run_hydra.py is in phc/learing/im_amp_players.py def run function.