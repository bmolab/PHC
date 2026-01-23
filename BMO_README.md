# How PHC Works

The main file that starts simulation is phc/run_hydra.py. A nice introduction to the file is https://github.com/Denys88/rl_games/blob/master/docs/ISAAC_GYM.md.

# Converting Files To PHC Readables

PHC converts files prior to loading it (refer to the READ.ME). AMASS files need to be converted by using scripts/data_process/convert_amass_data.py to pkl files before loading into running for run_hydra.py.

A module could be created to be created in dpg system to load and convert to the pkl format and preprocess.


# Sample Command

```
python phc/run_hydra.py learning=im_mcp_big learning.params.network.ending_act=False exp_name=phc_comp_kp_2 env.obs_v=7 env=env_im_getup_mcp robot=smpl_humanoid robot.real_weight_porpotion_boxes=False env.motion_file=sample_data/accad.pkl env.models=['output/HumanoidIm/phc_kp_2/Humanoid.pth'] env.num_prim=3 env.num_envs=1  headless=False epoch=-1 test=True
```
python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv env=env_im_getup_mcp env.task=HumanoidImMCPDemo robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False env.z_activation=relu env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth'] env.num_envs=1 env.obs_v=7 headless=False epoch=-1 test=True no_virtual_display=True

# Pulling Data For DPG System
Pulling out data for dpg system will require modifying the phc/run_hydra.py and files related. Specifically the main workings of the runner.run in phc/run_hydra.py is in phc/learing/im_amp_players.py def run function.
<<<<<<< Updated upstream:BMO_README.md

# Run real-time webcam Demo

See the [video_to_control_demo.md](docs\video_to_control_demo.md)

The key config parameter which leads to where to get the torque and force data are: 
- config file: env=env_im_getup_mcp_test
- task type: env.task=HumanoidImMCPDemo 
- env.obs_v=7 

# Getting Torque & Force Data in PHC
The model checkpoint `output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth` was trained with an observation shape of 574. Change `self_obs_v = 3` changes the observation size, causing a size mismatch crash. 

⚠️ So do not change `self_obs_v`. Instead, we added the `log_forces` flag, set it to True to trigger the force/torque streaming process.

## How the Data is Retrieved
[Isaac Gym Force Sensors Documentation](https://docs.robotsfan.com/isaacgym/programming/forcesensors.html)


### Internal Motor Efforts (`dof_force_tensor`)
* **PHC already have this implemented for every step** 
* **Source:** `gym.acquire_dof_force_tensor(sim)`
* **What it is:** The internal force/torque the robot's motors apply to move its own joints (Muscle/Actuation effort).
* **Units:** Newton-meters ($N \cdot m$) for rotational joints


### External Contact Forces (`vec_sensor_tensor`)
* **Source:** `gym.acquire_force_sensor_tensor(sim)`
* **What it is:** The external reaction forces exerted by the ground onto the robot's feet (Ground Reaction Force).
* **Units:** 6-DOF Wrench (3 Forces + 3 Torques)


### Implementation Note
we modified `phc/env/tasks/humanoid.py` and `phc/env/tasks/humanoid_im_mcp_demo.py`

To enable the changes, add `log_forces: True` to the environment configuration file used in your run command.

* **Example:** If your command uses `env=env_im_getup_mcp`, edit the file `phc/data/cfg/env/env_im_getup_mcp.yaml`


