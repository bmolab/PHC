# How PHC Works

⚠️ The explanation below shows how PHC works without our modifications. This is intended to give you a general sense of how to use PHC's default settings.

The main file that starts simulation is `phc/run_hydra.py`. A nice introduction to the file is https://github.com/Denys88/rl_games/blob/master/docs/ISAAC_GYM.md.

## Converting Files To PHC Readables

PHC converts files prior to loading it (refer to the READ.ME). AMASS files need to be converted by using scripts/data_process/convert_amass_data.py to pkl files before loading into running for run_hydra.py.

## Sample Command

```
python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv env=env_im_getup_mcp env.task=HumanoidImMCPDemo robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False env.z_activation=relu env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth'] env.num_envs=1 env.obs_v=7 headless=False epoch=-1 test=True no_virtual_display=True
```

## Run real-time webcam Demo

See the [video_to_control_demo.md](docs\video_to_control_demo.md)

The key config parameter which leads to where to get the torque and force data are: 
- config file: env=env_im_getup_mcp_test
- task type: env.task=HumanoidImMCPDemo 
- env.obs_v=7 

# Run AMASS to PHC Streaming Pipeline

**New Implementation based on PHC's current framework**: Streams 3D data from an AMASS `.npz` file (via `amassStreamer.py`) into a physics-based humanoid simulation (Isaac Gym via `run_hydra.py`). The two components communicate asynchronously over a WebSocket connection.

* **Server (`amassStreamer.py`):** Loads AMASS motion data, processes it (SMPL FK, rotation, scaling), and serves the pose frames via a WebSocket.
* **Client (`HumanoidImMCPDemo` in PHC):** Connects to the server, requests poses, mimics the motion, and optionally logs Torques/Forces.


## How to Run

### **Step 1: Start the Streamer**

This script acts as the "Motion Source." It must be running before the simulation starts.

```bash
# Basic usage
python amassStreamer.py --file path/to/your_motion.npz --smpl data/smpl/

# With tracking adjustments
python amassStreamer.py \
  --file path/to/motion.npz \
  --smpl data/smpl/ \
  --fps 30 \
  --scale 1.0 \
  --rot -90

```
#### Key Hyperparameters & Flags

* **`--rot -90`**: Critical for aligning AMASS (Y-up usually) with Isaac Gym (Z-up). The script rotates the skeleton around the Z-axis.
* **`--scale`**: Adjusts the size of the skeleton (default 0.85).

### **Step 2: Start the PHC Simulation**

Run PHC's demo script which connects to the streamer. This part is majority implemented by PHC. We did some minor modification and fix. 

The command is similar to how PHC originally runs the demo: See the [video_to_control_demo.md](docs\video_to_control_demo.md)

```bash
python phc/run_hydra.py \
    learning=im_mcp \
    exp_name=phc_kp_mcp_demo \
    env=env_im_getup_mcp \
    env.task=HumanoidImMCPDemo \
    robot=smpl_humanoid \
    env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
    env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth'] \
    env.num_envs=1 \
    env.obs_v=7 \
    headless=False \
    test=True \
```

#### Key Hyperparameters & Flags

* **`env.task=HumanoidImMCPDemo`**: This is crucial. It tells to load the specific class defined in `humanoid_im_mcp_demo.py` that contains the WebSocket client logic.
* **`env.obs_v=7`**: This observation version is hardcoded in PHC's code. Using other versions might result in shape mismatches for the tracking observation tensor.
*  To enable torque/forces logging, add **`log_forces: True`** to the environment configuration file used in your run command.  
    * **Example:** If your command uses `env=env_im_getup_mcp`, edit the file `phc/data/cfg/env/env_im_getup_mcp.yaml`


## How to Get Torque and Forces

The model checkpoint `output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth` was trained with an observation shape of 574. Change `self_obs_v = 3` changes the observation size, causing a size mismatch crash. 

⚠️ So do not change `self_obs_v`. Instead, we added the `log_forces` flag, set it to True to trigger the force/torque streaming process.

### How the Data is Retrieved
[Isaac Gym Force Sensors Documentation](https://docs.robotsfan.com/isaacgym/programming/forcesensors.html)


#### Internal Motor Efforts (`dof_force_tensor`)
* **PHC already have this implemented for every step** 
* **Source:** `gym.acquire_dof_force_tensor(sim)`
* **What it is:** The internal force/torque the robot's motors apply to move its own joints (Muscle/Actuation effort).
* **Units:** Newton-meters ($N \cdot m$) for rotational joints


#### External Contact Forces (`vec_sensor_tensor`)
* **Source:** `gym.acquire_force_sensor_tensor(sim)`
* **What it is:** The external reaction forces exerted by the ground onto the robot's feet (Ground Reaction Force).
* **Units:** 6-DOF Wrench (3 Forces + 3 Torques)


### How the Data is being sent out

We implemented `TorqueForceSender` to sends data via OSC or a similar UDP protocol (implied by `host="127.0.0.1", port=9000`), you need a **Receiver**.


