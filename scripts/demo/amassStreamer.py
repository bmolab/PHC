import numpy as np
import time
import argparse
import torch
import sys
import os
from aiohttp import web
import json
from smpl_sim.smpllib.smpl_parser import SMPL_Parser


'''
Coordinates:
amass file: z up
phc isaac gyn: z up
Note: no rotation needed when sending data
'''

'''
To Fix:
Add to run command "control.decimation=4 sim.physx.step_dt=\"1/120.0\"" //for physics rate 

Physics Rate sim_params.dt: the frequency at which the engine (Isaac Gym) updates the state 
Control Rate:  the frequency at which the agent's policy runs to decide on a new action
Control time step: the duration of one control cycle


'''

MAX_PEOPLE = 5 
NUM_JOINTS = 24  # Standard SMPL joint count
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Map SMPL joint order to Mujoco order if needed
SMPL_2_MUJOCO = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]

class AmassStreamer:
    def __init__(self, npz_path, smpl_model_path, target_fps=30, auto_ground=True, use_neutral_shape=True):
        self.auto_ground = auto_ground
        self.use_neutral_shape = use_neutral_shape
        self.target_fps = target_fps
        self.dt = 1.0 / self.target_fps 
        self.loop_print_count = -1
        
        print(f"--- Initializing AMASS Streamer---")
        print(f"Motion File: {npz_path}")
        print(f"Target FPS: {self.target_fps} (Client expected DT={self.dt:.4f}s)")

        # Load motion sequence
        self.j3d_sequence, self.source_fps = self.load_and_process_amass(npz_path, smpl_model_path)
        self.num_total_frames = self.j3d_sequence.shape[0]

        # To Test: SYNC 
        self.start_time = time.time()
        
        print(f"*** Successfully Loaded {self.num_total_frames} frames at Source FPS: {self.source_fps}")
        print(f"*** Runs on wall-clock time (does not wait for client).")
        print(f"Streaming ready.. (Press Ctrl+C to stop)")

    # Reads the .npz file (joint angles) and converts it to 3D Positions (in XYZ)
    # using the SMPL body model (Forward Kinematics)
    def load_and_process_amass(self, path, model_path):
        file_ext = os.path.splitext(path)[1]
        if file_ext != '.npz':
            raise ValueError(f"Unsupported file: '{file_ext}'. Only .npz supported.")

        try:
            data = np.load(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find AMASS file at: {path}")

        #get source file fps
        source_fps = 60 #default 60 for not found
        if 'mocap_framerate' in data:
            source_fps = int(data['mocap_framerate'])
            print(f"----mocap_framerate in file: {source_fps}")
        # total number of frames
        N = data['poses'].shape[0]
        # first 72 parameters: 24 joints * 3 axis angles
        # Note: SMPL_Parser expects the full pose tensor (72), not split
        poses = torch.tensor(data['poses'][:, :72], dtype=torch.float32).to(DEVICE)
        trans = torch.tensor(data['trans'], dtype=torch.float32).to(DEVICE)
        # If the file has shape info (beta), use it
        if self.use_neutral_shape:
            #ignore beta, use mean shape to force the same limb length 
            betas = torch.zeros((N, 10), dtype=torch.float32).to(DEVICE)
            print("[Flag] Use neutral shape")
        elif 'betas' in data:
            betas = torch.tensor(data['betas'][:10], dtype=torch.float32).unsqueeze(0).to(DEVICE).repeat(N, 1)
            print("[Flag] Use provided shape")
        else:
            betas = torch.zeros((N, 10), dtype=torch.float32).to(DEVICE)

        try:
            parser = SMPL_Parser(model_path=model_path, gender="neutral").to(DEVICE)
        except Exception as e:
            print(f"\n[ERROR] Could not load SMPL model. Ensure 'SMPL_NEUTRAL.pkl' is in {model_path}")
            raise e
        
        print("Calculating 3D Joint Positions...")
        with torch.no_grad():
            batch_size = 512  # process in batch to avoid running out of GPU memory
            joints_list = []
            for i in range(0, N, batch_size):
                batch_poses = poses[i:i+batch_size]
                batch_betas = betas[i:i+batch_size]
                batch_trans = trans[i:i+batch_size]
                _, output_joints = parser.get_joints_verts(batch_poses, batch_betas, batch_trans)
                # to test: move results to CPU to free GPU memory
                joints_list.append(output_joints.cpu())

            joints = torch.cat(joints_list, dim=0)

        # -------- Format Data ------
        # select standard 24 joints
        joints_np = joints.numpy()[:, :24, :] 

        # make sure the humanoid is on 0 ground! 
        # logic: Find the lowest Z-value (height) and shift the entire sequence so the lowest point is slightly above floor (0,0,0).
        if self.auto_ground:
            min_z = np.min(joints_np[..., 2]) # Index 2: Z axis, based on joib
            floor_buffer = 0.05           # unit in meter, temp buffer for shoe/foot thickness
            offset = -min_z + floor_buffer
            joints_np[..., 2] += offset
            print(f"[Auto-Ground] Lowest point detected: {min_z:.4f}m, applying floor_buff: {floor_buffer}, Z-offset: {offset:.4f}m")
            
            # #To test center start:
            # start_x = joints_np[0,0,0] #X
            # start_y = joints_np[0,0,1] #Y
            # joints_np[...,0] -= start_x 
            # joints_np[...,0] -= start_y 
            # print("Test: Change x and y based on start pose" )
        return joints_np, source_fps

    def get_current_joints(self):
       
        # To Test: WALL CLOCK TIME SYNC
        # calculates which frame should play right now based on elapsed time
        
        elapsed_time = time.time() - self.start_time
        
        # Calculate index: (Time * FPS) % Total_Frames
        # This automatically handles frame skip (e.g. if time jumps 0.1s =  skip 6 frames at 60fps)
        temp_loop_count = int(int(elapsed_time * self.source_fps) / self.num_total_frames)
        current_idx = int(elapsed_time * self.source_fps) % self.num_total_frames
        
        # Get data
        joints_data = self.j3d_sequence[current_idx]

        if current_idx == 0 and elapsed_time > 1.0 and temp_loop_count>self.loop_print_count:
            print(f"--Looped, start over, elapsed_time={elapsed_time}")
            self.loop_print_count = temp_loop_count
            
            pass

        # (people, 24 joints, 3 coordinate)
        output_j3d = np.zeros((MAX_PEOPLE, NUM_JOINTS, 3))
        output_j3d[0] = joints_data
        
        return output_j3d

# called by HumanoidImMCPDemo.py every simulation step.
streamer = None
async def pose_getter(request):
    j3d = streamer.get_current_joints()
    print(f"-----streamer.dt={streamer.dt}")
    response_data = {
        "j3d": j3d.tolist(), 
        "dt": streamer.dt,
        "j3d_curr": j3d.tolist(),
        "j3d_curr_vel": np.zeros_like(j3d).tolist() 
    }
    return web.json_response(response_data)

async def websocket_handler(request):
    '''
    Endpoint: GET /ws
    '''
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            if msg.data == "get_pose":
                # Returns the frame corresponding to Wall Clock
                j3d = streamer.get_current_joints()
                
                response_data = {
                    "j3d": j3d.tolist(), 
                    "dt": streamer.dt,
                }

                await ws.send_json(response_data)
                
            elif msg.data == 'close cmd':
                await ws.close()
                break
                
        elif msg.type == web.WSMsgType.ERROR:
            print('ws connection closed with exception %s', ws.exception())

    print('websocket connection closed')
    return ws

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream AMASS motion data to PHC Demo")
    parser.add_argument("--file", type=str, required=True, help="Path to the AMASS .npz file")
    parser.add_argument("--smpl", type=str, default="data/smpl/", help="Path to folder containing SMPL_NEUTRAL.pkl")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--target_fps", type=int, default=60, help="Target FPS for the client (default: 30)")
    
    args = parser.parse_args()

    streamer = AmassStreamer(
        npz_path=args.file, 
        smpl_model_path=args.smpl, 
        target_fps=args.target_fps,
        auto_ground=True
    )

    app = web.Application()
    app.router.add_route('GET', '/get_pose', pose_getter)
    app.router.add_route('GET', '/ws', websocket_handler)
    
    print(f"Server running at http://0.0.0.0:{args.port}")
    web.run_app(app, port=args.port)