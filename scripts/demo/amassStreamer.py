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

MAX_PEOPLE = 5 
NUM_JOINTS = 24  # Standard SMPL joint count
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class AmassStreamer:
    def __init__(self, npz_path, smpl_model_path, auto_ground=True, use_neutral_shape=True):
        self.current_frame = 0
        self.auto_ground = auto_ground
        self.use_neutral_shape = use_neutral_shape
        self.frame_rate = 60 # Default, will be overridden by file
        self.dt = 1.0 / self.frame_rate
        
        print(f"--- Initializing AMASS Streamer (use neutral body shape = {use_neutral_shape})---")
        print(f"Motion File: {npz_path}")
        print(f"Device: {DEVICE}")

        # Load motion sequence
        self.j3d_sequence = self.load_and_process_amass(npz_path, smpl_model_path)
        self.num_frames = self.j3d_sequence.shape[0]
        
        print(f"***Successfully loaded {self.num_frames} frames.")
        print(f"Streaming started.. (Press Ctrl+C to stop)")

    # Reads the .npz file (joint angles) and converts it to 3D Positions (in XYZ)
    # using the SMPL body model (Forward Kinematics)
    def load_and_process_amass(self, path, model_path):
        
        file_ext = os.path.splitext(path)[1]
        if file_ext != '.npz':
            raise ValueError(f"Unsupported motion file type: '{file_ext}'. This streamer only supports .npz files.")

        try:
            data = np.load(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find AMASS file at: {path}")

        # Check for mocap framerate and update streamer's rate
        if 'mocap_framerate' in data:
            self.frame_rate = data['mocap_framerate']
            print(f"----mocap_framerate={self.frame_rate }")
        else:
            raise ValueError(f"'mocap_framerate' not found in the AMASS file: {path}")

        self.dt = 1.0 / self.frame_rate


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
            betas = torch.tensor(data['betas'][:10], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            betas = betas.repeat(N, 1)
            print("[Flag] Use provided shape")
        else:
            betas = torch.zeros((N, 10), dtype=torch.float32).to(DEVICE)

        # load SMPL Model, use as the mathematical function: f(angles, shape) -> 3D_Positions
        try:
            parser = SMPL_Parser(model_path=model_path, gender="neutral").to(DEVICE)
        except Exception as e:
            print(f"\n[ERROR] Could not load SMPL model from: {model_path}. Ensure folder contains 'SMPL_NEUTRAL.pkl")
            raise e
        
        # compute Forward Kinematics 
        print("Calculating 3D Joint Positions...")
        with torch.no_grad():
            batch_size = 512 # process in batch to avoid running out of GPU memory
            joints_list = []
            for i in range(0, N, batch_size):
                batch_poses = poses[i:i+batch_size]
                batch_betas = betas[i:i+batch_size]
                batch_trans = trans[i:i+batch_size]
                
                # returns: (vertices, joints)
                _, output_joints = parser.get_joints_verts(batch_poses, batch_betas, batch_trans)
                
                # to test: move results to CPU to free GPU memory
                joints_list.append(output_joints.cpu())

            # Combine batches
            joints = torch.cat(joints_list, dim=0)

        # -------- Format Data ------
        # select standard 24 joints
        joints_np = joints.numpy()[:, :24, :] 

        # make sure the humanoid is on 0 ground! 
        # logic: Find the lowest Z-value (height) and shift the entire sequence so the lowest point is slightly above floor (0,0,0).
        if self.auto_ground:
            min_z = np.min(joints_np[..., 2]) # Index 2: Z axis
            floor_buffer = 0.05            # unit in meter, temp buffer for shoe/foot. To be discussed
            offset = -min_z + floor_buffer
            joints_np[..., 2] += offset
            
            print(f"[Auto-Ground] Lowest point detected: {min_z:.4f}m, applying Z-offset: {offset:.4f}m")
            
        return joints_np

    def get_current_joints(self):
        
        # Get the frame data
        joints_data = self.j3d_sequence[self.current_frame]
        
        # Loop the video
        # hard reset: increment frame, reset to 0 if we hit the end
        self.current_frame = (self.current_frame + 1) % self.num_frames
        
        if self.current_frame == 0:
            print(f"--Looped, start over, total frame = {self.num_frames}")
        
        # (people, 24 joints, 3 coordinate)
        output_j3d = np.zeros((MAX_PEOPLE, NUM_JOINTS, 3))
        output_j3d[0] = joints_data
        
        return output_j3d


# called by HumanoidImMCPDemo.py every simulation step.
async def pose_getter(request):
    '''   
    Endpoint: GET /get_pose
    return: JSON contains the j3d positions for the current frame
    '''
    j3d = streamer.get_current_joints()
    
    print(f"-----streamer.dt={streamer.dt}")
    
    response_data = {
        "j3d": j3d.tolist(), 
        "dt": streamer.dt, #calcualted by streamer side, not from the amass file
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
    
    # keep connection open until client disconnects
    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            if msg.data == "get_pose":
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
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    
    args = parser.parse_args()

    # Initialize Streamer
    streamer = AmassStreamer(npz_path=args.file, smpl_model_path=args.smpl)

    # Initialize Web Server
    app = web.Application()
    app.router.add_route('GET', '/get_pose', pose_getter)
    app.router.add_route('GET', '/ws', websocket_handler)
    
    print(f"Server running at http://0.0.0.0:{args.port}")
    web.run_app(app, port=args.port)