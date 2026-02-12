import os

import numpy as np
import time
import argparse
import torch
import sys
from aiohttp import web
from scipy.spatial.transform import Rotation as sRot

try:
    from smpl_sim.smpllib.smpl_parser import SMPL_Parser
except ImportError as e:
    print(f"Error: Could not import SMPL_Parser. {e}")
    sys.exit(1)

MAX_PEOPLE = 1
NUM_JOINTS = 24
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class AmassStreamer:
    def __init__(self, npz_path, smpl_model_path, target_fps=30, scale=0.85, rotate_z=-90.0):
        """
        Initialize the streamer.
        
        Args:
            npz_path: Path to the raw AMASS .npz file.
            smpl_model_path: Path to the SMPL body model files.
            target_fps: fps PHC side uses 
            scale: resizing factor
            rotate_z: Rotation adjustment to align with simulation environments in PHC Isaac Gym side (Z-up).
        """
        
        self.target_fps = target_fps
        self.dt = 1.0 / self.target_fps 
        self.scale = scale
        self.rotate_z = rotate_z
        
        self.last_loop_time = time.time()
        self.frame_cursor = 0.0 

        print(f"--- Initializing AMASS Streamer ---")
        print(f"File: {npz_path}")
        print(f"Target FPS: {self.target_fps}, dt: {self.dt:.4f}s")
        print(f"Scale: {self.scale:.2f} | Rotation: {self.rotate_z:.1f} deg (Z-axis)")

        # Load and Process
        self.j3d_sequence, self.source_fps = self.load_and_process_amass(npz_path, smpl_model_path)
        self.num_total_frames = self.j3d_sequence.shape[0]
        
        # Calculate how many source frames to skip to match target FPS
        # e.g. if source is 60fps and target is 30fps, step is 2.0
        self.frame_step = self.source_fps / self.target_fps
        
        print(f"*** Source: {self.source_fps} FPS | Skip Step: {self.frame_step:.2f} frames")
        print(f"*** Ready. Sending...")

    def load_and_process_amass(self, path, model_path):  
        """
        The core data pipeline:
        1. Load raw .npz (pose parameters).
        2. Run SMPL Forward Kinematics to get 3D joint positions.
        3. Rotate and Scale joints to match the Simulation environment (Isaac Gym is Z-up).
        4. Adjust height (grounding) so the character stands on the floor.
        """
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        data = np.load(path)
        
        if 'mocap_framerate' in data:
            source_fps = int(data['mocap_framerate'])
        else:
            source_fps = 60
            
        N = data['poses'].shape[0]
        
        # Load Data
        poses = torch.tensor(data['poses'][:, :72], dtype=torch.float32).to(DEVICE)
        trans = torch.tensor(data['trans'], dtype=torch.float32).to(DEVICE)
        betas = torch.zeros((N, 10), dtype=torch.float32).to(DEVICE) 

        # SMPL FK
        try:
            parser = SMPL_Parser(model_path=model_path, gender="neutral").to(DEVICE)
        except Exception as e:
            print(f"[Error] Failed to load SMPL model: {e}")
            raise e
        
        print("Running SMPL Forward Kinematics...")
        with torch.no_grad():
            batch_size = 256
            joints_list = []
            for i in range(0, N, batch_size):
                batch_poses = poses[i:i+batch_size]
                batch_betas = betas[i:i+batch_size]
                batch_trans = trans[i:i+batch_size]
                _, output_joints = parser.get_joints_verts(batch_poses, batch_betas, batch_trans)
                joints_list.append(output_joints.cpu())
            joints = torch.cat(joints_list, dim=0)

        # Process Joints
        joints_np = joints.numpy()[:, :24, :]

        # Rotation
        if self.rotate_z != 0:
            rot_mat = sRot.from_euler('z', self.rotate_z, degrees=True).as_matrix()
            shape = joints_np.shape
            flat_joints = joints_np.reshape(-1, 3)
            flat_joints = np.dot(flat_joints, rot_mat.T)
            joints_np = flat_joints.reshape(shape)

        # Scaling
        if self.scale != 1.0:
            joints_np *= self.scale

        # 6. GROUNDING FIX: Ground based on Frame 0 only
        # Old (Bad): min_z = np.min(joints_np[..., 2]) -> Used lowest point in whole video
        # New (Good): Use lowest point in the *first frame* only.
        
        start_pose = joints_np[0]
        min_z_start = np.min(start_pose[..., 2])
        
        ground_margin = 0.05 
        offset_z = -min_z_start + ground_margin
        
        print(f"Grounding Offset (Frame 0): {offset_z:.4f}m")
        joints_np[..., 2] += offset_z

        return joints_np, source_fps

    def get_current_pose(self):
        """
        Retrieves the pose for the current timestep, handling looping and throttling.
        """
        
        #Sleep if we are being called faster than target_fps
        frame_interval = self.dt
        now = time.time()
        elapsed = now - self.last_loop_time
        
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        
        self.last_loop_time = time.time()

        #Calculate which frame to show
        idx = int(self.frame_cursor) % self.num_total_frames
        pose = self.j3d_sequence[idx]
        
        self.frame_cursor += self.frame_step
        if self.frame_cursor >= self.num_total_frames:
            self.frame_cursor = 0 
        
        #Format output
        output_j3d = np.zeros((MAX_PEOPLE, NUM_JOINTS, 3))
        output_j3d[0] = pose
        
        return output_j3d, self.dt

streamer = None

async def pose_getter(request):
    j3d, dt = streamer.get_current_pose()
    return web.json_response({"j3d": j3d.tolist(), "dt": dt})

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            if msg.data == "get_pose":
                j3d, dt = streamer.get_current_pose()
                await ws.send_json({"j3d": j3d.tolist(), "dt": dt})
            elif msg.data == 'close cmd':
                await ws.close()
                break
    return ws

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to AMASS .npz file")
    parser.add_argument("--smpl", type=str, default="data/smpl/", help="Path to SMPL folder")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS")
    parser.add_argument("--scale", type=float, default=0.85, help="Scale factor")
    parser.add_argument("--rot", type=float, default=-90.0, help="Rotation around Z")
    args = parser.parse_args()

    streamer = AmassStreamer(
        npz_path=args.file, 
        smpl_model_path=args.smpl, 
        target_fps=args.fps, 
        scale=args.scale,
        rotate_z=args.rot
    )

    app = web.Application()
    app.router.add_route('GET', '/get_pose', pose_getter)
    app.router.add_route('GET', '/ws', websocket_handler)
    
    print(f"Server running at http://0.0.0.0:{args.port}")
    web.run_app(app, port=args.port)