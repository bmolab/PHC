import numpy as np
import time
import argparse
import torch
import sys
import os
from aiohttp import web
from scipy.spatial.transform import Rotation as sRot

# Try to import SMPL Parser
try:
    from smpl_sim.smpllib.smpl_parser import SMPL_Parser
except ImportError:
    print("Error: Could not import SMPL_Parser. Ensure 'smpl_sim' is in your PYTHONPATH.")
    sys.exit(1)

MAX_PEOPLE = 1
NUM_JOINTS = 24
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class AmassStreamer:
    def __init__(self, npz_path, smpl_model_path, target_fps=30, scale=1.0, rotate_z=-90.0, use_neutral_shape=True):
        self.target_fps = target_fps
        self.dt = 1.0 / self.target_fps 
        self.scale = scale
        self.rotate_z = rotate_z
        self.use_neutral_shape = use_neutral_shape
        
        self.last_loop_time = time.perf_counter()
        self.frame_cursor = 0.0 

        print(f"--- Initializing AMASS Streamer---")
        print(f"File: {npz_path}")
        print(f"FPS: {self.target_fps} | dt: {self.dt:.4f}s")
        print(f"Scale: {self.scale} | Rot: {self.rotate_z}")

        # Load movement
        self.j3d_sequence, self.source_fps = self.load_and_process_amass(npz_path, smpl_model_path)
        self.num_total_frames = self.j3d_sequence.shape[0]
        
        # Frame Step Calculation
        self.frame_step = self.source_fps / self.target_fps
        
        print(f"*** Frame Step: {self.frame_step:.2f}")
        print(f"*** Streaming...")

    def load_and_process_amass(self, path, model_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        data = np.load(path)
        
        if 'mocap_framerate' in data:
            source_fps = int(data['mocap_framerate'])
        else:
            source_fps = 30
            
        N = data['poses'].shape[0]
        poses = torch.tensor(data['poses'][:, :72], dtype=torch.float32).to(DEVICE)
        trans = torch.tensor(data['trans'], dtype=torch.float32).to(DEVICE)

        if self.use_neutral_shape:
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


        #to test: negative margin fix the the "hovering feet" issue 
        floor_height = np.percentile(joints_np[..., 2], 1.0)

        ground_margin = -0.02 
        
        offset_z = -floor_height + ground_margin 
        
        print(f"---- estimated floor at {floor_height:.4f}m. ground_margin by {ground_margin}m. Total Offset: {offset_z:.4f}m")
        joints_np[..., 2] += offset_z

        return joints_np, source_fps

    def get_current_pose(self):
        target_interval = self.dt
        now = time.perf_counter()
        elapsed = now - self.last_loop_time
        remaining = target_interval - elapsed
        
        if remaining > 0:
            if remaining > 0.002:
                time.sleep(remaining - 0.002)
            while (time.perf_counter() - self.last_loop_time) < target_interval:
                pass # Spin lock
        
        self.last_loop_time = time.perf_counter()

        # Interpolate
        idx_0 = int(self.frame_cursor)
        idx_1 = idx_0 + 1
        alpha = self.frame_cursor - idx_0
        
        idx_0 = idx_0 % self.num_total_frames
        idx_1 = idx_1 % self.num_total_frames
        
        pose_0 = self.j3d_sequence[idx_0]
        pose_1 = self.j3d_sequence[idx_1]
        
        pose_interpolated = pose_0 * (1 - alpha) + pose_1 * alpha
        
        self.frame_cursor += self.frame_step
        if self.frame_cursor >= self.num_total_frames:
            self.frame_cursor = 0.0 
        
        output_j3d = np.zeros((MAX_PEOPLE, NUM_JOINTS, 3))
        output_j3d[0] = pose_interpolated
        
        # RETURN CONSTANT DT (Fixes Jumping)
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
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor")
    parser.add_argument("--rot", type=float, default=-90.0, help="Rotation around Z")
    
    args = parser.parse_args()

    streamer = AmassStreamer(
        npz_path=args.file, 
        smpl_model_path=args.smpl, 
        target_fps=args.fps, 
        scale=args.scale,
        rotate_z=args.rot,
        use_neutral_shape = False
    )

    app = web.Application()
    app.router.add_route('GET', '/get_pose', pose_getter)
    app.router.add_route('GET', '/ws', websocket_handler)
    
    print(f"Server running at http://0.0.0.0:{args.port}")
    web.run_app(app, port=args.port)