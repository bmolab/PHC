import numpy as np
import argparse
import torch
import sys
import os
import matplotlib.pyplot as plt

# Try to import SMPL Parser (Matches your environment)
try:
    from smpl_sim.smpllib.smpl_parser import SMPL_Parser
except ImportError:
    print("Error: Could not import SMPL_Parser. Ensure 'smpl_sim' is in your PYTHONPATH.")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def analyze_file(npz_path, smpl_path):
    print(f"--- Analyzing File: {npz_path} ---")
    
    if not os.path.exists(npz_path):
        print(f"Error: File not found {npz_path}")
        return

    data = np.load(npz_path)
    N = data['poses'].shape[0]
    framerate = int(data['mocap_framerate']) if 'mocap_framerate' in data else 'Unknown'
    
    print(f"Frames: {N} | Framerate: {framerate}")

    # 1. Load Data
    poses = torch.tensor(data['poses'][:, :72], dtype=torch.float32).to(DEVICE)
    trans = torch.tensor(data['trans'], dtype=torch.float32).to(DEVICE)
    betas = torch.zeros((N, 10), dtype=torch.float32).to(DEVICE) # Neutral shape

    # 2. Setup SMPL
    try:
        parser = SMPL_Parser(model_path=smpl_path, gender="neutral").to(DEVICE)
    except Exception as e:
        print(f"Error loading SMPL: {e}")
        return

    # 3. Run Forward Kinematics (Get 3D Joints)
    print("Calculating Joint Positions...")
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

    # Convert to Numpy (N, 24, 3)
    joints_np = joints.numpy()[:, :24, :]

    # ---------------------------------------------------
    # 4. Analyze Z-Height (Floor)
    # ---------------------------------------------------
    
    # "Min Z per frame" = The lowest point of the character at that specific moment
    min_z_per_frame = np.min(joints_np[..., 2], axis=1)
    
    global_min = np.min(min_z_per_frame)
    global_max_of_min = np.max(min_z_per_frame)
    avg_floor = np.mean(min_z_per_frame)
    
    # Percentile 1% (Ignores glitches)
    robust_min = np.percentile(min_z_per_frame, 1.0)

    print("\n=== ANALYSIS REPORT ===")
    print(f"Global Lowest Point (Raw Min): {global_min:.4f} m")
    print(f"Robust Lowest Point (1% Min):  {robust_min:.4f} m  <-- Use this for grounding")
    print(f"Highest 'Low Point': {global_max_of_min:.4f} m")
    print(f"Variation (Max - Min):         {global_max_of_min - global_min:.4f} m")
    

    # ---------------------------------------------------
    # 5. Plotting
    # ---------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(min_z_per_frame, label="Lowest Point (Feet) Z")
    plt.axhline(y=global_min, color='r', linestyle='--', label=f"Min: {global_min:.3f}")
    plt.axhline(y=robust_min, color='g', linestyle='-', label=f"Robust Min: {robust_min:.3f}")
    
    plt.title(f"Floor Contact Variation: {os.path.basename(npz_path)}")
    plt.xlabel("Frame")
    plt.ylabel("Height (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_img = "floor_analysis.png"
    plt.savefig(output_img)
    print(f"\nGraph saved to: {output_img}")
    # plt.show() # Uncomment if running locally with display

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to AMASS .npz file")
    parser.add_argument("--smpl", type=str, default="data/smpl/", help="Path to SMPL folder")
    args = parser.parse_args()

    analyze_file(args.file, args.smpl)
    