import os
import sys
sys.path.append(os.getcwd())
import os.path as osp
import argparse
import numpy as np
import joblib
import torch
from scipy.spatial.transform import Rotation as sRot

from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot

SKELETON_XML_PATH = "phc/data/assets/mjcf_test/smpl_humanoid.xml"

def convert_one_npz(npz_path: str, out_pkl: str, upright_start: bool = False):
    if not osp.isfile(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    entry = dict(np.load(open(npz_path, "rb"), allow_pickle=True))
    if "mocap_framerate" not in entry:
        raise ValueError(f"'mocap_framerate' missing in {npz_path}")

    #get framerate & downsample 
    framerate = float(entry["mocap_framerate"])  
    skip = max(1, int(round(framerate/30))) 
    root_trans = entry["trans"][::skip, :] #(N, 3)
    
    # Pack 66 AA dofs (body) + 6 zeros
    pose_aa = np.concatenate([entry["poses"][::skip, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)  # (N,72)
    N = pose_aa.shape[0]
    if N < 2:
        raise ValueError(f"Too few frames after downsampling: {N}")

    # Map SMPL joints to MuJoCo order then convert to quat
    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
    pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]  # (N,24,3)
    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)  # (N,24,4)

    # Neutral body
    beta = np.zeros((16,), dtype=np.float32)
    gender_number = [0]  # 0=neutral

    # Build SMPL robot
    robot_cfg = {
        "mesh": False, 
        "rel_joint_lm": True, 
        "upright_start": upright_start,
        "remove_toe": False, 
        "real_weight": True, 
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True, 
        "replace_feet": True, 
        "masterfoot": False,
        "big_ankle": True, 
        "freeze_hand": False, 
        "box_body": False, 
        "master_range": 50,
        "body_params": {}, 
        "joint_params": {}, 
        "geom_params": {}, 
        "actuator_params": {},
        "model": "smpl",
    }
    smpl_local_robot = LocalRobot(robot_cfg)
    smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None, ...]), gender=gender_number, objs_info=None)

    # skeleton file
    tmp_xml = SKELETON_XML_PATH
    os.makedirs(osp.dirname(tmp_xml), exist_ok=True)
    smpl_local_robot.write_xml(tmp_xml)
    skeleton_tree = SkeletonTree.from_mjcf(tmp_xml)

    root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

    # Build SkeletonState to get both local & global rotations
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree, 
        torch.from_numpy(pose_quat), 
        root_trans_offset, 
        is_local=True
    )

    if upright_start:
        pose_quat_global = (
            sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy())
            * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
        ).as_quat().reshape(N, -1, 4)
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False
        )

    pose_quat_global = new_sk_state.global_rotation.numpy()  # (N,24,4)
    pose_quat_local  = new_sk_state.local_rotation.numpy()   # (N,24,4)
    fps = 30  # after downsample

    #  Result format
    out = {
        "pose_quat_global": pose_quat_global,
        "pose_quat":        pose_quat_local,
        "trans_orig":       root_trans,                 
        "root_trans_offset": root_trans_offset,        
        "beta":             beta,
        "gender":           "neutral",
        "pose_aa":          pose_aa,
        "fps":              fps,
    }

    # Key name: prefix with "0-" and use basename
    base = osp.basename(npz_path).replace(".npz", "")
    parent = osp.basename(osp.dirname(npz_path))
    key_name = f"0-{parent}_{base}"

    joblib.dump({key_name: out}, out_pkl, compress=True)
    print(f"Wrote {os.path.abspath(out_pkl)} with key '{key_name}' ({N} frames @ {fps} FPS)")
    

def main():
    parser = argparse.ArgumentParser(description="Convert a single AMASS .npz to SMPL-based motion pickle.")
    parser.add_argument("--npz", required=True, help="Path to a single AMASS .npz file")
    parser.add_argument("--out", default="phc/data/amass_test/amass_single_motion.pkl", help="Output .pkl path")
    parser.add_argument("--upright", action="store_true", help="Apply upright_start correction")
    args = parser.parse_args()

    convert_one_npz(args.npz, args.out, upright_start=args.upright)
 
if __name__ == "__main__":
    main()
