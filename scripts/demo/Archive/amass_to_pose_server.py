import sys,pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
import os
import time
import asyncio
import threading
import numpy as np
import torch
from aiohttp import web
from scipy.spatial.transform import Rotation as sRot
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
import argparse

'''
!!Deprecated because the mismatch in converted XML skeleton calcualtion
'''
'''
Streams 3D pose data from an AMASS .npz file in real time using forward kinematics on an SMPL skeleton. 
Runs web server to provide live joint positions.
'''

# required global variables
j3d = np.zeros([5, 24, 3],dtype=np.float32) 
num_ppl = 0
stop_event = threading.Event()
XML_PATH_FOLDER = 'phc/data/assets/mjcf_test'
PHC_FPS = 30
dt = 1.0/PHC_FPS
fps = PHC_FPS

#mutex lock ensures only one thread accesses the global j3d at a time
th_lock = threading.Lock()

#make sure a SMPL mjcf skeleton XML exists (required by PHC), create it if not
def ensure_skeleton_tree(xml_path: str):

    os.makedirs(os.path.dirname(xml_path), exist_ok=True)

    if not os.path.isfile(xml_path):
        robot_cfg = {
            "mesh": False,
            "rel_joint_lm": True,
            "upright_start": True,
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
        smpl_robot = LocalRobot(robot_cfg)
        smpl_robot.load_from_skeleton(
            betas=torch.zeros((1, 16)), gender=[0], objs_info=None
        )
        smpl_robot.write_xml(xml_path)

    return SkeletonTree.from_mjcf(xml_path)

#Forward kinematics
#given local joint rotations and skeleton hierarchy, computes global 3D positions of all joints
def fk_to_joints(skel: SkeletonTree, pose_quat_local: np.ndarray, root_trans: np.ndarray):
    '''
        skel: joint hierarchy
        pose_quat_local: local joint rotations, one quaternion per jointt(how each bone is rotated relative to its parent joint), xyzw order
        root_trans: Root translation in world coordinate (meters, z-up)
    '''
    pose_q = torch.from_numpy(pose_quat_local[None, ...].astype(np.float32))
    root_t = torch.from_numpy(root_trans[None, ...].astype(np.float32))
    state = SkeletonState.from_rotation_and_root_translation(skel, pose_q, root_t, is_local=True) #expects local joint rotations + root translation
    return state.global_translation.numpy()[0].astype(np.float32)  # (24, 3)

#amass files have different frame rates, Isaac expect 30hz
#generate frame idx to resample motion sequence from its original framerate to a target framerate
def resample_idx(n: int, src_fps: float, dst_fps: float):
    '''
    n:  num of frames in the source sequence
    src_fps: original framerate
    dst_fps: target framerate
    '''
    
    if n <= 0:
        return np.zeros(0, dtype=int)
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError("source FPS must be positive")
    
    idx = []               
    acc = 0.0
    step_size = src_fps / dst_fps   # num of source frames per target frame
    i = 0
    
    while i < n:
        idx.append(i)
        acc += step_size #accumulated time index by step size
        i = int(round(acc))
        
    return np.clip(np.asarray(idx, dtype=int), 0, n-1)

#Stream a one AMASS .npz motion file 
def stream_amass_realtime(npz_path: str):

    global j3d, dt, num_ppl, fps
    
    print(f"{'-'*3}Loading AMASS data from: {npz_path}")
    data = dict(np.load(npz_path, allow_pickle=True))
    
    pose_dim = data["poses"].shape[1]
    if pose_dim < 72:
        # pad missing joints with 0 (3 axis-angle per joint)
        missing = 72 - pose_dim
        pose_aa = np.concatenate(
            [data["poses"][:, :pose_dim],
            np.zeros((data["poses"].shape[0], missing), dtype=data["poses"].dtype)],
            axis=-1
        )
    else:
        pose_aa = data["poses"][:, :72] #(SMPL, 24 joints ×3)

    trans = data["trans"]             # (T,3) root translation
    src_fps  = int(data.get("mocap_framerate", 60)) #amass default 60
    target_fps = PHC_FPS
    
    # resample to PHC required fps
    T_src = pose_aa.shape[0]
    T_align = min(T_src, trans.shape[0]) 
    pose_aa = pose_aa[:T_align]
    trans   = trans[:T_align]
    idx = resample_idx(T_align, float(src_fps), float(target_fps))
    pose_aa = pose_aa[idx]
    trans   = trans[idx]
    N = pose_aa.shape[0]

    #convert to quaternion
    #to-test: by default as_quat use (x, y, z, w) order 
    pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(N, 24, 4).astype(np.float32)

    #skeleton XML
    file_stem = os.path.splitext(os.path.basename(npz_path))[0]
    xml_path = os.path.join(XML_PATH_FOLDER, f"{file_stem}_humanoid.xml")
    skel = ensure_skeleton_tree(xml_path)

    print(f"{'-'*3}Streaming {npz_path} ({N} frames @{target_fps}Hz)")
    print(f"{'-'*5}Using skeleton XML: {xml_path}")
    interval = 1.0/target_fps
    
    fps = target_fps
    dt  = interval

    while not stop_event.is_set():
        for i in range(N):
            t0 = time.time()
            try:
                joints_24x3 = fk_to_joints(skel, pose_quat[i], trans[i])
            except Exception as e:
                print(f"Forward kinematics failed at frame {i}: {e}")
                continue
            
            with th_lock:
                j3d.fill(0.0)
                j3d[0, :, :] = joints_24x3
                num_ppl = 1

            elapsed = time.time() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                stop_event.wait(timeout=sleep_t)

        print(f"{'*'*3}AMASS Motion looped")


async def pose_getter(request):
    global j3d, dt, num_ppl
    with th_lock:
        payload = {"j3d": j3d.tolist(), 
                   "dt": dt, 
                   "num_ppl": num_ppl}
    return web.json_response(payload)

async def websocket_handler(request):
    global j3d, dt, fps, num_ppl
    
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    try:
        while True:
            with th_lock:
                payload = {"j3d": j3d.tolist(), 
                           "dt": dt, 
                           "num_ppl": num_ppl}
            await ws.send_json(payload)
            await asyncio.sleep(1.0 / max(1, fps))
    except asyncio.CancelledError:
        pass
    finally:
        print("----WebSocket closed.")
    return ws

async def talk_websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT and msg.data.startswith("get_pose"):
            with th_lock:
                payload = {"j3d": j3d.tolist(), 
                           "dt": dt, 
                           "num_ppl": num_ppl}
                await ws.send_json(payload)
        await ws.send_str("Done!")
    return ws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, required=True,
                        help="Path to an AMASS .npz file")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    threading.Thread(target=stream_amass_realtime, args=(args.npz_path,), daemon=True).start()

    app = web.Application(client_max_size=1024**2)
    app.router.add_route("GET", "/get_pose", pose_getter)
    app.router.add_route("GET", "/ws", websocket_handler)
    app.router.add_route("GET", "/ws_talk", talk_websocket_handler)

    print("="*20)
    print("AMASS Pose Stream Server started ")
    print(f"--Using: {args.npz_path}")
    print(f"--Endpoints: ws://0.0.0.0:{args.port}/ws, http://0.0.0.0:{args.port}/get_pose")
    print("="*20)
    web.run_app(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
