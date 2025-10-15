import os
import sys
sys.path.append(os.getcwd())
import os.path as osp
import time
import json
import asyncio
import threading
import numpy as np
import joblib
import torch
from aiohttp import web
from collections import deque
from datetime import datetime

from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot

import threading, signal
stop_event = threading.Event()
bg_thread = None

AMASS_PKL = "phc/data/amass_test/amass_single_motion.pkl"   
HOST = "0.0.0.0"
PORT = 8080

# Globals expected by existing receivers in demo code
bbox, pose_mat, j3d, j2d, trans, dt, ws_talkers, reset_offset, offset_height, images_acc, recording, sim_talker, num_ppl, fps = (
    np.zeros([5, 4]), np.zeros([24, 3, 3]), np.zeros([5, 24, 3]), None, np.zeros([3]),
    1 / 10, [], True, 0.92, deque(maxlen=24000), False, None, 0, 0
)
superfast = True

SKELETON_XML_PATH = "phc/data/assets/mjcf_test/smpl_humanoid.xml"
_skeleton_tree = None


def _ensure_skeleton_tree():
    """
    Make sure the SMPL SkeletonTree used in the converter exists and matches layout.
    If the xml file isn't present yet, (re)generate it via SMPL_Robot with neutral betas.
    """
    global _skeleton_tree
    if _skeleton_tree is not None:
        return _skeleton_tree

    xml_dir = osp.dirname(SKELETON_XML_PATH)
    os.makedirs(xml_dir, exist_ok=True)

    if not osp.isfile(SKELETON_XML_PATH):
        robot_cfg = {
            "mesh": False, 
            "rel_joint_lm": True, 
            "upright_start": False,
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
        beta = np.zeros((16,), dtype=np.float32)
        smpl_local_robot.load_from_skeleton(
            betas=torch.from_numpy(beta[None, ...]),
            gender=[0],
            objs_info=None
        )
        smpl_local_robot.write_xml(SKELETON_XML_PATH)

    _skeleton_tree = SkeletonTree.from_mjcf(SKELETON_XML_PATH)
    return _skeleton_tree


# Load the single-motion pkl
def _load_motion(pkl_path: str):
    data = joblib.load(pkl_path)
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"Unexpected pkl format in {pkl_path}")
    key = list(data.keys())[0]
    motion = data[key]

    # Normalize types
    root_off = motion["root_trans_offset"]
    if hasattr(root_off, "numpy"):
        root_off = root_off.numpy()
    motion["root_trans_offset"] = np.asarray(root_off)

    # fps default to 30
    motion["fps"] = int(motion.get("fps", 30))
    return motion, key


# Build SkeletonState for a single frame and return the 24x3 global joint positions 
def _fk_to_joints_frame(skel: SkeletonTree, pose_quat_local_24x4: np.ndarray, root_trans_offset_3: np.ndarray):
  
    # torch inputs with batch dim 1
    pose_q = torch.from_numpy(pose_quat_local_24x4[None, ...])       # (1,24,4)
    root_t = torch.from_numpy(root_trans_offset_3[None, ...]).float()  # (1,3)

    state = SkeletonState.from_rotation_and_root_translation(
        skel, pose_q, root_t, is_local=True
    )
    return state.global_translation.numpy()[0]  # (1, 24, 3) -> (24,3)


def stream_amass_realtime(pkl_path: str):
    global j3d, dt, num_ppl, fps

    motion, key = _load_motion(pkl_path)
    skel = _ensure_skeleton_tree()

    pose_quat_local = np.asarray(motion["pose_quat"])           # (N,24,4)
    root_trans_off  = np.asarray(motion["root_trans_offset"])   # (N,3)
    fps = int(motion["fps"])
    N = pose_quat_local.shape[0]

    print(f"Loaded motion '{key}': {N} frames @ {fps} FPS")
    interval = 1.0 / max(1, fps)

    while not stop_event.is_set():
        t_loop = time.time()
        for i in range(N):
            
            if stop_event.is_set():
                break
            
            t0 = time.time()
            try:
                joints_24x3 = _fk_to_joints_frame(
                    skel,
                    pose_quat_local[i],
                    root_trans_off[i]
                )
            except Exception as e:
                # If FK fails for any reason, keep last frame
                print(f"WARN: FK failed at frame {i}: {e}")
                joints_24x3 = j3d[0]

            # Fill global 'j3d'
            j3d.fill(0.0)
            j3d[0, :, :] = joints_24x3
            num_ppl = 1
            dt = max(1e-6, time.time() - t0)

            # pace to target FPS
            elapsed = time.time() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                stop_event.wait(timeout=sleep_t)

        # Optional: loop seamlessly
        loop_elapsed = time.time() - t_loop
        print(f"---Completed one playback loop in {loop_elapsed:.2f}s — restarting.")


async def pose_getter(request):
    
    global j3d, dt
    return web.json_response({"j3d": j3d.tolist(), 
                              "dt": dt})


async def websocket_handler(request):

    global j3d, dt, fps
    print('Websocket connection starting')
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print('Websocket connection ready')

    try:
        while True:
            await ws.send_json({"j3d": j3d.tolist(), 
                                "dt": dt})
            await asyncio.sleep(1.0 / max(1, fps))
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Websocket Error: {e}")
    finally:
        print("Websocket connection closed")
    return ws


async def talk_websocket_handler(request):

    global reset_offset, recording, images_acc, j3d, dt
    print("Websocket TALK connection starting")
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print("Websocket TALK connection ready")

    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            text = msg.data.strip()
            print(f"Websocket TALK Received: {text}")
            if text.startswith("r"):
                reset_offset = True
            elif text.startswith("s"):
                recording = True
            elif text.startswith("e"):
                recording = False
            elif text.startswith("get_pose"):
                await ws.send_json({"j3d": j3d.tolist(), "dt": dt})
            await ws.send_str("Done!")
    print("Websocket TALK connection closed")
    return ws

#To preventing port being occupied when accidently stop the script
async def on_startup(app):
    global bg_thread
    bg_thread = threading.Thread(target=stream_amass_realtime,args=(AMASS_PKL,),daemon=True)
    bg_thread.start()

async def on_shutdown(app):
    stop_event.set()
    if bg_thread and bg_thread.is_alive():
        bg_thread.join(timeout=5)
        
    

def main():
    # Start realtime AMASS 
    threading.Thread(target=stream_amass_realtime, args=(AMASS_PKL,), daemon=True).start()

    # aiohttp app + routes
    app = web.Application(client_max_size=1024 ** 2)
    app.router.add_route('GET', '/ws', websocket_handler)
    app.router.add_route('GET', '/ws_talk', talk_websocket_handler)
    app.router.add_route('GET', '/get_pose', pose_getter)
    
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    print("==============================================================")
    print(" AMASS Pose Stream Server  started ")
    print(f" Using: {AMASS_PKL}")
    print(" Endpoints:")
    print(f"   ws://<{HOST}>:{PORT}/ws")
    print(f"   ws://<{HOST}>:{PORT}/ws_talk")
    print(f"   http://<{HOST}>:{PORT}/get_pose")
    print("==============================================================")

    try:
        web.run_app(app, host=HOST, port=PORT)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
