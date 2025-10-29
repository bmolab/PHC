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
Streams 3D pose data from an AMASS .npz file in real time using forward kinematics on an SMPL skeleton. 
Runs web server to provide live joint positions.
'''

# required global variables
j3d = np.zeros([5, 24, 3]) 
dt = 1/30.0 # frame delta time
num_ppl = 0
fps = 30 # current frame rate
stop_event = threading.Event()
XML_PATH_FOLDER = 'phc/data/assets/mjcf_test'

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
        pose_quat_local: local joint rotations, one quaternion per jointt(how each bone is rotated relative to its parent joint)
        root_trans: Root translation in world coordinate
    '''
    pose_q = torch.from_numpy(pose_quat_local[None, ...])
    root_t = torch.from_numpy(root_trans[None, ...]).float()
    state = SkeletonState.from_rotation_and_root_translation(skel, pose_q, root_t, is_local=True)
    return state.global_translation.numpy()[0]  # (24, 3)

#Stream a one AMASS .npz motion file 
def stream_amass_realtime(npz_path: str):

    global j3d, dt, num_ppl, fps

    data = dict(np.load(npz_path, allow_pickle=True))
    
    pose_dim = data["poses"].shape[1]
    if pose_dim < 72:
        # pad missing joints with 0 (3 axis-angle per joint)
        missing = 72 - pose_dim
        pose_aa = np.concatenate(
            [data["poses"][:, :pose_dim],
            np.zeros((data["poses"].shape[0], missing))],
            axis=-1
        )
    else:
        pose_aa = data["poses"][:, :72]

    trans = data["trans"]             # (T,3)
    fps = int(data.get("mocap_framerate", 30))

    N = pose_aa.shape[0]
    skip = int(fps / 30)
    pose_aa = pose_aa[::skip]
    trans = trans[::skip]
    N = pose_aa.shape[0]

    #convert to quaternion
    pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

    #skeleton XML
    file_stem = os.path.splitext(os.path.basename(npz_path))[0]
    xml_path = os.path.join(XML_PATH_FOLDER, f"{file_stem}_humanoid.xml")
    skel = ensure_skeleton_tree(xml_path)

    print(f"----Streaming {npz_path} ({N} frames @30Hz)")
    print(f"-------Using skeleton XML: {xml_path}")
    interval = 1.0 / 30.0

    while not stop_event.is_set():
        for i in range(N):
            t0 = time.time()
            try:
                joints_24x3 = fk_to_joints(skel, pose_quat[i], trans[i])
            except Exception as e:
                print(f"Forward kinematics failed at frame {i}: {e}")
                continue

            j3d.fill(0.0)
            j3d[0, :, :] = joints_24x3
            num_ppl = 1
            dt = max(1e-6, time.time() - t0)

            elapsed = time.time() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                stop_event.wait(timeout=sleep_t)

        print("****AMASS Motion looped")


async def pose_getter(request):
    global j3d, dt
    return web.json_response(
        {"j3d": j3d.tolist(), 
         "dt": dt}
        )

async def websocket_handler(request):
    global j3d, dt, fps
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    try:
        while True:
            await ws.send_json({"j3d": j3d.tolist(), "dt": dt})
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
            await ws.send_json({"j3d": j3d.tolist(), "dt": dt})
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
