from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY
from pythonosc.osc_message_builder import OscMessageBuilder
import numpy as np
import threading, time

class TorqueForceSender:
    def __init__(self, host="127.0.0.1", port=9000, fps=30, joint_names=None, sensor_names=None):
        
        self.client = SimpleUDPClient(host, port) #UDP socket for OSC
        
        self.fps = fps # desired send rate
        self.dt = 1.0/fps
        self.seq = 0 #frame counter
        self.joint_names = joint_names or []
        self.sensor_names = sensor_names or []
        
        self._stop = threading.Event()
        self._thread = None

    #to test: send  Send static information (joint & sensor names, units, etc.) to dpg
    def send_joint_order_once(self):
        
        if self.joint_names:
            msg = OscMessageBuilder(address="/rig/joint_order")
            for name in self.joint_names: msg.add_arg(name)
            self.client.send(msg.build())
            
        if self.sensor_names:
            msg = OscMessageBuilder(address="/rig/sensor_order")
            for name in self.sensor_names: msg.add_arg(name)
            self.client.send(msg.build())
        
        #to test: vetify dpg side's requirement on coord system
        meta2 = OscMessageBuilder(address="/meta2")
        for s in ["Z-up","meters","N","N·m","SMPL local quats XYZW"]:
            meta2.add_arg(s)
        self.client.send(meta2.build())

    #Start thread: sends OSC packets continuously
    def start(self, get_torques, get_wrenches=None):
        self.send_joint_order_once()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(get_torques, get_wrenches), daemon=True
        )
        self._thread.start()

    #thread exit 
    def stop(self,timeout_temp=1.0):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout_temp)

    def _run(self, get_torques, get_wrenches):
        t0 = time.time()
        while not self._stop.is_set():
            tic = time.time()
                        
            torques = get_torques() if get_torques else None # np.float32 [num_dof] OR [num_envs,num_dof]
            wrenches = get_wrenches() if get_wrenches else None # np.float32 [num_sensors*6] or None
            if torques is None:
                #skip this frame if data not ready
                time.sleep(self.dt)
                continue

            #pick env0 if a batch is provided
            if hasattr(torques, "ndim") and torques.ndim == 2:
                torques = torques[0]
            if wrenches is not None and hasattr(wrenches, "ndim") and wrenches.ndim == 2:
                wrenches = wrenches.reshape(-1)

            #flatten to 1-D float32 arrays
            torques = np.nan_to_num(torques.astype(np.float32)) 
            if wrenches is not None:
                wrenches = np.nan_to_num(wrenches.astype(np.float32))

            # build one bundle per frame
            bundle = OscBundleBuilder(IMMEDIATELY)

            # /meta: [seq, elapsed_time_s, fps]
            msg_meta = OscMessageBuilder(address="/meta")
            msg_meta.add_arg(int(self.seq))
            msg_meta.add_arg(float(time.time() - t0))   # to test: sim/stream time in sec
            msg_meta.add_arg(int(self.fps))
            bundle.add_content(msg_meta.build())

            # /rig/torques: flat list of all joint torques
            msg_tau = OscMessageBuilder(address="/rig/torques")
            for v in torques.flatten():
                msg_tau.add_arg(v)
            bundle.add_content(msg_tau.build())
    
            # /rig/wrenches message: 6D forces/torques
            if wrenches is not None:
                msg_w = OscMessageBuilder(address="/rig/wrenches")
                for v in wrenches.flatten():
                    msg_w.add_arg(v)
                bundle.add_content(msg_w.build())
            
            #Send bundle as one UDP datagram
            try:
                self.client.send(bundle.build())
                self.seq += 1
            except OSError as e:
                print(f"OSC send failed: {e}")
                pass

            # send at fixed fps
            sleep_t = self.dt - (time.time() - tic)
            if sleep_t > 0: 
                time.sleep(sleep_t)
