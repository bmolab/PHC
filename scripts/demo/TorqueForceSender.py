from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY
from pythonosc.osc_message_builder import OscMessageBuilder
import numpy as np
import threading
import time

class TorqueForceSender:
    def __init__(self, host="127.0.0.1", port=9000, fps=60, joint_names=None, sensor_names=None):
        self.client = SimpleUDPClient(host, port)
        self.target_dt = 1.0 / fps
        self.fps = fps
        self.seq = 0
        self.joint_names = joint_names or []
        self.sensor_names = sensor_names or []
        
        self._stop_event = threading.Event()
        self._thread = None
        
        # Buffers
        self.latest_torques = None
        self.latest_wrenches = None
        self.lock = threading.Lock()
        self.start_time = time.time()

    def update_data(self, torques, wrenches=None):
        """Call this from the main simulation loop"""
        with self.lock:
            # Ensure data is on CPU and Numpy before storing
            if hasattr(torques, 'cpu'): torques = torques.detach().cpu().numpy()
            if wrenches is not None and hasattr(wrenches, 'cpu'): wrenches = wrenches.detach().cpu().numpy()
            
            self.latest_torques = torques
            self.latest_wrenches = wrenches

    def start(self):
        self._send_metadata()
        self._stop_event.clear()
        self.start_time = time.time()  # Reset start time
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[TorqueForceSender] Streaming to {self.client._address}:{self.client._port}...")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _send_metadata(self):
        if self.joint_names:
            msg = OscMessageBuilder(address="/rig/joint_order")
            for name in self.joint_names: msg.add_arg(name)
            self.client.send(msg.build())
            
        if self.sensor_names:
            msg = OscMessageBuilder(address="/rig/sensor_order")
            for name in self.sensor_names: msg.add_arg(name)
            self.client.send(msg.build())

    def _run(self):
        while not self._stop_event.is_set():
            loop_start = time.time()
            
            # 1. Get latest data
            with self.lock:
                torques = self.latest_torques
                wrenches = self.latest_wrenches

            if torques is None:
                time.sleep(0.005)
                continue

            # 2. Build Bundle
            bundle = OscBundleBuilder(IMMEDIATELY)

            # --- FIX IS HERE ---
            # Header info: Send exactly 3 arguments to match the receiver
            msg_meta = OscMessageBuilder(address="/meta")
            msg_meta.add_arg(self.seq)
            msg_meta.add_arg(float(time.time() - self.start_time)) # Elapsed Time
            msg_meta.add_arg(int(self.fps))                        # FPS
            bundle.add_content(msg_meta.build())
            # -------------------

            # Send Torques
            flat_torques = np.nan_to_num(torques.astype(np.float32)).flatten()
            msg_tau = OscMessageBuilder(address="/rig/torques")
            for v in flat_torques: msg_tau.add_arg(float(v))
            bundle.add_content(msg_tau.build())

            # Send Contact Forces
            if wrenches is not None:
                flat_wrenches = np.nan_to_num(wrenches.astype(np.float32)).flatten()
                msg_w = OscMessageBuilder(address="/rig/wrenches")
                for v in flat_wrenches: msg_w.add_arg(float(v))
                bundle.add_content(msg_w.build())

            # 3. Send
            try:
                self.client.send(bundle.build())
                self.seq += 1
            except Exception as e:
                print(f"[TorqueSender Error] {e}")

            # 4. Sleep
            elapsed = time.time() - loop_start
            sleep_time = self.target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)