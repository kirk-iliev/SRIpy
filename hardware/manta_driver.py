import numpy as np
from vmbpy import *
from .camera_interface import CameraInterface

class MantaDriver(CameraInterface):
    def __init__(self, camera_id=None):
        self.camera_id = camera_id
        self.vmb = None
        self.cam = None

    def connect(self):
        # Start the Vimba System
        self.vmb = VmbSystem.get_instance()
        self.vmb.__enter__()
        
        # Find Camera
        if self.camera_id:
            try:
                self.cam = self.vmb.get_camera_by_id(self.camera_id)
            except VmbCameraError:
                raise RuntimeError(f"Camera {self.camera_id} not found.")
        else:
            cams = self.vmb.get_all_cameras()
            if not cams:
                raise RuntimeError("No cameras found via Vimba.")
            self.cam = cams[0]

        # Open camera
        self.cam.__enter__()
        
        print(f"Connected to: {self.cam.get_name()} (ID: {self.cam.get_id()})")
        self._configure_defaults()

    def _get_feature(self, name):
        """Helper to safely get a feature, returning None if missing."""
        try:
            return self.cam.get_feature_by_name(name)
        except VmbFeatureError:
            return None

    def _configure_defaults(self):
        """Apply recommended settings for reliable GigE streaming."""
        # Packet size adjustment 
        packet_size = self._get_feature("GevSCPSPacketSize")
        if packet_size:
            try:
                packet_size.set(1400)
            except Exception as e:
                print(f"Warning: Could not set Packet Size: {e}")
        else:
            packet_size = self._get_feature("PacketSize")
            if packet_size:
                packet_size.set(1400)

        # Trigger Mode (Manual Software Trigger)
        trigger_mode = self._get_feature("TriggerMode")
        trigger_source = self._get_feature("TriggerSource")
        
        if trigger_mode:
            trigger_mode.set('On')
            trigger_source.set('Software')
            print(f"debug: TriggerMode set to {trigger_mode.get()}, TriggerSource set to {trigger_source.get()}")
        # Pixel Format (Prefer Mono12)
        pix_fmt = self._get_feature("PixelFormat")
        if pix_fmt:
            available = pix_fmt.get_available_entries()
            if 'Mono12' in available:
                pix_fmt.set('Mono12')
            else:
                pix_fmt.set('Mono8')

    def set_exposure(self, exposure_time_s: float):
        # Manta uses microseconds. Clamp to ~40us minimum.
        exp_us = max(40, exposure_time_s * 1_000_000)
        
        # Try finding the correct feature name
        exposure_feat = self._get_feature("ExposureTimeAbs") # Legacy Manta
        if not exposure_feat:
            exposure_feat = self._get_feature("ExposureTime") # GenICam Standard
            
        if exposure_feat:
            exposure_feat.set(exp_us)
        else:
            print("Error: Could not find ExposureTime feature.")

    def set_gain(self, gain_db: float):
        gain_feat = self._get_feature("Gain")
        if gain_feat:
            min_g, max_g = gain_feat.get_range()
            target = max(min_g, min(max_g, gain_db))
            gain_feat.set(target)
        else:
            print("Error: Could not find Gain feature.")

    def acquire_frame(self) -> np.ndarray:
        if not self.cam:
            raise RuntimeError("Camera not connected. Call connect() first.")

        # Prepare to capture a single frame asynchronously
        captured_frame = []
        
        # Callback function to handle incoming frames
        def frame_handler(cam, stream, frame):
            if frame.get_status() == FrameStatus.Complete:
                # Convert to numpy array and store
                captured_frame.append(frame.as_numpy_ndarray())
            
            # Re-queue the frame for future use
            cam.queue_frame(frame)

        try:
            # start streaming
            self.cam.start_streaming(handler=frame_handler, buffer_count=3)
            
            # fire trigger if listening
            trigger_soft = self._get_feature("TriggerSoftware")
            if trigger_soft:
                trigger_soft.run()
            else:
                print("Warning: TriggerSoftware feature not found.")
            
            # wait for image
            import time
            timeout = 3.0
            start_time = time.time()
            
            while not captured_frame:
                if time.time() - start_time > timeout:
                    print("Error: Timeout waiting for frame (Async).")
                    break
                time.sleep(0.01) # Check every 10ms
            
            # stop listening
            self.cam.stop_streaming()
            
            if captured_frame:
                return captured_frame[0]
            else:
                return np.zeros((10, 10))

        except Exception as e:
            print(f"Error acquiring frame: {e}")
            try:
                self.cam.stop_streaming()
            except:
                pass
            return np.zeros((10, 10))

    def close(self):
        if self.cam:
            self.cam.__exit__(None, None, None)
        if self.vmb:
            self.vmb.__exit__(None, None, None)
        print("Camera Connection Closed.")