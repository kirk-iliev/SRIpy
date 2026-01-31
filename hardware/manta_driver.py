import logging
import queue
import numpy as np
from typing import Optional
from vmbpy import *

class MantaDriver: 

    def __init__(self, camera_id: Optional[str] = None):
        self.camera_id = camera_id
        self._vmb: Optional[VmbSystem] = None
        self._cam: Optional[Camera] = None
        self._feat_trigger_software = None
        
        # Streaming State
        self._is_streaming = False
        self._frame_queue = queue.Queue(maxsize=1)

    def connect(self):
        """Initializes Vimba and connects to the camera."""
        if self._cam:
            return

        try:
            self._vmb = VmbSystem.get_instance()
            self._vmb.__enter__()

            if self.camera_id:
                try:
                    self._cam = self._vmb.get_camera_by_id(self.camera_id)
                except VmbCameraError:
                    raise RuntimeError(f"Camera {self.camera_id} not found.")
            else:
                cams = self._vmb.get_all_cameras()
                if not cams:
                    raise RuntimeError("No cameras found via Vimba.")
                self._cam = cams[0]

            self._cam.__enter__()
            # Setup logging
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Connected to: {self._cam.get_name()} (ID: {self._cam.get_id()})")
            
            self._configure_defaults()
            
        except Exception as e:
            print(f"Connection failed: {e}")
            self.close()
            raise

    def close(self):
        """Safely closes camera and system resources."""
        self.stop_stream() # Ensure stream is stopped first
        
        if self._cam:
            try:
                self._cam.__exit__(None, None, None)
            except Exception as e:
                print(f"Error closing camera: {e}")
            self._cam = None
            
        if self._vmb:
            try:
                self._vmb.__exit__(None, None, None)
            except Exception as e:
                print(f"Error closing Vimba system: {e}")
            self._vmb = None
        print("Camera connection closed.")

    def _configure_defaults(self):
        """Sets up GigE packet sizes, triggers, and pixel formats."""
        if not self._cam:
            return

        # Optimize Packet Size 
        for feature_name in ["GevSCPSPacketSize", "PacketSize"]:
            feat = self._get_feature(feature_name)
            if feat:
                try:
                    feat.set(1400) 
                    break 
                except VmbFeatureError:
                    pass

        # Setup Software Triggering
        try:
            trigger_mode = self._get_feature("TriggerMode")
            trigger_source = self._get_feature("TriggerSource")
            
            if trigger_mode and trigger_source:
                trigger_mode.set('On')
                trigger_source.set('Software')
                
            self._feat_trigger_software = self._get_feature("TriggerSoftware")
            if not self._feat_trigger_software:
                print("TriggerSoftware feature not found. Acquisition may fail.")
                
        except VmbFeatureError as e:
            print(f"Error configuring trigger: {e}")

        # Pixel Format
        pix_fmt = self._get_feature("PixelFormat")
        if pix_fmt:
            try:
                available = pix_fmt.get_available_entries()
                target_fmt = 'Mono12' if 'Mono12' in available else 'Mono8'
                pix_fmt.set(target_fmt)
            except VmbFeatureError:
                pass

    @property
    def exposure(self) -> float:
        """Exposure time in seconds."""
        feat = self._get_feature("ExposureTimeAbs") or self._get_feature("ExposureTime")
        if feat:
            try:
                return feat.get() / 1_000_000.0
            except VmbFeatureError:
                pass
        return 0.0

    @exposure.setter
    def exposure(self, exposure_time_s: float):
        feat = self._get_feature("ExposureTimeAbs") or self._get_feature("ExposureTime")
        if feat:
            try:
                min_exp, max_exp = feat.get_range()
                req_us = exposure_time_s * 1_000_000.0
                target_us = max(min_exp, min(max_exp, req_us))
                feat.set(target_us)
            except VmbFeatureError as e:
                print(f"Failed to set exposure: {e}")

    @property
    def gain(self) -> float:
        """Gain in dB."""
        feat = self._get_feature("Gain")
        return feat.get() if feat else 0.0

    @gain.setter
    def gain(self, gain_db: float):
        feat = self._get_feature("Gain")
        if feat:
            try:
                min_g, max_g = feat.get_range()
                target = max(min_g, min(max_g, gain_db))
                feat.set(target)
            except VmbFeatureError as e:
                print(f"Failed to set gain: {e}")

    def start_stream(self):
        """Prepares the camera for continuous acquisition."""
        if self._is_streaming or not self._cam:
            return

        # Define the handler inside to capture 'self'
        def frame_handler(cam: Camera, stream: Stream, frame: Frame):
            if frame.get_status() == FrameStatus.Complete:
                # If queue is full, discard oldest frame to reduce latency
                if self._frame_queue.full():
                    try: self._frame_queue.get_nowait()
                    except queue.Empty: pass
                
                self._frame_queue.put(frame.as_numpy_ndarray().copy())
            
            cam.queue_frame(frame)

        try:
            # Start streaming with 5 buffers to prevent dropped frames
            self._cam.start_streaming(handler=frame_handler, buffer_count=5)
            self._is_streaming = True
            print("Continuous streaming started.")
        except Exception as e:
            print(f"Failed to start stream: {e}")

    def stop_stream(self):
        """Stops continuous acquisition."""
        if not self._is_streaming or not self._cam:
            return
        
        try:
            self._cam.stop_streaming()
            self._is_streaming = False
            
            # Clear the queue
            with self._frame_queue.mutex:
                self._frame_queue.queue.clear()
            print("Continuous streaming stopped.")
        except Exception as e:
            print(f"Failed to stop stream: {e}")

    def acquire_frame(self, timeout: float = 3.0) -> Optional[np.ndarray]:
        """
        Acquires a single frame. 
        If start_stream() was called, pulls from the live queue.
        If not, performs a one-off capture (snapshot mode).
        Returns None on timeout/error.
        """
        if not self._cam:
            raise RuntimeError("Camera not connected.")

        # --- MODE A: Streaming ---
        if self._is_streaming:
            try:
                # Fire Software Trigger
                if self._feat_trigger_software:
                    self._feat_trigger_software.run()
                
                # Wait for result in queue
                return self._frame_queue.get(block=True, timeout=timeout)
            except queue.Empty:
                print("Acquire timeout (Streaming)")
                return None

        # --- MODE B: Snapshot ---
        else:
            return self._acquire_single_snapshot(timeout)

    def _acquire_single_snapshot(self, timeout):
        """Legacy helper for one-off snapshots."""
        q = queue.Queue(maxsize=1)
        
        def handler(cam, stream, frame):
            if frame.get_status() == FrameStatus.Complete:
                q.put(frame.as_numpy_ndarray().copy())
            cam.queue_frame(frame)

        try:
            self._cam.start_streaming(handler=handler, buffer_count=1)
            if self._feat_trigger_software:
                self._feat_trigger_software.run()
            return q.get(block=True, timeout=timeout)
        except Exception:
            return None
        finally:
            try: self._cam.stop_streaming()
            except: pass

    def _get_feature(self, name: str):
        if not self._cam: return None
        try: return self._cam.get_feature_by_name(name)
        except VmbFeatureError: return None