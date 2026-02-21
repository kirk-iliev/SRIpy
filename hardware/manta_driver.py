import logging
import queue
import threading
import numpy as np
from typing import Optional
from vmbpy import (
    VmbSystem,
    Camera,
    Frame,
    FrameStatus,
    VmbCameraError,
    VmbFeatureError,
)
from typing import cast
from hardware.camera_interface import CameraInterface

class MantaDriver(CameraInterface):

    def __init__(self, camera_id: Optional[str] = None):
        self.camera_id = camera_id
        self.logger = logging.getLogger(__name__)
        self._vmb: Optional[VmbSystem] = None
        self._cam: Optional[Camera] = None
        self._feat_trigger_software = None

        # Streaming State
        self._is_streaming = False
        self._frame_queue = queue.Queue(maxsize=5)
        self._stream_lock = threading.Lock()  # Protects streaming state transitions
        self._accepting_frames = False  # Flag to discard in-flight frames during shutdown
        self._operation_lock = threading.Lock()

    def connect(self):
        """Initializes Vimba and connects to the camera."""
        if self._cam:
            return

        try:
            self._vmb = cast(VmbSystem, VmbSystem.get_instance())  # type: ignore
            self._vmb.__enter__()  # type: ignore

            if self.camera_id:
                try:
                    self._cam = self._vmb.get_camera_by_id(self.camera_id)  # type: ignore
                except VmbCameraError:
                    raise RuntimeError(f"Camera {self.camera_id} not found.")
            else:
                cams = self._vmb.get_all_cameras()  # type: ignore
                if not cams:
                    raise RuntimeError("No cameras found via Vimba.")
                self._cam = cams[0]

            self._cam.__enter__()  # type: ignore
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Connected to: {self._cam.get_name()} (ID: {self._cam.get_id()})")  # type: ignore

            self._configure_defaults()

        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.close()
            raise

    def close(self):
        """Safely closes camera and system resources."""
        self.stop_stream() # Ensure stream is stopped first

        if self._cam:
            try:
                self._cam.__exit__(None, None, None)  # type: ignore
            except Exception as e:
                self.logger.error(f"Error closing camera: {e}")
            self._cam = None

        if self._vmb:
            try:
                self._vmb.__exit__(None, None, None)  # type: ignore
            except Exception as e:
                self.logger.error(f"Error closing Vimba system: {e}")
            self._vmb = None
        self.logger.info("Camera connection closed.")

    def _configure_defaults(self):
        """Sets up GigE packet sizes, triggers, and pixel formats."""
        if not self._cam:
            return

        # Optimize Packet Size
        for feature_name in ["GevSCPSPacketSize", "PacketSize"]:
            feat = self._get_feature(feature_name)
            if feat:
                try:
                    feat.set(1400)  # type: ignore
                    break
                except VmbFeatureError:
                    self.logger.debug(f"Packet size feature {feature_name} not settable")
            else:
                self.logger.debug(f"Packet size feature {feature_name} not available")

        # Setup Software Triggering
        try:
            trigger_mode = self._get_feature("TriggerMode")
            trigger_source = self._get_feature("TriggerSource")

            if trigger_mode and trigger_source:
                trigger_mode.set('On')  # type: ignore
                trigger_source.set('Software')  # type: ignore

            self._feat_trigger_software = self._get_feature("TriggerSoftware")
            if not self._feat_trigger_software:
                self.logger.warning("TriggerSoftware feature not found. Acquisition may fail.")

        except VmbFeatureError as e:
            self.logger.error(f"Error configuring trigger: {e}")

        # Pixel Format
        pix_fmt = self._get_feature("PixelFormat")
        if pix_fmt:
            try:
                available = pix_fmt.get_available_entries()  # type: ignore
                target_fmt = 'Mono12' if 'Mono12' in available else 'Mono8'
                pix_fmt.set(target_fmt)  # type: ignore
            except VmbFeatureError:
                self.logger.debug("Pixel format set failed")
        else:
            self.logger.debug("PixelFormat feature not available")

    @property
    def exposure(self) -> float:
        """Exposure time in seconds."""
        feat = self._get_feature("ExposureTimeAbs") or self._get_feature("ExposureTime")
        if feat:
            try:
                return float(feat.get())  / 1_000_000.0  # type: ignore
            except VmbFeatureError:
                pass
        return 0.0

    @exposure.setter
    def exposure(self, exposure_time_s: float):
        feat = self._get_feature("ExposureTimeAbs") or self._get_feature("ExposureTime")
        with self._operation_lock:
            if feat:
                try:
                    min_exp, max_exp = feat.get_range()  # type: ignore
                    req_us = exposure_time_s * 1_000_000.0
                    target_us = max(min_exp, min(max_exp, req_us))
                    feat.set(target_us)  # type: ignore
                except VmbFeatureError as e:
                    self.logger.error(f"Failed to set exposure: {e}")

    @property
    def gain(self) -> float:
        """Gain in dB."""
        feat = self._get_feature("Gain")
        return float(feat.get()) if feat else 0.0  # type: ignore

    @gain.setter
    def gain(self, gain_db: float):
        feat = self._get_feature("Gain")
        with self._operation_lock:
            if feat:
                try:
                    min_g, max_g = feat.get_range()  # type: ignore
                    target = max(min_g, min(max_g, gain_db))
                    feat.set(target)  # type: ignore
                except VmbFeatureError as e:
                    self.logger.error(f"Failed to set gain: {e}")

    def start_stream(self):
        """Prepares the camera for continuous acquisition."""
        with self._operation_lock:
            with self._stream_lock:
                if self._is_streaming or not self._cam:
                    return
                with self._frame_queue.mutex:
                    self._frame_queue.queue.clear()
                    self._frame_queue.unfinished_tasks = 0

                # Define the handler inside to capture 'self'
                def frame_handler(cam: Camera, frame: Frame):
                    # Check shutdown flag under lock to prevent race condition
                    with self._stream_lock:
                        accepting = self._accepting_frames

                    # Skip frames if stream is shutting down
                    if not accepting:
                        try:
                            cam.queue_frame(frame)
                        except Exception as e:
                            self.logger.debug(f"Failed to re-queue frame during shutdown: {e}")
                        return

                    try:
                        if frame.get_status() == FrameStatus.Complete:
                            # If queue is full, discard oldest frame to make room for newest
                            if self._frame_queue.full():
                                try:
                                    self._frame_queue.get_nowait()
                                except queue.Empty:
                                    self.logger.debug("Frame queue empty while trimming")

                            arr = frame.as_numpy_ndarray().copy()
                            # Use block=True with timeout to ensure frame is queued
                            # If this fails, the frame is simply not captured (caller will timeout)
                            try:
                                self._frame_queue.put(arr, block=True, timeout=0.1)
                            except queue.Full:
                                self.logger.debug("Frame queue full, frame dropped")
                    except Exception as e:
                        self.logger.debug(f"Frame handler conversion error: {e}")
                    finally:
                        # Always re-queue frame to camera to prevent buffer starvation
                        try:
                            cam.queue_frame(frame)
                        except Exception as e:
                            self.logger.error(f"Failed to re-queue frame: {e}", exc_info=True)

                try:
                    # Enable frame acceptance before starting stream
                    self._accepting_frames = True
                    # Start streaming with 5 buffers to prevent dropped frames
                    self._cam.start_streaming(handler=frame_handler, buffer_count=5)  # type: ignore
                    self._is_streaming = True
                    self.logger.info("Continuous streaming started.")
                except Exception as e:
                    self.logger.error(f"Failed to start stream: {e}")
                    self._accepting_frames = False

    def stop_stream(self):
        """Stops continuous acquisition."""
        with self._operation_lock:
            with self._stream_lock:
                if not self._is_streaming or not self._cam:
                    return

                try:
                    # Signal frame handler to stop accepting frames
                    self._accepting_frames = False

                    # Stop the camera stream
                    self._cam.stop_streaming()
                    self._is_streaming = False

                    with self._frame_queue.mutex:
                        self._frame_queue.queue.clear()
                        self._frame_queue.unfinished_tasks = 0

                    self._frame_queue.put(None)
                    self.logger.info("Continuous streaming stopped & frame queue cleared.")

                except Exception as e:
                    self.logger.error(f"Failed to stop stream: {e}", exc_info=True)

    def acquire_frame(self, timeout: float = 3.0) -> Optional[np.ndarray]:
        """
        Acquires a single frame from the streaming queue.
        Requires start_stream() to be called first.

        Args:
            timeout: Seconds to wait for frame

        Returns:
            numpy.ndarray: Image data if successful
            None: On timeout

        Raises:
            RuntimeError: Camera not connected or streaming not started
            Exception: Vimba errors, hardware issues (logged with context)
        """
        if not self._cam:
            raise RuntimeError("Camera not connected.")

        # Check streaming state under lock to avoid race conditions
        with self._stream_lock:
            is_streaming = self._is_streaming

        if not is_streaming:
            raise RuntimeError("Streaming not started. Call start_stream() first.")

        try:
            # Fire Software Trigger
            if self._feat_trigger_software:
                try:
                    self._feat_trigger_software.run()  # type: ignore
                except Exception as e:
                    self.logger.warning(f"Trigger failed: {e}", exc_info=True)
                    # Continue anyway; frame might already be in queue

            # Wait for result in queue
            frame = self._frame_queue.get(block=True, timeout=timeout)

            if frame is None:
                self.logger.debug(f"Acquire returned None frame (stream stopped)")
                return None
            return frame

        except queue.Empty:
            self.logger.debug(f"Acquire timeout after {timeout}s")
            return None

        except Exception as e:
            self.logger.error(
                f"Unexpected error acquiring frame: {type(e).__name__}: {e}",
                exc_info=True
            )
            raise


    def _get_feature(self, name: str):
        if not self._cam: return None
        try: return self._cam.get_feature_by_name(name)
        except VmbFeatureError as e:
            self.logger.debug(f"Feature not available: {name} ({e})")
            return None