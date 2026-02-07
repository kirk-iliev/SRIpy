import logging
from vmbpy import *

logger = logging.getLogger(__name__)

def test_camera():
    logger.info("Initializing Vimba...")
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        logger.info(f"Cameras found: {len(cams)}")

        for cam in cams:
            logger.info(f" - ID: {cam.get_id()}, Name: {cam.get_name()}")
if __name__ == "__main__":
    test_camera()