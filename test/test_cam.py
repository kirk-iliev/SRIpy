from vmbpy import *

def test_camera():
    print("Initializing Vimba...")
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        print(f"Cameras found: {len(cams)}")

        for cam in cams:
            print(f" - ID: {cam.get_id()}, Name: {cam.get_name()}")
if __name__ == "__main__":
    test_camera()