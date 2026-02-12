import numpy as np

def process_roi_lineout(img, roi_slice, transpose, bg_frame=None, saturation_thresh=4095):
    """
    Shared image processing: Background sub, Transpose mapping, and ROI integration.
    Returns: (display_image, full_lineout, is_saturated)
    """
    # Cast & Background Subtraction
    proc_img = img.squeeze().astype(np.float32, copy=False)
    
    if bg_frame is not None and proc_img.shape == bg_frame.shape:
        proc_img -= bg_frame
        np.clip(proc_img, 0, None, out=proc_img)

    # Saturation Check
    is_saturated = bool(np.max(proc_img) >= saturation_thresh)

    # ROI Integration
    h, w = proc_img.shape

    if transpose:
        # Transpose Mode: User slice (rows) -> Array columns (Axis 1)
        col_start = max(0, min(roi_slice.start, w))
        col_end = max(col_start, min(roi_slice.stop, w))
        
        crop = proc_img[:, col_start:col_end]
        lineout = np.sum(crop, axis=1)  # Vertical integration
        display_img = np.ascontiguousarray(proc_img.T)
    else:
        # Standard Mode: User slice (rows) -> Array rows (Axis 0)
        row_start = max(0, min(roi_slice.start, h))
        row_end = max(row_start, min(roi_slice.stop, h))
        
        crop = proc_img[row_start:row_end, :]
        lineout = np.sum(crop, axis=0)  # Horizontal integration
        display_img = proc_img

    return display_img, lineout, is_saturated