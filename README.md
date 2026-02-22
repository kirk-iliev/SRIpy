# SRIpy

SRIpy is a high-performance diagnostic application for Synchrotron Radiation Interferometer (SRI) systems. It acquires images from GigE cameras (AVT Manta), reduces 2D interference patterns to 1D lineouts, and performs real-time curve fitting to measure beam size ($\sigma$) and visibility.

## Key Features

* **Real-time Analysis:** High-speed 4-stage non-linear least squares fitting for visibility and beam size.
* **Intelligent ROI:** Auto-centering and adaptive Region of Interest (ROI) tracking for the interference core.
* **Burst Mode:** High-speed capture of multiple frames to memory with asynchronous processing to prevent frame drops.
* **Data Export:** Save datasets and fit results to `.json`, `.npy`, and MATLAB `.mat` formats.
* **Calibration:** Integrated background subtraction and saturation monitoring.

## System Architecture

The application uses a multi-threaded producer-consumer model to maintain UI responsiveness:

* **Acquisition:** A low-latency thread manages the camera ring buffer and image preprocessing (ROI cropping, integration).
* **Analysis:** A dedicated worker thread executes the fitting algorithm, decoupling heavy math from the acquisition and UI loops.
* **Controller-Driven:** A central controller manages state synchronization between the hardware, analysis, and PyQt6 view.

## Physics Model
<p align="center">
  <img width="500" height="127" alt="Interference Pattern" src="https://github.com/user-attachments/assets/c6f700be-68a1-4bad-86db-8902ebc03a6e" />
  <br>
  <em>Figure 1: Typical SRI optical layout. <sup><a href="#ref1">[1]</a></sup></em>
</p>

The core fitting engine (`analysis.fitter.InterferenceFitter`) models the intensity profile using a Sinc²-modulated sine-interference function:

$$I(x) = B + A \cdot \text{sinc}^2(w(x - x_0)) \cdot [ 1 + V \cdot \sin(k(x - x_0) + \phi) ]$$

Where:
* $V$ is the Visibility (contrast).
* $\sigma$ (Beam Size) is derived from $V$ using:
    $$\sigma = \frac{\lambda D}{\pi d} \sqrt{0.5 \ln(1/V)}$$

## Hardware Requirements

* **Camera:** Allied Vision Technologies (AVT) Manta Series (GigE).
* **Software:** Vimba X or Vimba SDK (drivers must be installed for `vmbpy` to function).

## Installation

### 1. System Drivers
Install the **Vimba SDK** from Allied Vision. Verify the camera is detected in Vimba Viewer before starting SRIpy.

### 2. Python Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Running
```bash
python main.py
```

## Notes
* **Network:** Use a static IP address for the camera for reliable connection.
* **OS Support:** Optimized for Linux and Windows. MacOS is not recommended due to limited Vimba driver support.

## References

<a id="ref1"></a>[1] W. Li et al., ["Synchrotron radiation interferometry for beam size measurement at low current and in large dynamic range,"](https://doi.org/10.1103/PhysRevAccelBeams.25.080702) *Phys. Rev. Accel. Beams* **25**, 080702 (2022).
