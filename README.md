# SRIpy

SRIpy is a diagnostic application for Synchrotron Radiation Interferometer (SRI) systems. It acquires images from GigE cameras (AVT Manta), isolates the interference fringe pattern, and performs live curve fitting to extract beam size ($\sigma$) and visibility parameters.

## System Architecture

The application separates the camera and analysis loops:

* **Acquisition (Producer):** A dedicated thread handles the Vimba ring buffer, pushing raw frames to a thread-safe queue.
* **Analysis (Consumer):** A background worker pulls frames for least-squares fitting, ensuring the UI remains responsive even during heavy computation.
* **Burst Mode:** For high-speed capture, frames are buffered directly to memory and processed asynchronously to prevent frame drops.

## Physics Model
<p align="center">
  <img width="500" height="127" alt="Interference Pattern" src="https://github.com/user-attachments/assets/c6f700be-68a1-4bad-86db-8902ebc03a6e" />
  <br>
  <em>Figure 1: Typical SRI optical layout of an electron beam size measurement system. <sup><a href="#ref1">[1]</a></sup></em>
</p>

The core analysis engine (`analysis.fitter.InterferenceFitter`) models the intensity profile using a Sinc²-modulated cosine function:

$$I(x) = B + A \cdot \text{sinc}^2\left(\frac{x - x_0}{w}\right) \cdot \left[ 1 + V \cdot \cos(k x + \phi) \right]$$

Where:
* $V$ is the Visibility (contrast).
* $\sigma$ (Beam Size) is derived from $V$ using:
    $$\sigma = \frac{\lambda D}{\pi d} \sqrt{0.5 \ln(1/V)}$$

## Hardware Requirements

* **Camera:** Allied Vision Technologies (AVT) Manta Series (GigE).
* **Drivers:** Vimba X or Vimba SDK (must be installed at the OS level for `vmbpy` to function).

## Installation

### 1. System Drivers
Install the **Vimba SDK** from Allied Vision. Ensure the Vimba viewer can detect the camera before running Python scripts.

### 2. Download
Download or clone the repo into your folder of choice

### 3. Python Environment
Run this in the root folder:
```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt

```

## Running

In the root folder, open a terminal and run:
```bash
python main.py
```

## Notes
Recommend assigning your GigE camera a static IP address. This can be done in the Vimba Viewer app from AVT.

## References

<a id="ref1"></a>[1] W. Li et al., ["Synchrotron radiation interferometry for beam size measurement at low current and in large dynamic range,"](https://doi.org/10.1103/PhysRevAccelBeams.25.080702) *Phys. Rev. Accel. Beams* **25**, 080702 (2022).
