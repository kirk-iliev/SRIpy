# SRIpy: Synchrotron Radiation Interferometer Monitor

SRIpy is a high-performance Python application designed to monitor and analyze interference patterns from an SRI diagnostic system. 

## 🚀 Key Features

* **Asynchronous Processing:** GUI remains responsive during heavy math operations by offloading fitting to dedicated worker threads.
* **Dual-Phase Burst Mode:** Decouples camera acquisition from analysis to ensure 100% data integrity at high frame rates.
* **Real-Time Fitting:** Provides instantaneous Visibility and Sigma measurements using a Sinc²-modulated cosine model.
* **State Persistence:** Automatically saves and loads ROI, Physics, and Camera settings between sessions.

## 🛠️ Installation

### 1. Hardware Drivers
Ensure the **Allied Vision Vimba X** or **Vimba SDK** is installed on your system. This provides the underlying C-libraries required by `vmbpy` to communicate with Manta GigE cameras.

### 2. Python Environment
It is recommended to use a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
