# SRIpy Codebase Documentation

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [Repository Structure](#2-repository-structure)
3. [Architecture Overview](#3-architecture-overview)
4. [Threading Model](#4-threading-model)
5. [Runtime Lifecycle](#5-runtime-lifecycle)
6. [Data Pipeline](#6-data-pipeline)
7. [Module Reference](#7-module-reference)
8. [Configuration and State](#8-configuration-and-state)
9. [Coordinate System](#9-coordinate-system)
10. [Error Handling](#10-error-handling)
11. [Testing](#11-testing)
12. [External Dependencies](#12-external-dependencies)

---

## 1. Purpose and Scope

SRIpy is a PyQt6 desktop application for **Synchrotron Radiation Interferometer (SRI)** beam diagnostics. It acquires frames from an Allied Vision Manta camera (via the Vimba/VmbPy SDK), reduces 2D images to 1D interference lineouts, runs a multi-stage nonlinear fit, and reports **beam sigma** and **fringe visibility** in real time.

**Key capabilities:**
- Live camera streaming with real-time fitting
- Burst acquisition (N-frame capture with per-frame analysis and aggregate statistics)
- Static file analysis (`.mat`, `.png`, `.jpg`, `.tif`, `.bmp`)
- Automatic MATLAB `.mat` metadata extraction (wavelength, slit, distance)
- Background subtraction and saturation detection
- Auto-centering ROI based on detected beam peak
- Persistent configuration across sessions
- Export to JSON/NPY and MATLAB `.mat` formats

---

## 2. Repository Structure

```
SRIpy-master/
├── main.py                        # Application entry point
├── sri_config.json                # Persisted user configuration
├── requirements.txt               # Runtime and test dependencies
├── pytest.ini                     # Test discovery and marker config
│
├── analysis/
│   ├── fitter.py                  # InterferenceFitter + FitResult
│   └── analysis_worker.py         # Threaded fit execution (QObject worker)
│
├── core/
│   ├── acquisition_manager.py     # Central model / orchestration
│   ├── acquisition.py             # BurstWorker
│   ├── config_manager.py          # Config load/save/defaults
│   └── data_model.py              # Dataclasses + DataManager (save helpers)
│
├── hardware/
│   ├── camera_interface.py        # Abstract base class for cameras
│   ├── manta_driver.py            # MantaDriver (Vimba/VmbPy implementation)
│   └── camera_io_thread.py        # CameraIoThread + CameraCommand enum
│
├── gui/
│   ├── main_window.py             # InterferometerView (QMainWindow shell)
│   ├── controllers/
│   │   └── interferometer_controller.py  # Full UI <-> model wiring
│   └── widgets/
│       ├── control_panel.py       # ControlPanelWidget (all controls + labels)
│       ├── live_monitor.py        # LiveMonitorWidget (image + lineout + ROI)
│       └── history_widget.py      # HistoryWidget (rolling sigma trend)
│
├── utils/
│   └── image_utils.py             # process_roi_lineout (shared preprocessing)
│
└── test/
    ├── conftest.py                # Shared fixtures and MockDriver
    ├── test_fitter.py             # Fitter unit tests
    ├── test_physics.py            # Physics consistency tests
    ├── test_integration_examples.py  # Pipeline integration tests
    ├── test_config_manager.py     # Config isolation tests
    ├── test_acquisition.py        # BurstWorker tests
    ├── test_main_window.py        # GUI logic tests
    ├── test_cam.py                # Hardware camera listing (requires Vimba)
    ├── test_driver.py             # Hardware visualization flow (requires camera)
    └── test_analysis.py           # End-to-end script example (requires camera)
```

---

## 3. Architecture Overview

SRIpy follows a layered **MVC-like architecture** with explicit producer/consumer threading:

```
┌─────────────────────────────────────────────────────────┐
│                      GUI Layer                          │
│  InterferometerView  ·  ControlPanelWidget              │
│  LiveMonitorWidget   ·  HistoryWidget                   │
└────────────────────────┬────────────────────────────────┘
                         │  Qt Signals / Slots
┌────────────────────────▼────────────────────────────────┐
│                  Controller Layer                        │
│            InterferometerController                      │
│   Wires UI signals to model; formats model data for UI   │
└────────────────────────┬────────────────────────────────┘
                         │  Direct calls + Qt Signals
┌────────────────────────▼────────────────────────────────┐
│                   Model Layer                            │
│              AcquisitionManager                          │
│   Owns ROI state, physics params, cached results         │
│   Dispatches frames and fit jobs; manages thread lifecycle│
└──────────┬─────────────────────────┬────────────────────┘
           │                         │
┌──────────▼──────────┐   ┌─────────▼──────────────────┐
│   Hardware Layer    │   │       Analysis Layer        │
│  CameraIoThread     │   │  AnalysisWorker (QThread)   │
│  MantaDriver        │   │  InterferenceFitter         │
└─────────────────────┘   └─────────────────────────────┘
```

**Data flows in one direction:** Hardware → Model → Controller → GUI. User actions travel in reverse via Qt signals.

---

## 4. Threading Model

Four concurrent threads operate during live acquisition:

| Thread | Class | Role |
|---|---|---|
| **Main (Qt GUI)** | — | Widget rendering, user interaction, signal dispatch |
| **Camera I/O** | `CameraIoThread(QThread)` | Command queue + frame acquisition loop |
| **Analysis** | `QThread` hosting `AnalysisWorker` | Runs `InterferenceFitter.fit()` per frame |
| **Burst** (transient) | `QThread` hosting `BurstWorker` | N-frame capture and per-frame fitting |

**Thread-safety mechanisms:**

- **`_fitter_lock` (`threading.Lock`)** — Held by `AnalysisWorker.process_fit()` while fitting, and by `AcquisitionManager.set_physics_params()` while updating wavelength/slit/distance. Prevents physics constants from changing mid-fit. The same fitter instance and lock are shared between the live analysis worker and the burst fitter.
- **Request IDs (`_fit_request_id`, `_inflight_fit_id`)** — Each fit dispatch is tagged with a monotonically increasing integer. `_handle_fit_result` discards any result whose ID does not match `_inflight_fit_id`, preventing stale results from old frames from corrupting the display.
- **Analysis busy/timeout flags** — `_analysis_busy` prevents queuing a new fit while the previous one is running. If a fit takes longer than `_analysis_timeout_s` (default 3 s), the flag is reset and a new request is allowed.
- **Burst ID (`_burst_id`)** — Incremented on every `START_BURST` command. The `BURST_COMPLETED` internal command carries the ID it was created with; cleanup is only performed if the ID matches, preventing delayed signals from a previous burst from killing a new one.
- **`_burst_frame_queue` (`queue.Queue(maxsize=100)`)** — `CameraIoThread` pushes frames into this queue while `BurstWorker` consumes them. Frames are dropped with a debug warning if the queue is full.

---

## 5. Runtime Lifecycle

### 5.1 Startup

`main.py`:
1. Configures logging (`INFO` level, timestamped format).
2. Sets pyqtgraph image axis order to `row-major`.
3. Creates `QApplication` with the `Fusion` style.
4. Installs `sys.excepthook = handle_exception` to catch unhandled exceptions, log them, and show a `QMessageBox.Critical` dialog.
5. Constructs `InterferometerView` and `InterferometerController`.
6. Shows the window and starts the Qt event loop.

### 5.2 Controller Initialization

`InterferometerController.__init__`:
1. Creates `AcquisitionManager` and stores it as `self.model`.
2. Calls `_connect_signals()` — wires all View ↔ Controller ↔ Model signals.
3. Calls `self.model.initialize()`:
   - Creates and starts the `AnalysisWorker` `QThread`.
   - Calls `MantaDriver.connect()` (raises `RuntimeError` on failure; shown as a dialog).
   - Creates and starts `CameraIoThread`, connecting all of its signals.
4. Calls `_apply_config_to_view()` — populates widgets from the loaded config, blocking signals to avoid feedback loops.
5. Calls `_sync_all_params()` — pushes current widget state into the model.
6. Updates the burst button label with the configured default frame count.

### 5.3 Live Acquisition

1. User toggles **Start Live** → `toggle_live_clicked(True)` → `_handle_live_toggle(True)` → `model.start_live()`.
2. Manager enqueues `CameraCommand.START_LIVE` to `CameraIoThread`.
3. `CameraIoThread._handle_command` calls `driver.start_stream()`, sets `_live_running = True`, emits `live_state_changed(True)`.
4. The `CameraIoThread.run()` loop continuously calls `driver.acquire_frame(timeout=0.1)`.
5. Each acquired frame is emitted via `frame_ready` → `AcquisitionManager._process_live_frame`.

### 5.4 Frame Processing

`AcquisitionManager._process_live_frame(raw_img)`:
1. Calls `process_roi_lineout(raw_img, roi_slice, transpose, background, saturation_threshold)`.
   - Returns `(display_image, full_lineout, is_saturated)`.
2. Caches `last_raw_image` and `last_lineout`.
3. Emits `saturation_updated` only if the saturation flag has changed.
4. Emits `live_data_ready(display_image, full_lineout)` to update the GUI.
5. If `autocenter_enabled` and signal is strong enough, recenters the ROI on the dominant peak and emits `roi_updated`.
6. Checks analysis throttle: skips if `_analysis_busy`, or if the timeout has not yet elapsed.
7. Emits `_request_fit(full_lineout, roi_hint, req_id)` to `AnalysisWorker.process_fit`.

**Display throttle:** The controller suppresses `_update_display` calls using `_display_throttle_ms` (0 = every frame). This reduces UI paint load at high frame rates without affecting analysis throughput.

### 5.5 Fit Processing

`AnalysisWorker.process_fit(y_data, x_data, req_id)`:
1. Acquires `_fitter_lock`.
2. Calls `InterferenceFitter.fit(y_data, roi_hint=roi_hint)`.
3. Releases lock.
4. Emits `result_ready(fit_result, fit_x, req_id)`.

`AcquisitionManager._handle_fit_result(result, x_axis, req_id)`:
1. Drops result if `req_id != _inflight_fit_id` (stale).
2. Clears `_analysis_busy`, `_inflight_fit_id`, `_analysis_timed_out`.
3. Caches `last_fit_result` and `last_fit_x`.
4. Emits `fit_result_ready(result, x_axis)` → `InterferometerController._update_stats`.

`InterferometerController._update_stats`:
- Plots fit curve and raw lineout.
- Updates peak/valley scatter markers.
- Updates display labels: visibility, sigma, raw visibility, max/min intensity, saturation/file status.
- Appends sigma to `HistoryWidget` when a successful fit is produced.

### 5.6 Burst Acquisition

1. User clicks burst button → `burst_clicked` → `_handle_burst(n)`.
2. Controller disables interactive controls and starts a progress bar.
3. `model.start_burst(n)` records live state and enqueues `START_BURST` with ROI/physics context.
4. `CameraIoThread` handles `START_BURST`:
   - Stops live stream if active.
   - Creates a new `QThread` + `BurstWorker`, passes the shared `_burst_frame_queue`.
   - Starts the burst thread.
   - While burst is running, continues the acquisition loop and routes frames into the frame queue.
5. `BurstWorker.run_burst()`:
   - **Phase 1 (Acquisition):** Pulls up to N frames from the queue (2 s per-frame timeout). Calls `process_roi_lineout` on each and crops to `roi_x_map`.
   - **Phase 2 (Analysis):** Calls `fitter.fit()` on each cropped lineout. Collects visibility, sigma, raw visibility, and intensity stats.
   - Builds `BurstResult` with per-frame histories and aggregate mean/std.
   - Emits `finished(BurstResult)`.
6. `_cleanup_burst` runs in the camera I/O thread loop: re-starts live stream if it was active before burst.
7. Controller re-enables controls and shows a completion dialog summarizing results.

### 5.7 Static File Loading

`AcquisitionManager.load_static_frame(file_path)`:
1. Backs up current physics parameters to `_live_physics_backup`.
2. For `.mat` files:
   - **Image extraction:** Searches candidate keys (`raw`, `IMG`, `img`, `image`, `data`) at the top level and in nested structured arrays.
   - **Metadata extraction:** Recursively hunts for known aliases for wavelength (`Lambda`, `lambda`, `wavelength`, `wl`), slit separation (`Slit_Separation`, `slit_sep`, `d`, `D`), and distance (`L`, `distance`, `dist`, `z`). Applies unit heuristics (e.g., wavelength < 1.0 → assume meters → convert to nm; slit < 0.1 → assume meters → convert to mm).
   - Emits `physics_loaded` to update UI spinners and calls `set_physics_params`.
3. For image files (`.png`, `.jpg`, `.tif`, etc.): loads with `cv2.imread(IMREAD_UNCHANGED)`; converts 3-channel to grayscale by averaging channels.
4. Routes the loaded image through `_process_live_frame` so the normal preprocessing/fit pipeline handles it identically to a live frame.
5. When `start_live()` is called after static mode, the backed-up physics are restored.

### 5.8 Save

`DataManager.save_dataset(directory, prefix, raw_image, metadata, result)`:
- Creates a timestamped subdirectory `{prefix}_{YYYYMMDD_HHMMSS}/`.
- Saves `raw_image.npy` (raw float32 array).
- Saves `experiment_data.json` (metadata + results, serialized via `NumpyEncoder`).

`DataManager.save_matlab(directory, prefix, raw_image, metadata, result)`:
- Writes a single `.mat` file with an `IMG` struct (raw image, lineout, fit curve, visibility, sigma, intensities, saturation flag) and a `META` struct.

`DataManager.save_burst(directory, prefix, burst_result, metadata)`:
- Writes a `{prefix}_BURST_{timestamp}.json` with burst statistics and per-frame histories.

### 5.9 Shutdown

When the main window closes:
1. `close_requested` signal → `InterferometerController.cleanup(event)`.
2. Snapshots current widget state and calls `config_manager.save(current_state)`.
3. Calls `model.shutdown()`:
   - Enqueues `STOP_LIVE`.
   - Calls `an_thread.quit()` and waits up to 2 s.
   - Enqueues `SHUTDOWN` to `CameraIoThread` and waits up to 2 s.
   - Calls `driver.close()` to release Vimba resources.

---

## 6. Data Pipeline

### Input
Camera frames: `uint16` NumPy arrays from `MantaDriver.acquire_frame()`. Burst frames may arrive as raw Vimba frame objects; `BurstWorker` casts them to `np.uint16`.

### Preprocessing — `utils/image_utils.process_roi_lineout`

```
Inputs: raw_img (H×W), roi_slice, transpose, bg_frame, saturation_thresh
```

1. `squeeze()` and cast to `float32`.
2. If `bg_frame` is provided and shapes match: subtract and clip to `[0, ∞)`.
3. **Saturation check:** `is_saturated = max(proc_img) >= saturation_thresh` (default 4090).
4. **ROI integration:**
   - `transpose=False`: Crop rows using `roi_slice`; sum over `axis=0` → horizontal lineout of length = image width.
   - `transpose=True`: Crop columns using `roi_slice`; sum over `axis=1` → vertical lineout of length = image height. Display image is returned as `proc_img.T` so visual axes swap automatically.
5. Returns `(display_image, full_lineout, is_saturated)`.

### Fit Dispatch

The **full lineout** (entire image width or height) is always sent to the fitter. The ROI x-limits provide only a **width hint**; the fitter independently locates the beam center and uses the hint to determine how wide a region to fit over. This decouples the display ROI position from fit stability.

### Fit Engine — `analysis/fitter.py`

`InterferenceFitter.fit(lineout, roi_hint)` executes five stages:

| Stage | Model | Purpose |
|---|---|---|
| **0** | `_gaussian` via `curve_fit` | Robust center and width estimate on the full lineout |
| **1** | `_sinc_sq_envelope` via `curve_fit` | Baseline and envelope amplitude locked on full data |
| **2** | FFT on full lineout | Fringe frequency estimate (searches bins 10–200) |
| **3** | `_sine_model` via `curve_fit` | Refined frequency and phase on a ±1.5-fringe window around center |
| **4** | `_full_interference_model` via `curve_fit` | Final bounded fit: visibility, center, frequency, phase on centered ROI-width region |

**Full interference model:**

$$I(x) = B + \left(A \cdot \frac{\sin\bigl(\omega(x - x_0)\bigr)}{\omega(x - x_0)}\right)^2 \cdot \left[1 + V \sin\bigl(k(x - x_0) + \phi\bigr)\right]$$

Where:
- $B$ = baseline intensity
- $A$ = envelope amplitude (inside the sinc² square)
- $\omega$ = sinc width parameter
- $x_0$ = beam center
- $V$ = fringe visibility $\in [0, 1]$
- $k$ = fringe wavenumber (radians/pixel)
- $\phi$ = fringe phase

**Raw visibility** is computed independently via `_calculate_raw_contrast`: finds the global intensity peak and its nearest adjacent valley (using `scipy.signal.find_peaks`), then returns $(I_{max} - I_{min}) / (I_{max} + I_{min})$.

### Physics Conversion — `calculate_sigma`

$$\sigma = \frac{\lambda L}{\pi d} \sqrt{\frac{1}{2} \ln\!\left(\frac{1}{V}\right)}$$

Where:
- $\lambda$ = wavelength (m)
- $L$ = source-to-screen distance (m)
- $d$ = slit separation (m)
- $V$ = fitted visibility

Visibility is clamped to `[0.001, 0.999]` before the log to avoid domain errors. Result is returned in meters; converted to microns by the caller (`sigma_microns = sigma * 1e6`).

### Output to UI
- Raw image displayed in `LiveMonitorWidget` with false-color (matplotlib `jet` LUT when available).
- Lineout (white) and fit overlay (red dashed) plotted with peak (yellow) and valley (cyan) scatter markers.
- Statistics panel updated: visibility, sigma (µm), raw visibility, max/min intensity, saturation status.
- Sigma appended to rolling `HistoryWidget` (100-point window, NaN-filtered).

### Output to Disk
- JSON + NPY dataset, MATLAB `.mat`, burst JSON summaries (see [§5.8](#58-save)).

---

## 7. Module Reference

### `main.py`
| Symbol | Type | Description |
|---|---|---|
| `main()` | function | Creates `QApplication`, instantiates view and controller, starts event loop |
| `setup_logging()` | function | Configures root logger at INFO level with timestamp format |
| `handle_exception(exc_type, exc_value, exc_traceback)` | function | Global `sys.excepthook`; logs full traceback and shows `QMessageBox.Critical` |

---

### `analysis/fitter.py`

#### `FitResult` (dataclass)
All fields have default values so a failed result can be constructed with `FitResult(success=False, ...)`.

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Whether the final stage converged |
| `visibility` | `float` | Model-fitted fringe visibility $\in [0, 1]$ |
| `raw_visibility` | `float` | Peak/valley contrast estimate |
| `sigma_microns` | `float` | Beam sigma in µm |
| `fitted_curve` | `ndarray \| None` | Model evaluated over `fit_x` |
| `fit_x` | `ndarray \| None` | Pixel indices of the fitted region |
| `params` | `dict \| None` | Named final parameters: `baseline`, `amplitude`, `sinc_width`, `sinc_center`, `visibility`, `sine_k`, `sine_phase` |
| `param_errors` | `dict \| None` | 1-σ errors from diagonal of covariance matrix |
| `pcov` | `ndarray \| None` | Full covariance matrix from `curve_fit` |
| `message` | `str` | Human-readable status or failure reason |
| `peak_idx` | `int \| None` | Absolute pixel index of detected intensity peak |
| `valley_idx` | `int \| None` | Absolute pixel index of nearest valley to peak |
| `max_intensity` | `float` | Peak intensity value |
| `min_intensity` | `float` | Valley intensity value |

#### `InterferenceFitter`

| Member | Description |
|---|---|
| `__init__(wavelength, slit_separation, distance, min_signal)` | `wavelength` in m, `slit_separation` in m, `distance` in m. `min_signal` = minimum signal range to attempt a fit (default 50.0). |
| `fit(lineout, roi_hint)` | Main entry point. `roi_hint` is an optional `(start, stop)` pixel tuple used only to determine fit-region width — centering is always done internally. Returns `FitResult`. |
| `calculate_sigma(visibility)` | Converts visibility to beam sigma (meters). Returns 0.0 for invalid or degenerate inputs. |
| `_gaussian(x, baseline, amp, center, width)` | Model for Stage 0 |
| `_sinc_sq_envelope(x, baseline, amp, width, center)` | Model for Stage 1 |
| `_sine_model(x, baseline, amp, freq, phase)` | Model for Stage 3 |
| `_full_interference_model(x, baseline, amp, sinc_w, sinc_x0, visibility, sine_freq, sine_phase)` | Full interference pattern model for Stage 4 |
| `_calculate_raw_contrast(y)` | Returns `(visibility, peak_idx, valley_idx, i_max, i_min)` |

---

### `analysis/analysis_worker.py`

#### `AnalysisWorker(QObject)`
Lives in a dedicated `QThread`. Instantiated by `AcquisitionManager._setup_analysis_thread()`.

| Member | Description |
|---|---|
| `result_ready` | Signal `(FitResult, ndarray, int)` — fit result, x-axis, request ID |
| `__init__(fitter, fitter_lock)` | Raises `ValueError` if `fitter_lock` is `None` |
| `process_fit(y_data, x_data, req_id)` | Acquires lock, runs `fitter.fit()`, emits `result_ready`. `x_data` can be a `(start, stop)` tuple (ROI hint) or a legacy `ndarray`. Returns a failed `FitResult` on any exception. |

---

### `core/acquisition_manager.py`

#### `AcquisitionManager(QObject)`
Central model class. Owns all mutable runtime state.

**Public Signals:**

| Signal | Payload | Description |
|---|---|---|
| `live_data_ready` | `(image, lineout)` | Emitted after each preprocessed frame |
| `fit_result_ready` | `(FitResult, ndarray)` | Emitted after each accepted fit result |
| `roi_updated` | `(y_min, y_max, x_min, x_max)` | Emitted when autocenter moves the ROI |
| `saturation_updated` | `bool` | Emitted only when saturation state changes |
| `background_ready` | `ndarray \| None` | Emitted after background capture |
| `live_state_changed` | `bool` | Emitted when camera starts/stops streaming |
| `burst_progress` | `int` | Percentage 0–100 during burst |
| `burst_finished` | `BurstResult` | Emitted on successful burst completion |
| `burst_error` | `str` | Emitted on burst failure |
| `error_occurred` | `str` | General error message |
| `camera_disconnected` | — | Emitted on hardware disconnection detection |
| `physics_loaded` | `(wave_nm, slit_mm, dist_m)` | Emitted when `.mat` metadata is loaded |

**Key State:**

| Attribute | Type | Description |
|---|---|---|
| `roi_slice` | `slice` | Row integration bounds in original (pre-transpose) image coordinates |
| `roi_x_limits` | `(int, int)` | Column fit bounds in original image coordinates |
| `autocenter_enabled` | `bool` | Whether ROI auto-centering is active |
| `transpose_enabled` | `bool` | Whether image/lineout axes are transposed |
| `subtract_background` | `bool` | Whether background subtraction is active |
| `background_frame` | `ndarray \| None` | Captured background frame (float32) |
| `saturation_threshold` | `int` | Pixel value threshold for saturation flag (default 4090) |
| `fitter` | `InterferenceFitter` | Live-mode fitter instance |
| `fitter_burst` | `InterferenceFitter` | Separate burst-mode fitter instance (same physics, avoids lock contention) |
| `last_raw_image` | `ndarray \| None` | Most recent display image |
| `last_lineout` | `ndarray \| None` | Most recent full lineout |
| `last_fit_result` | `FitResult \| None` | Most recent fit result |
| `last_fit_x` | `ndarray \| None` | Most recent fit x-axis |

**Key Methods:**

| Method | Description |
|---|---|
| `initialize()` | Connects camera and starts threads. Raises `RuntimeError` on camera failure. |
| `apply_config()` | Reads `self.cfg` and updates all internal state and fitter physics. |
| `set_roi(y_min, y_max, x_min, x_max)` | Updates ROI; re-processes static image immediately if in static mode. |
| `set_physics_params(wavelength, slit, distance)` | Updates both fitters under lock; re-processes static image if in static mode. |
| `set_exposure(val_ms)` | Enqueues `SET_EXPOSURE`; auto-disables background subtraction. |
| `set_gain(val_db)` | Enqueues `SET_GAIN`; auto-disables background subtraction. |
| `set_transpose(enabled)` | Updates transpose flag; re-processes static image. |
| `start_live()` / `stop_live()` | Enqueues `START_LIVE` / `STOP_LIVE`. Restores backed-up physics when exiting static mode. |
| `start_burst(n_frames)` | Enqueues `START_BURST` with full ROI/transpose/background context. |
| `capture_background()` | Enqueues `CAPTURE_BACKGROUND`. |
| `toggle_background(enabled)` | Enables/disables background subtraction. Emits `error_occurred` if no frame has been captured. |
| `load_static_frame(file_path)` | Loads image or `.mat`; extracts metadata; routes through normal pipeline. |
| `shutdown()` | Graceful teardown of all threads and hardware. |

---

### `core/acquisition.py`

#### `BurstWorker(QObject)`
Runs in a transient `QThread` managed by `CameraIoThread`.

| Member | Description |
|---|---|
| `progress` | Signal `(int)` — percentage complete (0–50% acquisition, 50–100% analysis) |
| `finished` | Signal `(BurstResult)` |
| `error` | Signal `(str)` |
| `run_burst()` | **Phase 1:** Pulls N frames from `frame_queue` (2 s timeout per frame). Calls `process_roi_lineout` and crops to `roi_x_map`. **Phase 2:** Calls `fitter.fit()` on each cropped lineout. Emits a zero-filled `BurstResult` if no frames were captured. |

---

### `core/data_model.py`

#### `ExperimentMetadata` (dataclass)

| Field | Default | Description |
|---|---|---|
| `exposure_s` | `0.005` | Camera exposure (seconds) |
| `gain_db` | `0.0` | Camera gain (dB) |
| `wavelength_nm` | `550.0` | Illumination wavelength (nm) |
| `slit_separation_mm` | `50.0` | Slit separation (mm) |
| `distance_m` | `16.5` | Source-to-screen distance (m) |
| `timestamp` | `datetime.now(UTC)` | UTC time at save |

#### `ExperimentResult` (dataclass)

| Field | Description |
|---|---|
| `visibility` | Model-fitted visibility |
| `raw_visibility` | Peak/valley raw visibility |
| `sigma_microns` | Beam sigma (µm) |
| `max_intensity` | Peak pixel intensity |
| `min_intensity` | Valley pixel intensity |
| `lineout_x` / `lineout_y` | 1D lineout coordinate and value arrays |
| `fit_y` | Fitted model curve |
| `is_saturated` | Saturation flag |

#### `BurstResult` (dataclass)

| Field | Description |
|---|---|
| `n_frames` | Number of frames captured |
| `mean_visibility` / `std_visibility` | Aggregate visibility statistics |
| `mean_sigma` / `std_sigma` | Aggregate sigma statistics |
| `mean_raw_visibility` | Mean raw visibility |
| `mean_max_intensity` / `mean_min_intensity` | Mean peak/valley intensities |
| `vis_history` | Per-frame visibility list |
| `sigma_history` | Per-frame sigma list |
| `timestamps` | Per-frame UNIX timestamps |
| `lineout_history` | Per-frame lineout arrays |

#### `NumpyEncoder`
`json.JSONEncoder` subclass that handles `np.integer`, `np.floating`, `np.ndarray`, and `datetime` objects. Used in all JSON save paths.

#### `DataManager`
Static save methods: `save_dataset`, `save_matlab`, `save_burst` (see [§5.8](#58-save)).

---

### `core/config_manager.py`

#### `ConfigManager`

Default configuration structure:

```json
{
  "camera":   { "exposure_ms": 5.0, "gain_db": 0.0, "transpose": false,
                "subtract_background": false, "saturation_threshold": 4090 },
  "analysis": { "min_signal_threshold": 50, "autocenter_min_signal": 200,
                "analysis_timeout_s": 3.0 },
  "burst":    { "default_frames": 50 },
  "physics":  { "wavelength_nm": 550.0, "slit_separation_mm": 50.0, "distance_m": 16.5 },
  "roi":      { "rows_min": 400, "rows_max": 800,
                "fit_width_min": 800, "fit_width_max": 1200, "auto_center": true }
}
```

| Method | Description |
|---|---|
| `load()` | Loads `sri_config.json`, deep-merges into defaults. Falls back to defaults silently on missing file, or with a `WARNING` log on parse errors. Returns the merged dict. |
| `save(current_state)` | Writes the provided dict to disk as formatted JSON. |
| `_deep_update(base, update)` | Recursive dict merge; preserves default keys not present in the file. |

> `ConfigManager.__init__` uses `copy.deepcopy(DEFAULT_CONFIG)` to ensure no shared nested state between instances.

---

### `hardware/camera_interface.py`

#### `CameraInterface(ABC)`
Abstract base class defining the contract all camera drivers must implement:

| Member | Description |
|---|---|
| `connect()` | Initialize camera connection |
| `exposure` (property) | Get/set exposure time in **seconds** |
| `gain` (property) | Get/set gain in **dB** |
| `acquire_frame(timeout)` | Acquire a single frame; returns `None` on timeout |
| `close()` | Release all resources |

---

### `hardware/manta_driver.py`

#### Exception Hierarchy

```
HardwareException
├── CameraDisconnected       # Camera no longer responding (10 consecutive failures)
├── CameraStreamingError     # Buffer/re-queue errors during streaming
├── FrameAcquisitionTimeout  # Acquisition timed out
└── CameraParameterError     # Failed to set exposure/gain/etc.
```

#### `MantaDriver(CameraInterface)`

Implements `CameraInterface` using `vmbpy`.

**Connection and configuration:**
- `connect()`: Initializes the `VmbSystem` context, selects camera by ID or uses first discovered, calls `_configure_defaults()`.
- `_configure_defaults()`: Sets packet size, software trigger mode, and pixel format.
- `close()`: Stops stream, exits camera and VmbSystem contexts safely.

**Acquisition modes:**
- **Streaming mode** (`start_stream()` / `stop_stream()`): Opens a frame queue (maxsize 5); a Vimba callback re-queues completed frames into `_frame_queue`. `acquire_frame()` pops from this queue with a timeout.
- **Snapshot mode** (`_acquire_single_snapshot()`): Used for single-frame capture (e.g., background) when streaming is not active.

**Disconnection detection:** Tracks `_consecutive_timeouts` and `_consecutive_frame_errors`. After `_disconnection_threshold` (10) consecutive failures, raises `CameraDisconnected`.

**Thread safety:** `_stream_lock` protects streaming state transitions; `_operation_lock` serializes parameter changes.

---

### `hardware/camera_io_thread.py`

#### `CameraCommand` (Enum)

| Value | Description |
|---|---|
| `START_LIVE` | Start continuous frame acquisition |
| `STOP_LIVE` | Stop frame acquisition |
| `SET_EXPOSURE` | Set camera exposure (arg: value in ms) |
| `SET_GAIN` | Set camera gain (arg: value in dB) |
| `CAPTURE_BACKGROUND` | Pause live, capture one frame as background, resume |
| `START_BURST` | Start N-frame burst (args: n, roi_slice, roi_x_limits, transpose, background) |
| `BURST_COMPLETED` | Internal: burst thread has finished (arg: burst_id) |
| `SHUTDOWN` | Exit the thread loop |

#### `CameraIoThread(QThread)`

Runs a tight loop: dequeues one command per iteration (non-blocking), then acquires a frame if live or burst is active.

**Signals:** `frame_ready`, `background_ready`, `live_state_changed`, `burst_progress`, `burst_finished`, `burst_error`, `error`, `camera_disconnected`.

**Key behaviors:**
- Commands other than `SHUTDOWN`, `BURST_COMPLETED`, and `START_BURST` are silently ignored while a burst is active.
- `START_BURST` increments `_burst_id` before starting, invalidating any pending `BURST_COMPLETED` signal from a previous burst.
- `_cleanup_burst()` is only executed from within the run loop (via the `BURST_COMPLETED` command), ensuring thread-safe object cleanup.
- Live stream is automatically re-started after burst completion if it was active before.
- On `CameraDisconnected`, emits `camera_disconnected` and `burst_error`, halts all activity.

---

### `gui/main_window.py`

#### `InterferometerView(QMainWindow)`
Minimal shell; all layout built in `_setup_ui()`.

- Window title: `"SRIpy"`, default size: 1300×950 px.
- Layout: `QHBoxLayout` with a `QTabWidget` (stretch 4) containing `LiveMonitorWidget` and `HistoryWidget`, and a `ControlPanelWidget` sidebar (stretch 1).
- Sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` to `"1"` at module import time to prevent NumPy/SciPy from over-subscribing CPU cores.
- Emits `close_requested(event)` from `closeEvent` so the controller handles cleanup and config saving.

---

### `gui/controllers/interferometer_controller.py`

#### `InterferometerController(QObject)`
Full signal wiring and UI state management.

**Signal connections (UI → Model):**

| Widget signal | Model action |
|---|---|
| `toggle_live_clicked` | `start_live()` / `stop_live()` |
| `exposure_changed` | `set_exposure()` |
| `gain_changed` | `set_gain()` |
| `physics_changed` | `set_physics_params()` (reads current spinner values) |
| `chk_transpose.toggled` | `set_transpose()` + one-time ROI value swap |
| `chk_autocenter.toggled` | `set_autocenter()` |
| `chk_bg.toggled` | `toggle_background()` |
| `acquire_bg_clicked` | `capture_background()` |
| `burst_clicked` | `start_burst(n)` |
| `roi_changed` | `set_roi()` (deferred until drag-end) |
| `reset_roi_clicked` | Reset ROI to config defaults, disable autocenter |
| `load_file_clicked` | `load_static_frame()` |
| `save_data_clicked` | `DataManager.save_dataset()` |
| `save_mat_clicked` | `DataManager.save_matlab()` |

**Signal connections (Model → UI):**

| Model signal | UI action |
|---|---|
| `live_data_ready` | Update image and lineout in `LiveMonitorWidget` (subject to display throttle) |
| `fit_result_ready` | Update fit overlay, stats labels, history |
| `roi_updated` | Move ROI handles (signals suppressed via `_suppress_roi_sync` to avoid feedback) |
| `saturation_updated` | Update `lbl_sat` text and color |
| `background_ready` | Enable background subtraction checkbox |
| `live_state_changed` | Toggle button text and interactive control enable states |
| `burst_progress` | Update progress bar |
| `burst_finished` | Show results dialog, re-enable controls |
| `burst_error` | Show error dialog, re-enable controls |
| `physics_loaded` | Update physics spinners with signals blocked |
| `camera_disconnected` | Show warning dialog, update UI state |

**Transpose ROI swap:** When `chk_transpose` is toggled, the controller performs a one-time swap of the row and column ROI values (rows ↔ columns) so the visible ROI handles track the beam in the new orientation. Subsequent updates use the normal coordinate mapping without any swap.

---

### `gui/widgets/control_panel.py`

#### `ControlPanelWidget(QWidget)`
Sidebar panel containing all user controls, organized into `QGroupBox` sections.

**Sections:** Analysis Results, ROI Controls, Calibration, Camera Settings, Physics Parameters, Burst Acquisition, File I/O.

**Key display labels:**

| Label | Color | Description |
|---|---|---|
| `lbl_vis` | Yellow (28px bold) | Fitted visibility |
| `lbl_sigma` | Cyan (28px bold) | Beam sigma (µm) |
| `lbl_sat` | Green/Red bold | Sensor saturation status |
| `lbl_raw_vis` | Magenta (16px bold) | Raw peak/valley visibility |
| `lbl_intensity` | Lime (14px bold) | Max and min intensity values |

**Spinboxes:** `spin_exp` (exposure ms, 0.05–1000), `spin_gain` (dB, 0–40), `spin_throttle` (display throttle ms, 0–200), `spin_lambda` (wavelength nm), `spin_slit` (slit mm), `spin_dist` (distance m), `spin_burst_count` (frame count).

---

### `gui/widgets/live_monitor.py`

#### `LiveMonitorWidget(QWidget)`
Two-panel layout: camera image (top, stretch 3) and interference lineout (bottom, stretch 2).

**Image panel:**
- `pg.ImageItem` with optional matplotlib `jet` LUT (gracefully omitted if matplotlib is unavailable).
- `roi_rows`: horizontal `pg.LinearRegionItem` (blue tint) — defines the vertical integration band. Bounds clamped to image height on each frame update.

**Lineout panel:**
- `curve_raw`: white line, raw lineout data.
- `curve_fit`: red dashed line, fitted model curve.
- `scatter_peak` / `scatter_valley`: yellow/cyan scatter markers at peak and valley positions.
- `roi_fit_width`: vertical `pg.LinearRegionItem` (green tint) — defines the horizontal fit region. Bounds clamped to lineout length on each frame update.

**Signals:**

| Signal | Description |
|---|---|
| `roi_changed()` | Emitted when either ROI region changes (during or after drag) |
| `roi_drag_start()` | Emitted at the start of any drag interaction |
| `roi_drag_end()` | Emitted when the user releases the mouse (triggers a final `set_roi` sync) |

**Auto-ranging:** On first frame or when image dimensions change, axes are reset to fit the data. Subsequent updates preserve user zoom.

---

### `gui/widgets/history_widget.py`

#### `HistoryWidget(QWidget)`
Rolling sigma trend plot backed by a fixed-length `np.ndarray` of `NaN`.

| Member | Description |
|---|---|
| `history_len` | Buffer size (default 100 points) |
| `history_data` | `np.ndarray` initialized to `NaN`; rolls on each `add_point` call |
| `add_point(value)` | Rolls the buffer, ignores non-finite values, plots only valid (non-NaN) points |

---

### `utils/image_utils.py`

#### `process_roi_lineout(img, roi_slice, transpose, bg_frame, saturation_thresh)`
The single shared preprocessing function used by both the live path and `BurstWorker`. See [§6](#6-data-pipeline) for detailed behavior.

**Returns:** `(display_image: ndarray, full_lineout: ndarray, is_saturated: bool)`

---

## 8. Configuration and State

Configuration is persisted in `sri_config.json` in the project root. `ConfigManager.load()` deep-merges the file into `DEFAULT_CONFIG`, so new keys added to defaults are always present even in existing config files from older versions.

**Config keys:**

| Section | Key | Default | Description |
|---|---|---|---|
| `camera` | `exposure_ms` | `5.0` | Initial exposure (ms) |
| | `gain_db` | `0.0` | Camera gain (dB) |
| | `transpose` | `false` | Image/axis transpose |
| | `subtract_background` | `false` | Background subtract (saved but not auto-applied on startup) |
| | `saturation_threshold` | `4090` | Pixel value threshold for saturation flag |
| `analysis` | `min_signal_threshold` | `50` | Minimum signal range to attempt a fit |
| | `autocenter_min_signal` | `200` | Minimum signal range to trigger auto-centering |
| | `analysis_timeout_s` | `3.0` | Seconds before a stalled analysis busy-flag is reset |
| `burst` | `default_frames` | `50` | Default burst frame count |
| `physics` | `wavelength_nm` | `550.0` | Illumination wavelength (nm) |
| | `slit_separation_mm` | `50.0` | Slit separation (mm) |
| | `distance_m` | `16.5` | Source-to-screen distance (m) |
| `roi` | `rows_min` / `rows_max` | `400` / `800` | Vertical integration bounds (pixels) |
| | `fit_width_min` / `fit_width_max` | `800` / `1200` | Horizontal fit region bounds (pixels) |
| | `auto_center` | `true` | Auto-center enable |

**State persistence flow:**
1. Manager loads config at construction; `apply_config()` sets all internal state.
2. Controller populates widgets from config (signals blocked to prevent feedback), then syncs all widget values back to the model.
3. On window close, controller snapshots current widget/model state and saves to disk.

---

## 9. Coordinate System

The ROI is always stored in **original (pre-transpose) image coordinates**, regardless of the current transpose setting. This ensures configuration remains consistent across transpose toggles and save/load cycles.

```
Stored in model:
  roi_slice       →  rows in original image  (vertical extent of integration band)
  roi_x_limits    →  columns in original image (horizontal extent of fit region)

Processing in process_roi_lineout:
  transpose=False:  crop roi_slice rows → sum axis=0 → lineout length = image width
  transpose=True:   use roi_slice as column range → sum axis=1 → lineout length = image height
                    display image returned as proc_img.T (visual axes swap automatically)

Auto-center logic:
  transpose=False:  peak_idx indexes a column → updates roi_x_limits
  transpose=True:   peak_idx indexes a row    → updates roi_slice

UI ROI handle mapping (controller):
  Vertical ROI handles   → always map to roi_slice
  Horizontal ROI handles → always map to roi_x_limits
  On transpose toggle:   one-time swap of row ↔ column values so handles track the beam
                         in the new visual orientation; subsequent updates need no swapping
```

---

## 10. Error Handling

| Layer | Strategy |
|---|---|
| **Application** | `sys.excepthook` converts all unhandled exceptions to a `QMessageBox.Critical` dialog with full traceback in the "Details" section |
| **Controller** | `RuntimeError` from `model.initialize()` shown as a connection error dialog; burst/file errors shown as warning dialogs |
| **AcquisitionManager** | Frame processing exceptions caught and logged; analysis and background state not corrupted |
| **AnalysisWorker** | All exceptions produce `FitResult(success=False, message=...)` rather than propagating; logged at ERROR level with traceback |
| **BurstWorker** | Per-frame conversion errors are logged and the frame is skipped; queue timeouts emit `error(str)` with the frame index |
| **MantaDriver** | Broad exception handling on all hardware calls with contextual logging; connection failures propagate as `RuntimeError` to allow the controller to show a UI dialog |
| **ConfigManager** | Missing file → returns defaults silently; corrupt/unparseable file → returns defaults with a `WARNING` log |
| **Static file loading** | `.mat` parse errors are logged at ERROR; raises `ValueError("Could not find image data")` to the controller if no valid image is extracted |

---

## 11. Testing

**Test runner:** `pytest` with discovery in `test/`. `pytest.ini` configures markers (`unit`, `integration`, `hardware`, `gui`) and enables `--strict-markers`.

| File | Markers | What is tested |
|---|---|---|
| `test_fitter.py` | `unit` | Parameter recovery from synthetic signals, frequency lock robustness, noise handling, edge cases (low signal, empty input) |
| `test_physics.py` | `integration` | End-to-end sigma recovery: synthesize fringe pattern from known beam size → fit → compare sigma |
| `test_integration_examples.py` | `integration` | Mock-based pipeline (frame → lineout → fit); basic acquisition lifecycle using `MockDriver` |
| `test_config_manager.py` | `unit` | Deep-copy isolation between instances, deep-merge behavior with partial configs |
| `test_acquisition.py` | `unit` | `BurstWorker` progress emission semantics, no-frame completion behavior |
| `test_main_window.py` | `gui` | ROI value mapping, saturation label logic, null-safety for missing fit results |
| `test_cam.py` | `hardware` | Camera listing utility (requires Vimba SDK + connected camera) |
| `test_driver.py` | `hardware` | Manual visualization flow (requires camera + display) |
| `test_analysis.py` | `hardware` | Full script-style connect → acquire → fit → plot (requires camera + display) |

**`test/conftest.py`** provides:
- `MockDriver`: Fake camera that replays a list of `ndarray` frames; supports `raise_on_connect=True` to test error paths.
- Synthetic interference pattern fixtures with configurable amplitude, frequency, noise, and physics parameters.

> Hardware tests require a physical Vimba-compatible camera and are not suitable for headless CI.

---

## 12. External Dependencies

| Package | Min Version | Role |
|---|---|---|
| `numpy` | 1.24 | Array operations throughout the codebase |
| `scipy` | 1.10 | `curve_fit` (all fit stages), `find_peaks` (raw contrast), `savemat` (MATLAB export) |
| `vmbpy` | 1.0.4 | Allied Vision Vimba SDK Python binding (`MantaDriver`) |
| `PyQt6` | 6.4 | GUI framework, signals/slots, `QThread`, `QObject` |
| `pyqtgraph` | 0.13 | Real-time image display and plot rendering |
| `matplotlib` | 3.7 | Optional: `jet` colormap LUT for camera display in `LiveMonitorWidget` |
| `opencv-python` | 4.8 | Loading static image files in `load_static_frame` |
| `pytest` | 7.3 | Test runner |

**Runtime notes:**
- The Vimba SDK and driver stack must be installed separately (not via pip) for live camera operation.
- `matplotlib` is imported inside a `try/except` in `LiveMonitorWidget`; the application runs correctly without it (greyscale display only).
- `OMP/MKL/NUMEXPR_NUM_THREADS=1` is set at import time in `gui/main_window.py` to prevent NumPy/SciPy internal thread pools from competing with the application's own threads.
