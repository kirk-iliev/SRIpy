# SRIpy Codebase Documentation

## 1. Purpose and Scope

SRIpy is a PyQt-based desktop application for Synchrotron Radiation Interferometer (SRI) diagnostics. It captures camera frames (AVT Manta via Vimba), reduces 2D images to a 1D interference lineout, performs a multi-stage nonlinear fit, and reports beam size (`sigma`) and visibility in real time.

This document describes:
- the full runtime lifecycle,
- end-to-end data pipeline,
- architecture and threading model,
- all important modules, classes, and functions,
- configuration/state persistence,
- test coverage and what each test file is validating.

---

## 2. Repository Structure

Top-level files:
- `/home/runner/work/SRIpy/SRIpy/main.py` — application entrypoint and global exception handling.
- `/home/runner/work/SRIpy/SRIpy/README.md` — project introduction and usage.
- `/home/runner/work/SRIpy/SRIpy/requirements.txt` — runtime/test dependencies.
- `/home/runner/work/SRIpy/SRIpy/sri_config.json` — persisted runtime defaults.
- `/home/runner/work/SRIpy/SRIpy/pytest.ini` — pytest discovery and marker configuration.

Main packages:
- `analysis/` — fitting engine and analysis worker thread object.
- `core/` — acquisition orchestration, burst worker, config and data models.
- `hardware/` — camera abstraction and Vimba-backed Manta driver + camera I/O thread.
- `gui/` — Qt main window, controller, and widgets.
- `utils/` — shared image preprocessing helpers.
- `test/` — unit/integration-style tests and fixtures.

---

## 3. High-Level Architecture

SRIpy uses a layered architecture with explicit producer/consumer threading:

1. **GUI Layer (`gui/`)**
   - Displays images/lineouts/fit and controls settings.
   - Emits user actions (start live, set exposure, ROI drag, burst, save).

2. **Controller Layer (`gui/controllers/interferometer_controller.py`)**
   - Mediates UI ↔ model.
   - Converts widget state into model updates.
   - Receives model signals and updates visual components.

3. **Model/Orchestration Layer (`core/acquisition_manager.py`)**
   - Owns current state (ROI, background, physics params, last outputs).
   - Manages camera and analysis threads.
   - Applies image preprocessing, auto-centering, analysis dispatch.

4. **Hardware Layer (`hardware/`)**
   - `MantaDriver`: actual Vimba camera operations.
   - `CameraIoThread`: command queue + frame acquisition loop.

5. **Analysis Layer (`analysis/`)**
   - `InterferenceFitter`: 4-stage fit algorithm.
   - `AnalysisWorker`: executes fit in separate thread with lock protection.

6. **Persistence/Data Layer (`core/data_model.py`, `core/config_manager.py`)**
   - Save dataset JSON/NPY, MATLAB `.mat`, burst summary.
   - Load/save app configuration.

---

## 4. Full Runtime Lifecycle

### 4.1 Application Startup

`main.py`:
1. `setup_logging()` configures global logging.
2. Sets pyqtgraph image axis order (`row-major`).
3. Creates `QApplication`.
4. Installs `sys.excepthook = handle_exception` to convert uncaught exceptions into GUI dialogs.
5. Constructs:
   - `InterferometerView` (GUI shell),
   - `InterferometerController` (wires view/model).
6. Shows the window and starts Qt event loop.

### 4.2 Controller Initialization

`InterferometerController.__init__`:
1. Instantiates `AcquisitionManager`.
2. Connects all View↔Controller↔Model signals.
3. Calls `self.model.initialize()`:
   - connects to camera,
   - spins `CameraIoThread`,
   - starts analysis `QThread` (`AnalysisWorker`).
4. Applies config values to widgets.
5. Pushes widget state back into model (`_sync_all_params`).

### 4.3 Live Acquisition Lifecycle

- User toggles **Start Live** button.
- Controller calls `AcquisitionManager.start_live()`.
- Manager enqueues `CameraCommand.START_LIVE` to `CameraIoThread`.
- Camera thread calls `driver.start_stream()` and enters rapid `acquire_frame(timeout=0.1)` loop.
- Each frame emits `frame_ready` → manager `_process_live_frame`.

### 4.4 Frame Processing & Analysis Lifecycle

`AcquisitionManager._process_live_frame` for each frame:
1. Calls `process_roi_lineout(...)` for:
   - optional background subtraction,
   - transpose handling,
   - ROI integration into full 1D lineout,
   - saturation detection.
2. Caches `last_raw_image`, `last_lineout`.
3. Emits `live_data_ready` for display.
4. Optionally auto-centers ROI based on lineout peak.
5. Throttles/guards analysis submission using:
   - `_analysis_busy`,
   - `_analysis_timeout_s`,
   - request IDs.
6. Emits `_request_fit(full_lineout, roi_hint, req_id)` to `AnalysisWorker`.

`AnalysisWorker.process_fit`:
1. Acquires `_fitter_lock` to keep physics constants stable.
2. Executes `InterferenceFitter.fit(...)`.
3. Emits `result_ready(fit_result, fit_x, req_id)`.

Manager `_handle_fit_result`:
- drops stale results by request ID,
- resets busy flags,
- caches `last_fit_result`, `last_fit_x`,
- emits `fit_result_ready` to controller.

Controller `_update_stats`:
- plots fit,
- updates numeric labels (visibility, sigma, raw vis, intensity),
- updates saturation/file status,
- appends sigma to history when relevant.

### 4.5 Burst Lifecycle

- User clicks burst button.
- Controller disables interactive controls and calls `start_burst(n)`.
- `CameraIoThread` handles `START_BURST`:
  - pauses live stream,
  - creates new QThread + `BurstWorker`,
  - routes acquired frames into burst queue.
- `BurstWorker.run_burst()`:
  1. Captures up to `n` frames from queue,
  2. Extracts/crops lineouts,
  3. Fits each frame,
  4. Computes aggregate statistics in `BurstResult`.
- Results/progress are emitted back to controller.
- UI re-enabled; completion dialog summarizes beam/visibility/intensity/physics.

### 4.6 Static File Lifecycle

Controller `_handle_load_file`:
1. Stops live if running.
2. Lets user select image or `.mat`.
3. Calls `AcquisitionManager.load_static_frame(path)`.

`load_static_frame`:
- For `.mat`: attempts broad key search for image data + metadata extraction for wavelength/slit/distance (with unit heuristics).
- For image formats: loads with OpenCV, converts color to grayscale by averaging channels.
- Sets static mode and processes through normal `_process_live_frame` pipeline.

### 4.7 Save Lifecycle

Controller provides:
- `_save_dataset()` → `DataManager.save_dataset(...)`
  - creates timestamped folder,
  - saves `raw_image.npy`,
  - saves JSON metadata/results.
- `_save_matlab()` → `DataManager.save_matlab(...)`
  - writes MATLAB-compatible struct with image/result/metadata.

### 4.8 Shutdown Lifecycle

When main window closes:
1. Controller `cleanup(event)` saves current config.
2. Calls manager `shutdown()`:
   - stops live,
   - stops analysis thread,
   - sends camera thread shutdown command,
   - closes driver resources.

---

## 5. Data Pipeline (End-to-End)

### Input
- Camera frames (`uint16`) from `MantaDriver.acquire_frame()`.

### Preprocessing (`utils/image_utils.process_roi_lineout`)
- squeeze and cast to `float32`.
- optional background subtraction and clipping to nonnegative.
- saturation flag (`max >= threshold`).
- ROI integration:
  - normal mode: sum selected rows over axis=0 (horizontal lineout),
  - transpose mode: select columns and sum over axis=1 (vertical lineout).
- returns `(display_image, full_lineout, is_saturated)`.

### Fit Input Selection
- Manager computes ROI display limits and provides `roi_hint` width.
- Full lineout is still passed to fitter; fitter recenters region around detected peak for stability.

### Fit Engine (`analysis/fitter.py`)
`InterferenceFitter.fit` stages:
1. **Signal sanity + smoothing**.
2. **Stage 0** Gaussian fit (center estimate).
3. **Stage 1** Sinc² envelope fit on full data.
4. **Stage 2** FFT for fringe frequency estimate.
5. **Stage 3** local sine fit for refined frequency/phase.
6. **Stage 4** bounded full interference model fit.

Outputs `FitResult` with:
- `success`, `visibility`, `raw_visibility`, `sigma_microns`,
- `fitted_curve`, `fit_x`, parameter dictionary, optional covariance/errors,
- peak/valley indices and max/min intensity.

### Physics Conversion
`calculate_sigma(visibility)`:
- clamps visibility to avoid domain errors,
- computes
  \[
  \sigma = \frac{\lambda L}{\pi d} \sqrt{0.5\ln(1/V)}
  \]
- returns meters (converted to microns in result payload).

### Output to UI
- live image and lineout plotted in `LiveMonitorWidget`.
- fit overlay + peak/valley markers updated.
- stats labels and history trend updated.

### Output to Disk
- JSON, NPY, MATLAB `.mat`, burst JSON summaries.

---

## 6. Core Domain Models

### `core/data_model.py`
- `ExperimentMetadata`: exposure/gain/physics + UTC timestamp.
- `ExperimentResult`: visibility/sigma/intensity + lineout/fit arrays + saturation flag.
- `BurstResult`: frame count, mean/std metrics, per-frame histories.
- `NumpyEncoder`: serializes numpy scalars/arrays and datetimes.
- `DataManager`: save helpers:
  - `save_dataset`,
  - `save_matlab`,
  - `save_burst`.

### `core/config_manager.py`
- `ConfigManager.DEFAULT_CONFIG`: camera, analysis, burst, physics, ROI defaults.
- `load()`: deep-merges file into defaults, with fallback to defaults on errors.
- `save(...)`: writes full config JSON.
- `_deep_update(...)`: recursive dict merge.

---

## 7. Module-by-Module Reference

## 7.1 `analysis/`

### `analysis/fitter.py`
- `FitResult` dataclass: canonical fit result schema.
- `InterferenceFitter`:
  - model helpers: `_gaussian`, `_sinc_sq_envelope`, `_sine_model`, `_full_interference_model`.
  - `_calculate_raw_contrast`: peak/valley-based raw visibility estimator.
  - `fit(...)`: robust multi-stage fitting routine.
  - `calculate_sigma(...)`: visibility→beam-size conversion.

### `analysis/analysis_worker.py`
- `AnalysisWorker(QObject)`:
  - signal: `result_ready(result, x, req_id)`.
  - `process_fit(...)`: threaded fit execution with lock, exception-safe fallback.

### `analysis/__init__.py`
- empty package marker.

## 7.2 `core/`

### `core/acquisition_manager.py`
Central orchestration class: `AcquisitionManager(QObject)`.

Key responsibilities:
- load/apply config,
- own runtime state (ROI, transpose, background, saturation, cached outputs),
- start/stop camera I/O and analysis threads,
- preprocess incoming frames,
- auto-center ROI,
- dispatch fit jobs with timeout/de-duplication,
- static file loading and metadata extraction,
- graceful shutdown.

Important signals:
- `live_data_ready`, `fit_result_ready`, `roi_updated`, `saturation_updated`,
- `background_ready`, `live_state_changed`,
- burst and error signals.

### `core/acquisition.py`
- `BurstWorker(QObject)`:
  - captures N frames from queue,
  - converts and preprocesses lineouts,
  - fits each frame,
  - emits progress and final `BurstResult`.

### `core/data_model.py`
- Persistence model dataclasses and save helpers (detailed above).

### `core/config_manager.py`
- Config defaults, loading, deep-merge, saving (detailed above).

## 7.3 `hardware/`

### `hardware/camera_interface.py`
- `CameraInterface(ABC)` defines contract:
  `connect`, `exposure`, `gain`, `acquire_frame`, `close`.

### `hardware/manta_driver.py`
- `MantaDriver(CameraInterface)` for Vimba camera operations.

Notable behavior:
- Connects to camera by ID or first discovered.
- Configures packet size, trigger mode/source, trigger software feature, pixel format.
- Supports two acquisition modes:
  - streaming queue mode,
  - snapshot mode (`_acquire_single_snapshot`).
- Thread/stream safety via `_stream_lock` and `_operation_lock`.
- Cleans up stream and Vimba contexts robustly.

### `hardware/camera_io_thread.py`
- `CameraCommand` enum for command queue protocol.
- `CameraIoThread(QThread)` handles:
  - command dispatch (start/stop live, set exposure/gain, capture BG, burst, shutdown),
  - acquisition loop while live/burst active,
  - burst thread lifecycle and cleanup,
  - resuming live after burst if needed.

## 7.4 `gui/`

### `gui/main_window.py`
- `InterferometerView(QMainWindow)`:
  - creates tabbed center area (live monitor + history),
  - adds control panel sidebar,
  - emits `close_requested` on close event.
- Sets `OMP/MKL/NUMEXPR` threads to 1 to avoid over-threading.

### `gui/controllers/interferometer_controller.py`
- `InterferometerController(QObject)`:
  - full signal wiring between UI and model,
  - applies config to controls,
  - synchronizes physics/ROI/camera settings,
  - handles live/burst/background/load/save flows,
  - updates UI state labels, plots, and history,
  - persists settings on exit.

### `gui/widgets/control_panel.py`
- `ControlPanelWidget(QWidget)` provides all controls and display labels.
- Emits typed Qt signals for user actions.
- Includes camera settings, physics settings, ROI actions, burst settings, save/load actions.

### `gui/widgets/live_monitor.py`
- `LiveMonitorWidget(QWidget)`:
  - image display (pyqtgraph),
  - lineout + fit curves,
  - peak/valley scatter overlays,
  - interactive ROI regions (rows + fit width),
  - automatic axis re-range logic.

### `gui/widgets/history_widget.py`
- `HistoryWidget(QWidget)`:
  - fixed-length rolling sigma history,
  - ignores non-finite values,
  - plots only valid points.

### `gui/controllers/__init__.py`
- empty package marker.

## 7.5 `utils/`

### `utils/image_utils.py`
- `process_roi_lineout(...)`: shared preprocessing primitive used by live and burst paths.

---

## 8. Configuration and State

Primary config file: `/home/runner/work/SRIpy/SRIpy/sri_config.json`.

Sections:
- `camera`: exposure/gain/transpose/background/saturation threshold.
- `analysis`: signal threshold, autocenter threshold, timeout.
- `burst`: default frame count.
- `physics`: wavelength, slit separation, distance.
- `roi`: vertical integration bounds and fit-width bounds, autocenter toggle.

State behavior:
- Config loaded at manager initialization.
- UI populated from config, then model synced from UI.
- On app close, controller snapshots current UI/model state and saves config.

---

## 9. Concurrency and Threading Model

Threads used:
1. **Qt GUI main thread**: widgets, user interaction.
2. **CameraIoThread**: command handling + frame acquisition loop.
3. **Analysis QThread**: `AnalysisWorker.process_fit`.
4. **Burst QThread** (transient): `BurstWorker.run_burst`.

Safety mechanisms:
- `_fitter_lock` protects mutable fitter physics parameters during fit.
- Request IDs in manager prevent stale fit result application.
- Busy/timeout logic avoids analysis pile-up.
- Burst IDs in camera I/O thread avoid old completion signals corrupting new burst session.

---

## 10. Error Handling Strategy

- `main.py` installs global exception hook that logs traceback and displays critical dialog.
- Driver methods have broad exception capture with contextual logs.
- Analysis worker returns failed `FitResult` on exceptions.
- Burst path emits error signals and performs cleanup.
- Config load failure falls back to defaults with warning.
- File-load path raises user-facing errors for unsupported/unreadable files.

---

## 11. Testing Layout and Coverage Intent

`pytest.ini` defines:
- discovery in `test/`,
- markers (`unit`, `integration`, `hardware`, `gui`, etc.),
- strict marker and warnings behavior.

### Test Files
- `/home/runner/work/SRIpy/SRIpy/test/conftest.py`
  - reusable mock driver/fitter fixtures and synthetic data fixtures.

- `/home/runner/work/SRIpy/SRIpy/test/test_fitter.py`
  - validates fitter parameter recovery, frequency lock behavior, noise robustness, edge cases.

- `/home/runner/work/SRIpy/SRIpy/test/test_physics.py`
  - integration-like physics consistency test for sigma recovery from synthetic data.

- `/home/runner/work/SRIpy/SRIpy/test/test_integration_examples.py`
  - mock-based pipeline tests for frame→lineout→fit and basic lifecycle checks.

- `/home/runner/work/SRIpy/SRIpy/test/test_config_manager.py`
  - ensures no shared nested state via deep copy.

- `/home/runner/work/SRIpy/SRIpy/test/test_acquisition.py`
  - burst worker progress semantics and no-frame completion behavior.

- `/home/runner/work/SRIpy/SRIpy/test/test_main_window.py`
  - logic-oriented checks for ROI/saturation/annotation/null-check expectations.

- `/home/runner/work/SRIpy/SRIpy/test/test_cam.py`
  - hardware camera listing utility-style test (requires Vimba/camera).

- `/home/runner/work/SRIpy/SRIpy/test/test_driver.py`
  - manual-style hardware visualization flow (requires camera/display).

- `/home/runner/work/SRIpy/SRIpy/test/test_analysis.py`
  - script-like example of connect→acquire→lineout→fit→plot.

---

## 12. Important Feature Summary

1. **Live acquisition with responsive UI** via decoupled camera and analysis threads.
2. **Robust multi-stage fitting** tuned to stabilize center/frequency/visibility extraction.
3. **Auto-centering ROI** behavior based on dominant signal peak and mode-aware axis semantics.
4. **Burst mode** with buffered frame capture and post-analysis statistics.
5. **Static file analysis** with support for common image formats and `.mat` metadata extraction.
6. **Data export** to both JSON/NPY and MATLAB `.mat` formats.
7. **Configuration persistence** and runtime restoration.
8. **Saturation/background handling** integrated in shared preprocessing.

---

## 13. External Dependencies and Their Roles

From `requirements.txt`:
- `numpy`, `scipy` — numeric processing and fitting.
- `vmbpy` — Allied Vision camera SDK binding.
- `PyQt6`, `pyqtgraph` — GUI framework and plotting.
- `matplotlib` — optional colormap/plotting utilities.
- `opencv-python` — loading static image files in controller/model path.
- `pytest` — test runner.

---

## 14. Practical Execution Notes

- Runtime assumes Vimba SDK/driver stack is installed and camera is reachable.
- Some tests are hardware/manual-script oriented; not all tests are pure headless unit tests.
- The code supports both live hardware and offline static-file inspection workflows.

