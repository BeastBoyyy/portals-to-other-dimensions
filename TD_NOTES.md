# Earlier technical notes (TouchDesigner stage of the project; partly superseded, see README.md)

Make a "frame" with both hands (thumb + index of each hand form an L). A white-outlined quad is pinned to the four fingertips;
outside it you see the live camera, inside it a black "other dimension" where your body is a cloud of white/pink specks
(or a lit asteroid). Priorities, in order: **tracking correctness, smoothness, latency**, then looks.

Read `REPORT.md` for measured numbers (and what is NOT measured) and `DECISIONS.md` for why things are the way they are.

## Run it (no TouchDesigner, no key presses)

```bash
.venv\Scripts\python app\portal_app.py          # add --hud for the debug overlay, --cell 4 for fewer particles
# --engine tasks = old stock MediaPipe tracker; --tta 1..5 = crops averaged per hand; --record-video = OPT-IN local recording of the camera for offline tests
# tracking runs in its own process at the webcam rate (30 Hz here); for a smoother picture set Windows Settings > Graphics > this python.exe > High performance
```
Open your hand to open the box, pinch to close it, twist your wrists to step through the effects (frosted glass, moire, slit scan, pixel sort, CRT, datamosh), move fast to make it glitch. Full guide: **[GESTURES.md](GESTURES.md)**.
TouchDesigner is optional (see DECISIONS D16); the `td/` network predates the gesture controls.

## What is verified, and how

| Component | Status |
|---|---|
| `handcore/` tracking library (validation, identity, filters, quad, presenter) | **verified by tests** (`pytest`, 60+ tests) and measured on synthetic clips with exact ground truth |
| Real MediaPipe HandLandmarker on sample images | **verified by tests** (hands found, no hallucination on 3 non-hand images, real skeletons pass the gates) |
| Wire protocol, shared-memory layout, self-check maths, script syntax | **verified by tests** |
| GLSL shaders | **verified offline**: compile + render with TD's symbols (`tools/shader_preview.py`, `fixtures/preview/*.png`) |
| Sidecar (`adapters/mediapipe_sidecar.py`) | **verified by an end-to-end test**: fake-TD shared-memory writer -> real MediaPipe LIVE_STREAM -> HandCore -> UDP packets (`tests/test_sidecar_e2e.py`); inference benchmarked at ~14-17 ms/frame CPU. The real TD frame writer and a live webcam **need user verification** |
| TouchDesigner network (`td/`), 60 fps, frame times, latency | **needs user verification in TD** (`td/MANUAL_STEPS.md`, `td/selfcheck.py`) |
| Real-camera accuracy (99 % target) | **NOT MEASURED** (needs your clips + labels: see "Measure your own accuracy") |

## Quickstart (Windows/macOS/Linux; sidecar Python 3.10-3.14)

```bash
cd portal
python -m venv .venv
.venv\Scripts\python -m pip install mediapipe numpy pyyaml pytest python-osc moderngl   # macOS/Linux: .venv/bin/python
.venv\Scripts\python -m pytest -q                      # everything that can run without TD or a webcam
.venv\Scripts\python tools/setup_check.py              # camera fps / light / blur check (needs a webcam)
.venv\Scripts\python tools/export_td_config.py         # config.yaml -> td/config.json
```

Then build the network in TouchDesigner (`td/MANUAL_STEPS.md`), start `python adapters/mediapipe_sidecar.py`, make the frame pose.
Models (`models/hand_landmarker.task`, `models/selfie_segmenter.tflite`) are downloaded from `storage.googleapis.com/mediapipe-models/...`
(commands in `MANUAL_STEPS.md`); they ship in the folder if you got this repo from the author's machine.

No TouchDesigner? `python adapters/mediapipe_sidecar.py --source camera --preview` runs the full tracker on a webcam and draws the
portal outline in an OpenCV window (this is the fastest way to check tracking before touching TD).

## Architecture

```
TD camera --(30 Hz, shared memory, TD-stamped)--> sidecar: MediaPipe HandLandmarker (LIVE_STREAM) + selfie mask
                                                          |  Frame(t, hands)
                                                          v
                                       handcore.HandCore: ingest(mirror) -> gates -> identity -> filter -> quad
                                                          |  UDP struct (corners, velocity, held, telemetry)
TD: UDP In DAT -> portal_state -> Script CHOP 'presenter' (handcore.predict.Presenter @ 60 Hz) -> Null CHOP 'portal_null'
                                                          |  q0..q3, alpha, parallax, scene_mix ...
        GLSL TOPs: scene_a (particles) / scene_b (asteroid) -> scene_blend -> blur(bloom) -> composite -> HUD -> window
```

`handcore` has no TouchDesigner/MediaPipe imports and runs identically from live data, JSONL replay or the simulator.

## Config

Everything tunable is in `config.yaml` (documented inline; every feature has an `enable` switch). Workflow:
edit YAML -> `python tools/export_td_config.py` -> rebuild/restart. Key knobs:

| Want | Change |
|---|---|
| less jitter / more lag | `filter.min_cutoff`, `filter.beta`, `filter.palm_*` (see `tools/sweep.py tune*`) |
| fewer false hands | keep `gates.person_mask: true`; raise `gates.min_score` / `acquire_n` |
| mirror display | `camera.mirror` (one place; the TD Flip TOPs follow it after `build_network`) |
| particle count / speed | `td/config.json` -> `quality_cell_px` (4 = 57,600 points at 1280x720) |
| per-user fingertip constraint | `constraint.enable: true` (+ `calibrate_frames`); off by default, see DECISIONS D9 |

## Camera setup (matters more than any code)

- 1280x720, MJPEG, 30 fps minimum (60 if the camera has it). `tools/setup_check.py` prints the negotiated mode and the *measured* fps.
- **Lock exposure, focus and white balance**; short shutter (~1/120 s) with more light instead of more gain. Auto-exposure hunting
  makes landmarks shimmer. (`--lock-exposure` tries it via DirectShow; otherwise use the camera vendor app.)
- Face the main light. A window *behind* you silhouettes your hands and hurts tracking.
- Keep both hands fully inside the frame; hands cut by the frame edge are rejected on purpose.

## Self-check inside TD (2 minutes)

```python
op('/project1/portal/selfcheck').module.start()      # guided, ~45 s;  start(soak_s=60) for the full soak
```
Prints PASS/FAIL for camera fps, tracker fps, tracker-to-render age, rest jitter (hold hands still 5 s), corner kinks while
crossing hands (a slot swap shows as a >100 px kink), render fps / 1st-percentile fps / frames over 20 ms, then a plain verdict and next step.

## Measure your own accuracy (the 99 % number)

```bash
python tools/record_clip.py still --seconds 45        # also: slow fast crossing reentry nohands onehand lowlight backlit tilt occluded
python tools/evaluate.py --jsonl clips/still.jsonl    # stability: 1-frame jumps, portal-valid %
python tools/label_tool.py clips/slow.mp4             # click 4 fingertips on ~200 random frames
python tools/evaluate.py --jsonl clips/slow.jsonl --labels clips/slow.labels.json   # corner accuracy (<=6 px at 1280x720) and outliers
```
Latency: `tools/latency_test.md` (phone slow-motion). Sweeps: `python tools/sweep.py tune|tune_palm|tune_floor|tune_gate|tune_spike|compare|ablate`.

## Repo layout

```
config.yaml  DECISIONS.md  REPORT.md  ABLATION.md        handcore/   one_euro kalman validate identity constraints predict quad pipeline config types
adapters/    mediapipe_sidecar.py (Option B/C front half)  protocol.py (wire formats)  td_plugin_bridge.py (Option A glue, text)
td/          build_network.py  selfcheck.py  MANUAL_STEPS.md  config.json  scripts/*.py  shaders/*.glsl
tools/       evaluate sweep synth setup_check record_clip label_tool extract_landmarks bench_sidecar shader_preview export_td_config latency_test.md
tests/       pytest (unit, replay, property, real-MediaPipe, shaders)     fixtures/  sample images, real landmarks, shader previews
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Corners wander / jitter at rest | exposure hunting or low light (`setup_check.py`); then `filter.*` |
| Portal lags on fast moves | `filter.beta` / `palm_beta` too low; check `age_ms` on the HUD; camera latency (try 60 fps MJPEG) |
| Portal appears with no hands / on a poster | `gates.person_mask` off or segmenter model missing; sidecar console says `mask=off` |
| Hands swap when crossing | keep hands at different depths; raise `identity.swap_margin`; record the `crossing` clip and evaluate |
| `GPU delegate unavailable` | normal on Windows pip builds (measured here: CPU only); CPU inference is ~15 ms, enough for 30-60 fps |
| TD frame drops | `quality_cell_px` 4 -> 6, hide HUD (`H`), check Performance Monitor (`F2`) |
| Reference video missing | it was not reachable when this was built; all numbers come from the simulator and the sample images. Record your own clips |

## Sample images

`fixtures/images/*.jpg` are sample images fetched from the public MediaPipe assets bucket (`storage.googleapis.com/mediapipe-assets/`), used only as local test fixtures for the real detector; `fixtures/preview/*.png` are renders of the shaders from them.
