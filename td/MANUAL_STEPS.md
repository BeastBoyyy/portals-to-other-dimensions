# TouchDesigner: manual steps

A `.toe` cannot be authored outside TouchDesigner, so the delivery is `build_network.py` (creates and wires every operator)
plus this list. **Nothing in `td/` has been run inside TouchDesigner by the author** (no TD in the build sandbox):
treat the first run as a checkout and use `selfcheck.py` as the judge. Shaders *were* compiled and rendered offline
(`tools/shader_preview.py`, `tests/test_adapters_td.py`) against TD's GLSL TOP symbols; the Python logic they rely on
(protocol, presenter, analysis) is unit-tested.

Tested-against assumptions (verify, TD changes): TouchDesigner 2025 official build 2025.32460 (current official per
docs.derivative.ca at the time of writing), non-commercial licence (1280x720 cap is fine), Python 3.11 inside TD with numpy.

## 0. One-time setup (outside TD)

```bash
cd portal
python -m venv .venv
.venv\Scripts\python -m pip install mediapipe numpy pyyaml pytest python-osc moderngl
# models (also done by the author; re-run if models/ is empty)
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -L -o models/selfie_segmenter.tflite https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite
.venv\Scripts\python tools/export_td_config.py
```

## 1. Build the network

1. Start TouchDesigner, new project, save it somewhere (e.g. `portal.toe`).
2. Open the Textport (`Alt+T`) and run (adjust the path):
   ```python
   REPO = r'D:\TDMP\portal'
   exec(open(REPO + r'\td\build_network.py').read())
   ```
3. It prints either `all parameters applied.` or a list under *Set these by hand*. Apply that list using section 3 below.
   Re-running the script is safe (idempotent).
4. Optional: save the network as a reusable component: `op('/project1/portal').save(REPO + r'\td\portal.tox')`.
   Drag that `.tox` into any project; it is self-contained except for the repo path stored in `config_json` (re-run
   `tools/export_td_config.py` and rebuild if the repo moves).

## 2. Run

1. `python adapters/mediapipe_sidecar.py` (start it after TD: it waits for the `portal_frame_v2` shared memory that TD creates).
2. In TD: select `/project1/portal/cam` and choose your camera in *Device* (Video Device In TOP).
3. Open the output window: select `win` and press its *Open* pulse, or `F1` for Perform mode on the main display.
4. Keys (window focused): `1` particle self-view, `2` asteroid, `H` HUD on/off.
5. Verify: HUD shows `tracker ~30 fps`, `infer ~15 ms`, `render ~60 fps`. Then run the self-check (README).

## 3. Parameters TD may reject (set by hand if listed by the build script)

TD renames parameters between builds, so each assignment in the script is wrapped; anything it could not set is printed. Reference values:

| Operator | Parameter | Value |
|---|---|---|
| `cam` (Video Device In TOP) | Device / Resolution | your camera / 1280 x 720 (custom) |
| `cam_small` (Resolution TOP) | Output Resolution | Custom 640 x 360 |
| `cam_disp`, `mask_disp` (Flip TOP) | Flip X | on when `camera.mirror: true` (the ONE display mirror) |
| (none) | state packets arrive on UDP 127.0.0.1:9000 via a socket thread in `portal_state`; there is no UDP In DAT | allow Python through the firewall prompt if Windows asks |
| `frame_writer` (Execute DAT) | Active / Frame End | on / on |
| `presenter` (Script CHOP) | Callbacks | `presenter_cb` |
| `mask_in` (Script TOP) | Callbacks | `mask_cb` |
| `keys` (Keyboard In CHOP) | Keys | `1 2 h` (channels `k1 k2 kh`; if TD names them differently edit `td/scripts/keys.py`) |
| each GLSL TOP | Pixel Shader | the matching `*_pixel` DAT; Output Resolution Custom 1280x720; Format 16-bit float RGBA |
| GLSL TOPs | Vectors page | uniform names/expressions exactly as in `build_network.py` (`uQA`, `uQB`, `uP1`, `uP2`, `uA`, `uB`, `uMix`) |
| `hud_level` (Level TOP) | Opacity | `op('portal_null')['hud']` |
| `win` (Window COMP) | Window Operator / Size | `out` / 1280 x 720 |
| Project | Cook rate / Real Time | 60 / on |

Project settings you must check by hand: Timeline FPS = 60, *Realtime* on; vsync on in Perform mode (Preferences, Display).

## 4. Things this build deliberately does not do

- No `.toe` / `.tox` binary is shipped (cannot be produced outside TD).
- The optional both-fists scene switch (spec 6.5 stretch) is not implemented; scenes switch with keys `1`/`2`.
- No NASA texture for the asteroid; it is procedural (`scene_b.glsl`). To use a public-domain image (NASA Image and Video
  Library content is generally not copyrighted, but check each item's page), wire it into `scene_b` input 0 and set `uB.x = 1`.
- Option A (the `mediapipe-touchdesigner` plugin) is not wired; the sidecar replaces it. See `DECISIONS.md` D1.

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Sidecar prints `waiting for TouchDesigner shared memory` forever | `frame_writer` is not active, or it errored: check the Textport; make sure `cam_small` has a cooking input |
| `numpyArray(delayed=True)` error | older build: remove the `delayed=True` argument in `frame_writer` (adds a GPU stall) |
| HUD `tracker fps` 0 | port 9000 already in use (old UDP In DAT or another program), or the sidecar `--port` differs from `td/config.json` |
| Portal never appears | hold BOTH hands in the L pose, thumbs and index fingers fully visible, hands well inside the frame |
| Portal appears on a poster | person mask gate off: the sidecar needs `models/selfie_segmenter.tflite` (the sidecar console prints `mask=on/off` at start) |
| Shader compile error | the Textport/GLSL TOP info shows the line; `tools/shader_preview.py` shows the same file compiles with TD's symbols, so suspect a renamed uniform |
| Left/right hand corners jump when crossing | run the self-check *cross* phase; raise `identity.swap_margin` |
