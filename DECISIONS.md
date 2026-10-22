# DECISIONS (fixed constraints once written)

Format: **what** - considered - why - rejected. Numbers come from `REPORT.md`, `ABLATION.md`, `bench_results.json` and
`python tools/sweep.py compare|tune*` run on this machine (Windows 11, Python 3.14.3, mediapipe 1.1.1, NVIDIA RTX 4050 laptop + Intel iGPU).
**Synthetic-clip numbers describe handcore under an assumed noise model, not real MediaPipe accuracy** (see REPORT.md).

## Environment findings (Phase 0)

- The prompt assumed macOS/Apple Silicon. The actual machine is **Windows 11**; everything is written cross-platform
  (Spout/Syphon is not used; shared memory + UDP work on both). `mediapipe` 1.1.1 installs on Python 3.14 and exposes
  `vision.HandLandmarker` (options: `running_mode`, `num_hands`, `min_hand_detection_confidence`, `min_hand_presence_confidence`,
  `min_tracking_confidence`, `result_callback`) and `vision.ImageSegmenter`. Checked in the installed package, not from memory.
- TouchDesigner official build per docs.derivative.ca: 2025.32460 (March 2026). TD and a webcam were not available:
  all TD work is "needs user verification". The `mediapipe-touchdesigner` repo README (fetched) documents a Chromium/WebAssembly
  pipeline, a 720p input cap and extra Web Render TOP frame delay; it does not document CHOP channel names.
- **The reference video was not reachable** (`/root/.claude/...` does not exist on this machine). No clip derived from it exists.
  The mosaic-blur remark in the brief is therefore moot; nothing was reproduced from it.

## D1 Landmark source: Option B (Python sidecar), TD owns the camera
- Considered: A (in-TD plugin), B (sidecar + shared memory), C (sidecar owns camera + Spout/Syphon).
- Measured (`tools/bench_sidecar.py`, 1280x720, CPU): HandLandmarker IMAGE 13.6 ms, VIDEO 16.0 ms, **LIVE_STREAM paced 30 fps 17.3 ms mean / 18.9 p95**
  (capacity ~58 fps). GPU delegate: unavailable in this Windows pip build ("GPU processing is disabled in build flags"); the sidecar falls back to CPU automatically.
  Option A: NOT MEASURED (cannot run outside TD).
- Chosen B: first candidate that can be benchmarked and meets the budget (17 ms << 33 ms), keeps inference off TD's cook thread, and needs no Spout/Syphon library.
  C rejected (extra platform-specific dependency, camera exclusivity makes TD's own preview awkward); A kept as an *option*
  (`adapters/td_plugin_bridge.py`, unverified) because HandCore is source-agnostic.

## D2 Clock: capture time is stamped by TouchDesigner
- TD writes `absTime.seconds` into the frame header; the sidecar passes it through unchanged; the Presenter runs on the same `absTime.seconds`.
  No clock sync problem exists. Rejected: `time.perf_counter` in two processes (undefined shared reference per Python docs).

## D3 Transport: raw binary UDP struct + shared memory, not OSC
- Frame TD->sidecar via `multiprocessing.shared_memory` (640x360 RGB, seq written last, torn reads dropped). State sidecar->TD as one fixed 104-byte datagram per frame (`adapters/protocol.py`).
- Why: nothing is parsed in TD's hot path (no DAT cells), fewer moving parts than OSC In CHOP channel naming. Mask goes back through a second shared-memory block.

## D4 Mirroring is decided once: `camera.mirror`
- Applied in `HandCore.ingest` (landmarks, world x, handedness label swap, person mask) and by the TD Flip TOPs (`cam_disp`, `mask_disp`) from the same config. Tested both ways.
- Slot identity never uses the label as truth (weak vote, weight 0.6 x |vote|); a unit test flips labels with p=0.5 and requires 0 swaps.

## D5 Filter design: palm centre and tip-offset-from-palm are filtered separately (One Euro each)
- Spec starting values on the tips (min_cutoff 1.0, beta 0.007, d_cutoff 1.0, `sweep.py compare`): jitter 0.18 px but **fast-clip accuracy 2.2 %, tilt 5.9 %, outliers 30.9 %** (lag ~85 px at 2700 px/s).
  Plain One Euro, tuned: acc fast 88.0, tilt 90.8, jitter 0.34. **Decomposed (final): acc fast 95.4, tilt 91.9, jitter 0.25, outliers 1.3 %.**
- Why it works: the palm centre (mean of 5 stable anchors, spec 4.5) is ~0.3 px noisy so it can follow with a high beta (no lag); the finger offset is the noisy,
  slow part and keeps heavy smoothing, with a speed dead-zone (80 px/s) so derivative noise never opens the cutoff at rest.
- Final values: offset `min_cutoff 0.7, beta 0.3, d_cutoff 6, speed_floor 80`; palm `1.0, 0.5, 15, floor 20`; `tune`, `tune_palm`, `tune_floor` sweeps.
- Rejected: moving average (fast acc 1.4 %, ABLATION), no filter (jitter 0.76 vs 0.25), stacking a second low-pass (lag).

## D6 Thumb snaps: palm-anchor gate (constant-velocity prediction of the tip offset)
- A tip offset that deviates from its prediction by > 0.2 palm sizes (+ 1.0 x offset speed x dt) is a spike: follow the prediction, accept it as real only if it stays at the same new place > 3 frames; roll back
  if the "real" place turns out to be a snap. Ablation: without it **jitter 2.28 px vs 0.25, outliers 1.94 % vs 0.81 %**. `tune_spike` sweep: this point is on the Pareto front (jitter 0.25 / outlier 0.99 / acc 96.3 vs 0.07 / 1.1 / 96.2 and 0.07 / 1.9 / 94.9).
- Rejected: strength-scaling of the filter (the filter's own derivative absorbs the spike first); fixed (non-predictive) threshold (flags fast hand rotation as spikes: tilt acc 63 %).

## D7 Motion gate (gate 4)
- Kalman CV per palm centre (accel sigma 15000 px/s^2, meas 4 px), squared-Mahalanobis gate 16, plus deviation from prediction <= 0.25 palm + 0.5 a dt^2, plus a hard 12000 px/s cap.
  Huber-style down-weighting of far measurements. Ablation: no motion gate => jitter 1.91 px (teleport re-detections leak into the filters).

## D8 Identity (4.4)
- Brute-force assignment over <=2x2, cost = Mahalanobis/threshold + 2.0 x |log palm-size ratio| + label vote; **ambiguity rule**: one detection that fits both slots within margin 0.6 is assigned to nobody (both coast);
  swap hysteresis margin 0.6 / 4 frames; canonical slot on (re)acquisition = screen half. Ablation: naive x-order => **43 swaps, mean accuracy 91.8 %**; final: 0 swaps over 24 crossing/near-crossing clips (10 + 8 seeds in tests/sweeps).
- Found by test: with one detection during overlap, nearest-prediction assignment swapped hands; fixed by the ambiguity rule.

## D9 Kinematic tip constraint: implemented, **OFF by default**
- `constraint.enable: false`. Ablation "kinematic constraint ON": jitter 0.40 vs 0.25, outliers 0.97 vs 0.81, mean accuracy 96.75 vs 97.01. The anchor gate (D6) already handles what the simulator throws at the tip;
  the length bound has nothing extra to catch. Per spec ("if a feature does not measurably help, remove it") it is not on; the code stays (cheap, tested path, `calibrate_frames` hook) because real MediaPipe
  may violate bone lengths in ways the simulator does not model. Enable and re-run `ablate` on real clips to decide.

## D10 Gates 1-3, 5
- Gate 1 score >= 0.70; gate 2 geometry ranges set from real MediaPipe output on 5 real hands (palm length 1.31-1.67 palm widths, finger chains 0.81-1.66, thumb IP-tip 0.35-0.57) widened generously;
  gate 3 temporal acquire 3 / lose 6; gate 5 person mask **on by default** (selfie segmenter in the sidecar, cached 256x144, mirrored with the landmarks).
- Ablation: no gates => false-hand 100 % (flicker/random ghosts), jitter 11 px; no temporal gate => false-hand 2.2 %; **two persistent poster hands: 99.2 % false portal without the mask gate, 0.0 % with**.
- The geometry gate shows **no measurable effect in the simulator** (other gates reject first). Kept anyway because (a) it measurably rejects >90 % of garbage skeletons in `test_garbage_skeletons_rejected`, (b) real hands pass (test on real MediaPipe output),
  (c) it is cheap. This is the one place the "remove if no help" rule was consciously overridden; remove it if real clips show false rejects.

## D11 Presenter (4.7/4.8): latency-compensated damped constant-acceleration extrapolation + critically-damped error decay
- The spec clamp (1.5 camera periods) is applied to time *since arrival*; capture-to-arrival latency is compensated in full (otherwise the output sawtooths). Velocity = palm velocity + dead-zoned finger-relative velocity
  (noise would otherwise be extrapolated: rest jitter 0.56 px -> 0.25 px). Acceleration term halves the extrapolation error on oscillating motion (perfect-sine test: 59 -> 28 px median at blend 20 ms).
- A new sample that disagrees with the extrapolation is absorbed by a critically-damped spring on the *position and velocity* offset (C1-continuous: exponential position-only decay produced velocity kinks, 61 vs 16 pops).
- Measured effect (ABLATION): median render error vs ground truth with prediction **6.2 px slow / 85 px fast; without 18.1 / 119** (jitter 0.25 vs 0.14 px; pops 119 vs 71 - the price).
- Fade out 200 ms / in 100 ms; re-acquisition easing is scaled by the portal's opacity (an invisible portal needs no easing: this removed a 90 px trailing lag after re-entry).
- Rejected: pure position-decay blend; interpolation with a one-frame buffer (adds ~33 ms latency; spec asks for extrapolation).

## D12 Visuals
- Particles are a **fragment-shader point cloud** (one dot per grid cell, jittered; cell 4 px at 1280x720 = 57,600 points; `quality_cell_px` scales it), not instancing: one cheap pass, no CPU work, same visual.
  Rim = mask gradient; bloom = Blur TOP of the scene; glitch bands time-driven; parallax from quad centroid + tilt (`par_x/par_y`).
- Asteroid is procedural (Worley craters, hard side light): downloading a NASA image was not necessary and its licence would need per-item checking. A texture slot exists.
- Both-fists scene switch (6.5 stretch) not implemented. Tunnel (6.6) implemented in `composite.glsl` (inset quad + connecting edges), toggled by `uP1.w`.
- Shaders were compiled and rendered offline (moderngl) with TD's symbol set; previews in `fixtures/preview/`.

## D13 How "99 %" is defined and what was measured (see REPORT.md for the numbers)
- Corner accuracy = fraction of (hand-visible frame, corner) pairs within 6 px, counting frames with no valid portal as misses. "Hand-visible" = both hands fully inside the frame AND the ground-truth quad itself is convex/valid
  (a twisted ground-truth quad has no portal to draw).
- Result: **pooled 97.4 % (FAIL vs 99 %)**: still 99.6, slow 99.6, crossing 99.6, near-crossing 99.5, backlit 99.2, occluded 98.4, reentry 97.7, lowlight 93.0, **fast 95.4, tilt 91.9**. The fast/tilt clips peak at ~2700 px/s: 6 px = 2 ms of motion, so any
  smoothing that gives 0.25 px rest jitter costs accuracy there. This is a real trade-off, not an unfinished fix: raising `palm_beta`/`beta` moves points along the jitter/accuracy curve (see `tune_*`).
- Pops: second-difference excess > 20 px at 60 Hz (a kink that real motion does not explain; steady velocity error is not a pop). Re-entry pop = first-difference excess x portal opacity, window 0.6 s.
  Recovery "stable" = valid and within 12 px (not 6: the hand is moving fast when it re-enters) for 3 consecutive frames.
- Rest jitter: RMS of the >3 Hz band (FFT) of the 60 Hz *presenter output* on the `still` clip, including its snaps and teleports (not a cleaned subset).

## D14 Python in TouchDesigner is kept tiny
- Per-frame TD Python: read <= 1 packet, `Presenter.render` on ~30 floats, write ~45 CHOP channels. HandCore (0.68 ms/camera frame measured, incl. Kalman/gates) runs in the sidecar. Frame writer: one 640x360 readback at 30 Hz (`numpyArray(delayed=True)`).
- Garbage collection: HandCore allocates only tiny numpy arrays; no `gc.disable()` was added (not measured to be needed; add it around the Script CHOP if the self-check shows >20 ms spikes).

## D15 Whole-hand control: open / pinch, twist dial, speed (supersedes the first finger-raise version)
- First attempt (finger-raise selects the effect, box follows the raised finger, opening by hand distance) was WRONG per feedback on the reference clip: the box stays on the
  same fingers; opening is the whole-hand pinch <-> open; the effect changes with hand TWIST. Rebuilt; `handcore/gesture.py` deleted, replaced by `handcore/pose.py`.
- Measured on every frame of the clip (`tools/analyze_twist.py`, `tools/analyze_clip.py`): fingertip spread / palm size = 0.2-0.4 pinched (0-0.5 s, 8.3-8.5 s), 0.8-1.2 open.
  Portal openness = smoothstep(0.35, 0.80, min over both hands). Twist = angle between the palm normal (3D world landmarks) and the camera axis; a natural open palm reads 28-50 deg,
  strongly turned hands 70-88 deg. A 2D palm-width/length alternative was tried and rejected (ratio often > 1: unreliable).
- Dial sectors (54/62/70/77/84 deg, cross-fade 6 deg) put the natural open palm on particles+moon; deliberate twisting walks through glass, ASCII, bitmap, web, neon. The dial uses the
  MEAN twist of both hands (noisy estimates average out) smoothed 150 ms; each hand's own twist also pushes its side of the box deeper (3D twist of the box, `twist_depth`).
- Anchors are fixed: thumb tip, index tip, lid = mean(middle, ring). Small / closed boxes are valid (quad min edge 3 px, min area 0.01 %) so the portal grows out of the pinch.
- Reactivity: hand speed (palm centre velocity / frame width, 0.6 widths/s = 1.0) drives particle comet streaks, glass distortion + ripples, ASCII scramble, bitmap band sliding, neon hue rate and a
  global glitch (band shifts + RGB split) in the composite.
- Effects are new GLSL (td/shaders/fx_*.glsl): ASCII (Consolas glyph atlas, density ramp, rain, scan band), bitmap (Bayer 8x8 1-bit), glass (animated refraction field, chromatic aberration, frost blur,
  caustics), web (displaced scan lines + threads), neon (Sobel x3 scales, hue drift, thermal wash). Moon/asteroid/ghost kept as scenes; asteroid and ghost are not on the default dial.
- Not measured: how well a real person can hit each twist sector on a real webcam (noise +-10 deg). If the dial feels twitchy or sluggish, adjust `dial_edges` / `blend_deg` / `tau_twist_s`.

## D16 TouchDesigner vs. the standalone app
- Measured/experienced here: TD needed a free Derivative sign-in, a manual build step, and produced four integration bugs on first contact (Script TOP method name, UDP In DAT delivery,
  shared-memory sizing, camera exclusivity). The standalone app (`app/portal_app.py`: OpenCV + MediaPipe + moderngl/glfw, same GLSL files) opens the webcam directly, runs tracker + renderer in one
  process, needs no licence, and renders at >60 fps on the integrated GPU (the heavy asteroid pass only runs when selected).
- Recommendation: use the standalone app for this piece. Use TouchDesigner only if you want to author visuals node-by-node, route output to VJ/projection tools (Spout/NDI/Syphon), or
  combine with audio/MIDI/DMX. The `td/` network is kept but is NOT updated for gestures (it still has the original two scenes and keys); the shaders in `td/shaders/` are shared by both.


## D17 Precision and rate: two processes, full-resolution crop tracker (measured on this machine, 2026-10)
- Measured: the webcam delivers 30 fps at every mode tried (MJPG/YUY2, 360p-720p), so 30 Hz is the ceiling for tracking; the old in-process Tasks pipeline reached only ~20.5 Hz (live logs) because tracking, camera and GL shared one Python process (inside the app the new crop tracker fell to 13 Hz until it was moved out). Tracking now runs in its own process (`app/tracker_proc.py`, frames via shared memory, states via a pipe): 28-30 Hz.
- The MediaPipe Tasks palm detector now only FINDS hands; `app/roi_tracker.py` follows them with the same landmark network (unpacked from `hand_landmarker.task`, run through LiteRT) on rotated crops of the FULL 1280x720 frame, averaging `--tta` slightly different crops per hand (noise ~ 1/sqrt(K)). Tracks survive 4 weak frames (twists) instead of being re-detected; HandCore's gates still judge every hand.
- Label-free repeatability (std of landmarks across random crops, % of palm width, hand-tip median): RTMPose-m 4.9, MediaPipe Tasks 7.8, MediaPipe zoo-ONNX 10.2. RTMPose is the most repeatable but sits ~0.12 palm further out along the finger than MediaPipe (different tip convention) and no labelled set was available to say which is MORE ACCURATE, so it is NOT used live. Repeatability is not accuracy; absolute accuracy is NOT MEASURED.
- Found and fixed with the user's own recordings: a hand returning after a long coast kept the previous pose's filter and spike-gate state (filtered corners hundreds of px from the detected tips). `Slot.rearm` now starts clean. The person-mask dilation is now in units of a 256-wide mask (it swallowed the 64-wide synthetic mask).
- Opt-in `--record-video` saves the raw camera video locally so engines can be compared on real footage (coverage, jitter, twisted segments).


## D18 Effect set: six "image processing" effects (frosted glass, moire, slit scan, pixel sort, CRT, datamosh) (measured 2026-10-22, Intel iGPU)
- Chosen from the user's list. NOT built: point cloud and depth map (they need a depth source; the webcam gives none and no depth model was added, so a fake one would be dishonest).
- Anti-aliasing / smoothness decisions: moire is the analytic cosine of the phase difference of the screens (no aliasing, continuous in time); slit scan blends between camera frames (delay is fractional);
  pixel sort reads its place in a per-pixel odd-even sorting network (63 compare-exchanges, K=16 samples, interpolated); datamosh runs once per CAMERA frame (like a codec) with integer-pixel crisp copies.
- Performance facts found by measuring (offscreen draw ms incl. portal composite, Intel UHD, 1280x720): moire 6.7, slit scan 6.5, CRT 7.8, pixel sort 8.2 (half-res), glass 12.0, datamosh 15.2 (per camera frame). Live window fps (10 s each, webcam): moire/CRT/slit scan 144 (vsync), pixel sort 102, glass 91, datamosh 114 (before final tuning).
- Two real bugs found by measuring: (1) pixel sort with a dynamically indexed array was 355 ms/frame (scratch memory) -> unrolled sorting network, 12 ms; (2) writing a texture the GPU was still sampling (slit-scan history) stalled the driver: 63 fps -> 144 fps by double-buffering the history arrays.
- Previews and tests were silently showing the NEXT effect because a twist of 60 deg made the ratchet step; previews/tests now use the resting twist (35 deg).
- Image quality was judged on a compressed clip, not on the user's webcam: NOT MEASURED there.
