# Motion-to-photon latency test (physical, ~10 minutes)

Latency = time from a fingertip changing direction in the real world to the portal corner changing direction on the screen.
`td/selfcheck.py` only measures the *tracker -> render* part (`age_ms`); the camera exposure/readout/driver and the display
add more. This test measures all of it. Target: <= 80 ms. Report whatever you get.

## What you need
- A phone that records slow motion at 120 or 240 fps (any recent iPhone/Android).
- The running portal (sidecar + TD window, 60 fps, mirror as you normally use it).

## Procedure
1. Place the phone so ONE shot contains both your hands (in the frame pose) and the monitor showing the portal. Mount it still.
2. Start slow-motion recording (240 fps if available; note the fps you actually use: `F_slow`).
3. Make a clear, fast direction reversal with the left index finger: move it sharply right, stop, and sharply left (a "flick").
   Do this 8-10 times in 10 s, hands otherwise still in the frame pose.
4. Open the video in a frame-stepping player (VLC: `E` steps one frame; or any editor).
5. For each flick: find frame `a` = the frame where the real fingertip stops (turn-around point), and frame `b` = the frame
   where the portal corner nearest that fingertip stops. `latency_ms = (b - a) / F_slow * 1000`.
6. Take the median and the worst of the 8-10 flicks. Write them into `REPORT.md` under "Motion-to-photon latency".

## Reading the result
- Median <= 80 ms: meets the target. 80-120 ms: usable, noticeable on fast moves. >120 ms: check the list below.
- Components (typical): camera exposure+readout+USB 30-60 ms, TD readback (1 frame) ~16 ms, MediaPipe ~15 ms,
  UDP + Presenter ~1 ms, render + display 16-32 ms. The Presenter's extrapolation (config `predict`) hides roughly the
  `age_ms` part (it moves the portal to where the hand *will be* now), so the physical result is usually lower than the sum.

## Reducing it
- Camera at 60 fps (MJPEG 720p60) halves the camera-side term; short shutter; disable camera "smart" processing.
- `numpyArray(delayed=True)` in `td/scripts/frame_writer.py` trades one frame of latency for no GPU stall; set `delayed=False`
  only if the self-check shows the sidecar is not the bottleneck and TD still holds 60 fps.
- Run the TD window full-screen with vsync on (Perform mode, F1); disable display post-processing / HDR on the monitor.
