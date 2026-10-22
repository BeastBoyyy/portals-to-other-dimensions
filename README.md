# Portals to Other Dimensions

**Open a portal between your hands. Twist your wrists to change what is inside it. No mouse, no keyboard, no special camera, no wearables: just a normal webcam and your hands.**

*IT Exhibition project  |  Team: `[write your names, class and school here]`  |  Guide: `[teacher's name]`*

---

## Contents

1. [What is it? (the 30-second version)](#1-what-is-it-the-30-second-version)
2. [What visitors see and do](#2-what-visitors-see-and-do)
3. [Why did we build it?](#3-why-did-we-build-it)
4. [How it works: the architecture](#4-how-it-works-the-architecture)
5. [The six effects, and the idea behind each one](#5-the-six-effects-and-the-idea-behind-each-one)
6. [Problems we had to solve (the interesting part)](#6-problems-we-had-to-solve-the-interesting-part)
7. [What this project teaches](#7-what-this-project-teaches)
8. [Honest results: what we measured and what we did not](#8-honest-results-what-we-measured-and-what-we-did-not)
9. [How to explain it at the exhibition](#9-how-to-explain-it-at-the-exhibition)
10. [Exhibition set-up checklist](#10-exhibition-set-up-checklist)
11. [Install and run](#11-install-and-run)
12. [Troubleshooting](#12-troubleshooting)
13. [Project map: where everything is](#13-project-map-where-everything-is)
14. [Limitations and future work](#14-limitations-and-future-work)
15. [Glossary](#15-glossary)
16. [Credits and licences](#16-credits-and-licences)

---

## 1. What is it? (the 30-second version)

You stand in front of a webcam and hold your hands up like a picture frame. The computer finds **both of your hands**, works out where your **thumb tips and index finger tips** are, and draws a glowing **glass box** that is pinned to those four fingertips. The box follows your hands in real time.

- **Inside the box** you see an "other dimension": your own camera picture, transformed by a special visual effect.
- **Outside the box** you see the normal, untouched camera picture.
- **Open your hands wide** and the box unfolds. **Pinch your fingers together** and it folds shut.
- **Twist your wrists** and the box twists in 3D, and the effect inside changes to the next one (there are six).
- **Move your hands fast** and the effect "glitches" and reacts to the speed.

Everything is controlled by your hands alone. Nobody touches the keyboard or the mouse.

The idea comes from "portal" videos made with a professional (paid) program called TouchDesigner. We rebuilt the whole thing ourselves in **Python, with free and open-source tools only**, and then spent most of our time making it **accurate, smooth and fast**.

---

## 2. What visitors see and do

| You do this | The project does this |
|---|---|
| Hold both hands up, thumbs and index fingers like an "L" / picture frame | A glass box appears, pinned to your four fingertips |
| Spread your fingers wide | The box **opens** (unfolds in about a fifth of a second) |
| Pinch all fingertips together (like holding a pinch of sand) | The box **closes** and disappears |
| Move your hands around, tilt them, move them closer or further | The box follows, rotates and changes size with them |
| Twist your wrists forward (palms turning edge-on to the camera) | The box twists in 3D, and after a short hold the **effect changes** to the next one |
| Twist back the other way | Goes back to the previous effect |
| Move your hands quickly | Every effect reacts: it glitches, smears, tears or races |

Full guide for visitors: **[GESTURES.md](GESTURES.md)**. No key presses are ever needed. (The only keys that exist are for the person running the stand: `H` debug overlay, `F` fullscreen, `Esc`/`Q` quit.)

---

## 3. Why did we build it?

**The inspiration.** We saw a short video online where someone "opens a portal" between their hands. It was made with TouchDesigner (paid software) plus Google's MediaPipe hand tracking. We wanted to find out how that works, and whether we could build it **without paid software**, as a complete program that anyone with a laptop and a webcam can run.

**The goal.** Not just "make it work once", but make it work **well**. We set ourselves priorities, in this order:

1. **Correctness**: the box must really sit on the fingertips, and must never appear on the wrong thing (a poster of a hand, a second person, a ghost detection).
2. **Smoothness**: no shaking, no jumping, no flicker.
3. **Low delay**: the box should feel attached to your hand, not trailing behind it.
4. **Looks**: beautiful, reactive effects.

Looks are last on purpose. A pretty effect on a shaky box looks bad; a steady box with simple effects looks professional. Most real computer-vision engineering is exactly this: the glamorous part (graphics) is easy compared with getting the *measurement* (where are the fingers?) right.

**The point of the project, in one sentence:** it is a hands-on way to learn how a computer *sees* (computer vision and machine learning), how raw, noisy measurements are cleaned up (signal processing), and how images are produced and transformed on a graphics card (shaders), all joined together in one real-time system.

---

## 4. How it works: the architecture

### 4.1 The big picture

```
        WEBCAM (1280 x 720, 30 frames per second)
              |
              v
  +-----------------------------------  TRACKER PROCESS  ------------------------------------+
  |                                                                                           |
  |  Camera thread  --->  Hand finder (palm detector)         "Where are the hands?"          |
  |       |                      |                                                            |
  |       |                      v                                                            |
  |       |          Hand follower: 21-point landmark model on                                |
  |       |          full-resolution crops, 2 hands, ~30 times/second                         |
  |       |                      |      "Where is every joint of every finger?"               |
  |       |                      v                                                            |
  |       |          HandCore  (our own tracking library)                                     |
  |       |            1. mirror + convert to pixels                                           |
  |       |            2. checks: is this really a hand? (score, edges, jumps)                |
  |       |            3. identity: which hand is hand 0 / hand 1? (never swap)               |
  |       |            4. smoothing: Kalman + One Euro filters                                |
  |       |            5. build the box from 4 fingertips (+ lid from middle/ring fingers)    |
  |       |            6. pose: how open? how twisted? how fast?                              |
  |       v                      |                                                            |
  |  camera frames               |  box corners, velocity, pose  (small message)              |
  +-------|----------------------|------------------------------------------------------------+
          | shared memory        | pipe
          v                      v
  +--------------------------------  RENDER PROCESS  --------------------------------------+
  |  Presenter: predicts ahead to hide delay, springs for smooth motion, fade in/out        |
  |  ViewState: opening animation, 3D twist of the box, twist-dial (which effect), speed    |
  |  GPU renderer (OpenGL shaders):  effect  ->  bloom glow  ->  glass-box composite        |
  |  Window drawn at the screen's refresh rate (e.g. 60-144 times per second)               |
  +-----------------------------------------------------------------------------------------+
```

### 4.2 Why two processes?

A program normally has to do two jobs at once: *tracking* (measure the hands, ~30 times per second) and *drawing* (update the picture, 60 to 144 times per second). In Python, threads inside one program share a lock (the "GIL"), so a busy drawing loop can starve the tracker. We **measured** this: inside one process the tracker dropped to 13 updates per second; in its own process it holds the webcam's full 30. So the project is split into a **tracker process** and a **render process** that talk through fast shared memory (camera pictures) and a pipe (tiny state messages).

### 4.3 The tracking pipeline, step by step

| Step | What happens | Why it is needed |
|---|---|---|
| **1. Find the hands** | Google's MediaPipe palm detector finds hands in the picture | Cheap way to know *where to look* |
| **2. Follow and measure** | The MediaPipe hand-landmark network (the same network Google uses) is run on a rotated crop of the **full-resolution** frame around each hand, giving **21 landmarks per hand** (every finger joint, the wrist, the fingertips). Two slightly different crops are averaged per hand. | More pixels on the fingertips = more precise. Averaging reduces random error. A hand keeps being followed even while it twists, instead of being re-detected from scratch |
| **3. Validate** | Each detection is checked: confidence score, not cut by the frame edge, not teleporting | A wrong measurement must never reach the screen |
| **4. Identity** | A small tracker keeps "slot 0" and "slot 1" consistent over time, including when hands cross. If it is ambiguous, it refuses to guess | Without this, the box twists into a bow-tie when two hands cross |
| **5. Smooth** | A **Kalman filter** predicts the hand and bridges short gaps; a **One Euro filter** removes jitter when still but follows quickly when moving; a "spike gate" ignores a single fingertip that jumps and then comes back | Raw landmarks shimmer by a pixel or so even when you hold still |
| **6. Build the box** | Four corners: your two thumb tips and two index tips. The glass "lid" reaches your middle and ring fingertips. The shape is checked (convex, not tiny, sensible size) | The box is *always* on the same fingers, whichever fingers you raise |
| **7. Read the pose** | **Openness** = how spread the fingertips are compared to the palm size (pinched ~0.2-0.4, open ~0.8-1.2). **Twist** = angle between the palm's 3D normal and the camera. **Speed** = how fast the hands move | These drive the open/close animation, the 3D twist of the box, the effect choice and the reactivity |
| **8. Send** | Corners, velocity and pose go to the render process | |

### 4.4 The render pipeline

1. **Presenter**: the tracker only produces ~30 updates a second, but the screen refreshes much faster. The presenter **predicts forward** by the measured delay (latency compensation) and uses a critically damped spring so the box moves continuously between measurements. This is what makes motion look "buttery".
2. **ViewState**: turns the pose into visuals. The box *unfolds* as your hand opens, each hand's twist pushes its side of the box deeper (real perspective), and a **relative twist ratchet** decides which effect is shown: rotate your wrists about 20 degrees further and hold for a quarter of a second to step to the next effect. (We use a *relative* ratchet, not an absolute angle, because everyone rests their hand at a different angle.)
3. **Effects (GLSL shaders)**: each effect is a small program that runs on the graphics card for every pixel, every frame. Effects are only drawn while visible, and cross-fade when you switch.
4. **Bloom and composite**: a blurred copy of the effect adds glow; the final pass draws the perspective **glass box** (back face, edges, lid, glitch band) over the camera picture so that only the inside of the box shows the effect.

### 4.5 Design principles we followed

- **Separation of concerns**: `handcore/` (the tracking brain) has **no** camera, GPU or MediaPipe code in it. It takes landmarks in and gives a box out, so the same code runs live, on a recorded file, or on a simulator. That is what made it testable.
- **Everything tunable lives in one file** (`config.yaml`), and every feature has an on/off switch.
- **Never guess**: when a measurement is doubtful the system holds, predicts or hides the portal rather than draw something wrong.
- **Measure, don't assume**: every claim in section 8 comes from a measurement; the things we could not measure are marked **NOT MEASURED**.

---

## 5. The six effects, and the idea behind each one

Twisting steps through them in this order (and wraps around). Each is a different idea from computer graphics or video art, implemented as a GPU shader.

| # | Effect | What you see | The idea behind it |
|---|---|---|---|
| 1 | **Frosted glass** | The room seen through etched glass with tiny water droplets; each droplet is a little clear lens showing the room upside-down; a glint drifts across. Moving fast wipes the frost clear. | Frosted glass scatters light, which is a blur with a random grain. Each pixel blurs with a randomly rotated disc, and droplets act like small lenses (refraction). |
| 2 | **Moire** | Big, slow, colourful fringes flowing across the picture; the shapes in the room bend the fringes. | Lay two fine line patterns over each other at a tiny angle and a large new pattern appears. We compute it exactly from the *phase difference* of the line patterns, so it is perfectly smooth. |
| 3 | **Slit scan** | Anything that moves is stretched into flowing ribbons of colour; the still room stays still. | Every pixel shows a **different moment in time** (up to about 1.5 seconds ago), more the further it is from a "live line". The program keeps the last 48 camera frames on the GPU. |
| 4 | **Pixel sort** | Bright shapes melt into long streaks of sorted colour. | Pixels in a column are **sorted by brightness**. We sort 16 samples per pixel with a sorting network (a fixed list of compare-and-swap steps), entirely on the GPU. |
| 5 | **CRT** | An old curved TV: scan lines, coloured phosphor dots, glow, a rolling bar, and tearing when you move fast. | A simulation of how a cathode-ray tube draws: a beam per scan line (bright lines are fatter), a phosphor mask, light bleeding in the glass (halation), barrel distortion. |
| 6 | **Datamosh** | Moving parts drag old pixels along in square blocks and smear, until a recovery band sweeps through and fixes them. | What happens when a video codec loses its key frame: the picture is the *previous* picture pushed along by **motion vectors** (found by block matching, like a video encoder does). We do the same each camera frame. |

**All six react to your hand speed** (see [GESTURES.md](GESTURES.md)).

**Which effects did we *not* build?** *Point cloud* and *depth map*. They need to know how far away things are, and a normal webcam does not measure distance. We could have faked a depth map, but a fake would be dishonest, so we left them out.

---

## 6. Problems we had to solve (the interesting part)

This section is the best material for explaining the project: **real problems, found by measuring, and how we fixed them.**

1. **Raw hand tracking shimmers.** Even a perfectly still hand gives landmarks that move by a pixel or so. *Fix:* Kalman filter + One Euro filter + averaging several crops. Measured on our recordings: raw jitter about 0.6 px, filtered about 0.5 px (our target of 0.4 px was *not* reached).
2. **The portal disappeared when we twisted the hand.** A hand turned edge-on is very hard for any model. Our first safety checks were *too strict* and rejected real, twisted hands. *Fix:* we replayed our own recordings through the checks, found which rules rejected real hands, relaxed them, and made the tracker keep following a hand through a few weak frames instead of dropping it. Portal-valid frames in a replay went from about 60 % to about 80 %.
3. **The box jumped far from the fingers after a hand came back.** After a hand was lost for a while, the filters still remembered the *old* pose. *Fix:* when a hand returns after a long gap, all of its filters restart clean. A good example of a bug that only shows up with *real* data, never in neat tests.
4. **Only 20 tracking updates per second.** *Cause (measured):* the webcam can only deliver 30 frames per second in every mode we tried, and the program was also sharing one process between tracking and drawing. *Fix:* separate processes and a faster tracker; now about 30 updates per second, the webcam's limit.
5. **A single slow effect: 355 milliseconds per frame.** The first pixel-sort shader used an array indexed by a variable, which the graphics card handles very slowly. *Fix:* unrolled the sorting network so every index is constant: 12 ms, and about 8 ms after rendering it at half resolution.
6. **The GPU stalled when we updated the slit-scan history.** Writing into a texture the GPU is still reading forces a wait: the effect ran at 63 frames per second. *Fix:* two copies of the history, read one while writing the other: 144 frames per second.
7. **We were showing the wrong effect in our own previews.** A twist of 60 degrees made the ratchet step one effect further, so every preview image was labelled with the *neighbouring* effect. We only noticed because the images did not match their labels. *Lesson:* check that your test measures what you think it measures.
8. **A hand on a poster, a second person, or a random "ghost" detection.** *Fix:* acquisition needs several consecutive good frames; identity tracking refuses to guess; the optional person-mask gate is available (and tested against simulated posters and flicker).

---

## 7. What this project teaches

| Area | What you learn from it |
|---|---|
| **Computer vision** | How a computer finds hands and 21 joint positions in a picture; what a "landmark" is; why more pixels on the fingertips gives a more precise answer |
| **Machine learning in practice** | Running a trained neural network (inference) in real time; why the *model* is only part of the system; confidence scores; false positives |
| **Signal processing** | Noise vs signal; the Kalman filter (predict, then correct); the One Euro filter (smooth when slow, responsive when fast); latency vs smoothness trade-off |
| **Mathematics** | Vectors, rotations and perspective; the 3D normal of a palm; phase and interference (moire); sorting networks |
| **Computer graphics** | Fragment shaders (GLSL), textures, framebuffers, bloom, refraction, feedback loops, anti-aliasing |
| **Concurrency** | Threads vs processes; shared memory; why one Python process can starve itself; frame pacing |
| **Software engineering** | Separating a "brain" from its inputs and outputs; configuration files; automated tests (121 of them); replaying recorded data to reproduce bugs |
| **Scientific honesty** | Measuring before claiming; writing down what is **not** measured; finding that your own test was wrong |
| **Human-computer interaction** | Designing controls that need no keys: choosing gestures that are natural and hard to trigger by accident (open/pinch, relative twist) |

**Why it matters outside school:** the same ideas run gesture control in cars and games, hand tracking in AR/VR headsets, sign-language and accessibility tools, touch-free interfaces in hospitals and kitchens, and the real-time effects in live concerts and film.

---

## 8. Honest results: what we measured and what we did not

All numbers below are from this project on the team's own laptop (Windows 11, integrated Intel graphics, a built-in 30 fps webcam). They will differ on other machines.

### Measured

| What | Result |
|---|---|
| Webcam frame rate | **30 fps** in every mode we tried (this is the ceiling for tracking) |
| Tracking updates per second | **about 30** (was about 20 before the rebuild; it dropped to 13 when tracker and renderer shared one process) |
| Both hands tracked | about **86-89 %** of frames in two live sessions (the rest: hands out of view or lost) |
| Portal valid on screen | about **84-91 %** of frames in two live sessions |
| Fingertip jitter, hands held still | raw about **0.6 px**, after our filters about **0.5 px** (median of the few "still" moments in the recordings; target 0.4 px **not reached**) |
| Frame rate per effect (live window, integrated GPU, 10 s each) | moire, slit scan, CRT: **144 fps** (the screen's refresh limit); pixel sort **about 100**; datamosh **about 114**; frosted glass **about 91** |
| Automated tests | **121 passing** (tracking logic, filters, geometry, the renderer, effect shaders, the app's data hand-off) |
| Repeatability of landmark models (how much the answer wobbles for slightly different crops, as % of palm width at the fingertips) | RTMPose-m 4.9, MediaPipe Tasks 7.8, MediaPipe ONNX 10.2 (lower is steadier) |

### NOT measured (we do not claim these)

- **Absolute accuracy**: how many pixels our fingertip is from the *true* fingertip. This needs thousands of hand-labelled frames, which we do not have. Everything above is about *steadiness and coverage*, not truth. Our app can record a video (`--record-video`, off by default) so this can be measured later.
- **End-to-end delay** (finger moves -> box moves on screen) with a high-speed camera.
- **How it looks and performs on other people's webcams**, in different rooms, with different skin tones and lighting. Most of our tuning used the team's own recordings and one public reference clip.
- **Any claim of "99 % accuracy"**: we set that as a goal early on, but we can only honestly say what is listed under "Measured".

---

## 9. How to explain it at the exhibition

### 9.1 The 30-second explanation (for anyone walking past)

> "This is a portal made with your hands. The webcam finds your thumbs and index fingers, pins a glass box to them, and anything inside the box is an effect on the live camera picture. Open your hands to open it, pinch to close it, and twist your wrists to change the effect. There are no keys and no special hardware: just a normal webcam and a lot of maths. Want to try?"

### 9.2 The 2-minute explanation (for visitors who stay)

1. **Demo first.** Let them try it for 30 seconds *before* explaining anything.
2. **What the computer sees.** "A neural network from Google finds 21 points on each hand: every finger joint. We use four of them for the corners of the box."
3. **The hard part is not the effects.** "The effects are the easy, fun part. The hard part is that those 21 points *shake* a little in every frame. Without cleaning, the box would vibrate. We use two filters: one *predicts* where your hand will be (a Kalman filter), one *smooths when you are still and follows when you move* (a One Euro filter)."
4. **Speed.** "The webcam gives 30 pictures per second, but your screen draws up to 144. So the program *predicts* between pictures, and that makes the motion smooth."
5. **The effects.** "Each effect runs on the graphics card for every pixel, every frame. For example slit scan shows a different moment in time at every pixel."
6. **What we are proudest of.** Pick one story from section 6 (the stale-filter bug or the 355 ms shader are good ones).

### 9.3 The 5-minute explanation (for teachers, judges, and curious people)

Walk through the diagram in section 4.1: *camera -> find hands -> follow (21 landmarks) -> validate -> identity -> smooth -> build box -> pose -> send -> predict -> draw*. Then explain **why two processes**, then tell **two or three stories** from section 6, then finish with section 8: **what we measured and what we did not**. Judges respect honesty about limitations far more than big claims.

### 9.4 Questions you may be asked, and honest answers

| Question | Answer |
|---|---|
| **Is this AI?** | Part of it is. A neural network (Google MediaPipe) finds the hand landmarks. Everything else (filters, geometry, gestures, effects) is our own code and maths. |
| **Did you train the AI?** | No. We use Google's pre-trained MediaPipe models (free, open source). Our work is the system around it: validation, identity tracking, smoothing, prediction, gestures, rendering. |
| **Why not use TouchDesigner?** | It is paid software and a closed tool. We wanted to understand each step and have a program anyone can run for free. (TouchDesigner is excellent for artists; we did try it first.) |
| **Does it record me / send my picture anywhere?** | The camera picture stays on this computer; nothing is sent over a network. The program writes small log files of *hand-point coordinates* (numbers, not pictures) to the `logs/` folder. It only saves video if started with `--record-video`. |
| **Does it work with any hand / skin tone / lighting?** | MediaPipe works on many hands, but we have **not** tested a wide range of people or rooms, so we cannot claim it works equally for everyone. Good, even front lighting helps most. |
| **What if two people stand in front?** | It is built for one person with two hands. A second person's hands may confuse it. |
| **Why does it sometimes lose the box?** | Hands turned edge-on, covered, or cut by the frame edge are very hard for any model. The program prefers to hide the box rather than show it in the wrong place. |
| **How accurate is it?** | We measured steadiness (about half a pixel of jitter when still) and how often it tracks (about 86-89 % of frames), but we have **not** measured true accuracy against labelled data. |
| **What would you improve?** | A camera with 60 fps, a depth camera for real depth/point-cloud effects, labelled data to measure true accuracy, and testing with many more people. |
| **How long did it take / how many lines of code?** | `[fill in honestly]`. (The core tracking library is about 2,400 lines of Python together with the app, plus the GLSL shaders and tests.) |

---

## 10. Exhibition set-up checklist

**Equipment:** a laptop (a recent one with a decent graphics card is best), a webcam (the built-in one is fine), a monitor or TV facing the visitor, and the power cable.

**Before the day**
- [ ] Do a full test run at home with the exact laptop you will bring.
- [ ] If the laptop has an NVIDIA/AMD graphics card, make the app use it (otherwise it may run on the weaker integrated chip): see [Switching to the dedicated graphics card](#switching-to-the-dedicated-graphics-card) below, then check that the first lines printed on start-up say `GL: NVIDIA ...`.
- [ ] Plug the laptop into power and set the power mode to "best performance".
- [ ] **Record a backup:** screen-record a 1-minute demo with a phone or screen recorder. If the stand fails, you can still show it.
- [ ] Close every other program that uses the camera (Zoom, Teams, TouchDesigner, the Camera app).

**At the stand**
- [ ] Put the camera at about chest-to-face height, and the visitor about **1 metre back**, with both hands fully inside the picture.
- [ ] **Light the visitor from the front** (a lamp behind the screen is perfect). A bright window *behind* the visitor hides their hands.
- [ ] A plain, uncluttered background helps (no posters of people).
- [ ] Start with `.venv\Scripts\python app\portal_app.py`, then press `F` for fullscreen.
- [ ] One visitor at a time.
- [ ] Put a small sign next to the screen: *"Hold both hands up like a picture frame. Spread fingers = open. Pinch = close. Twist wrists = change effect."*

**Afterwards (privacy):** delete the `logs/` folder. It holds coordinates of the visitors' hand points for the session (no pictures unless you used `--record-video`).

---

## 11. Install and run

### Requirements
- Windows 10/11 (the project was built and tested on Windows 11; other systems are untested)
- Python 3.10 or newer (built with Python 3.14)
- A webcam and a graphics card/driver that supports OpenGL 3.3 (almost all do)

### Install

```bash
cd portal
python -m venv .venv
.venv\Scripts\python -m pip install mediapipe opencv-python numpy pyyaml moderngl glfw pillow ai-edge-litert pytest
```

Then download the model files (they are not stored in the repository; this fetches them from Google's public MediaPipe bucket, about 8 MB):

```bash
.venv\Scripts\python tools\get_models.py
```

(We developed and measured with a `hand_landmarker.task` that was 7.5 MB; the bucket's current "latest" file is 7.8 MB, so results on a fresh download may differ slightly.) This creates `models/hand_landmarker.task` (the MediaPipe hand model), `models/selfie_segmenter.tflite`, and unpacks the landmark network into `models/tflite/hand_landmarks_detector.tflite`.

### Run

```bash
.venv\Scripts\python app\portal_app.py
```

Useful options:

| Option | What it does |
|---|---|
| `--hud` | Show a debug overlay (frame rates, hand counts, markers on fingertips). `H` toggles it while running. |
| `--effect moire` | Start on a particular effect (`glass`, `moire`, `slitscan`, `pixelsort`, `crt`, `datamosh`). |
| `--engine tasks` | Use the stock MediaPipe tracker (a fallback if the default tracker misbehaves on a machine). |
| `--tta 1` | Average fewer crops per hand: faster but a little less steady (default 2; up to 5). |
| `--camera 1` | Use a second webcam. |
| `--source clip.mp4` | Play a video file instead of the webcam (good for testing without a camera). |
| `--no-mirror` | Do not mirror the picture. |
| `--record-video` | **Opt-in**: save the raw camera video to `logs/session_video.mp4` for offline testing. Off by default. |
| `--seconds 60` | Quit automatically after 60 seconds (testing). |

Keys (for whoever runs the stand): `H` overlay, `F` fullscreen, `Esc` or `Q` quit.

### Run the automated tests

```bash
.venv\Scripts\python -m pytest -q
```
(The tests that need a graphics context create their own off-screen one; no window or webcam is needed.)

---

### Switching to the dedicated graphics card

Many laptops have two graphics chips: a weak **integrated** one (Intel/AMD) and a stronger **dedicated** one (NVIDIA RTX/GTX). Windows decides per program which one to use, and for Python it usually picks the weak one. To make the app use the strong one:

1. Open **Settings -> System -> Display -> Graphics** (on Windows 10: *Settings -> System -> Display -> Graphics settings*).
2. Click **Add desktop app -> Browse** and choose the Python program the project runs with. Add **both** of these if you can find them (the project starts a second helper process):
   - `<project folder>\.venv\Scripts\python.exe`
   - the base Python it points to (run `.venv\Scripts\python -c "import sys; print(sys.base_prefix)"` to see the folder; the file is `python.exe` inside it).
3. Click each entry -> **Options** -> choose **High performance** (your NVIDIA card) -> **Save**.
4. Close the app completely and start it again.
5. **Check it worked:** the first lines printed in the terminal include `GL: <name of the graphics chip>`. It should now say NVIDIA (for example `NVIDIA GeForce RTX 4050 Laptop GPU`) instead of Intel. You can also open **Task Manager -> Performance -> GPU** while the app runs and see which GPU is busy.

If you use NVIDIA Control Panel instead: *Manage 3D settings -> Program Settings -> add python.exe -> Preferred graphics processor: High-performance NVIDIA processor*.

## 12. Troubleshooting

| Problem | What to try |
|---|---|
| "cannot open the camera" | Close other programs using the webcam (Zoom, Teams, TouchDesigner). Try `--camera 1`. |
| The box wobbles / disappears | Light the hands from the front; keep both hands fully inside the frame; avoid a bright window behind you. |
| The box lags behind fast motion | Normal at 30 fps; make sure the laptop is on power and the program runs on the faster graphics chip. |
| Low frame rate | Use `--cell 4` (fewer particles in older effects) and set Windows Graphics preference to High performance for `python.exe`. |
| Effects change when you did not mean to | Rest your hands still for a second so the program re-centres its "resting" twist angle; then make a deliberate twist. |
| Nothing appears when hands are open | Both hands must be open: the box only opens when **both** hands spread their fingers. |
| "GPU delegate unavailable" message | Normal on Windows. MediaPipe falls back to the CPU; the program still works. |

---

## 13. Project map: where everything is

```
README.md            this file                 GESTURES.md   how to use it (for visitors)
DECISIONS.md         why every design choice was made, with measurements (D1-D18)
REPORT.md            measured results from the early (simulation) stage      TD_NOTES.md  notes from the earlier TouchDesigner stage
config.yaml          every tunable setting, documented inline

app/                 the program you run
  portal_app.py        window + render loop + command-line options (render process)
  tracker_proc.py      the tracker process: camera -> hands -> HandCore -> messages; shared-memory frames
  roi_landmarker.py    glue: hand-finder + hand-follower with the same interface as stock MediaPipe
  roi_tracker.py       follows each hand with the landmark network on full-resolution rotated crops
  hand_models.py       the neural-network wrappers (landmark models, crop geometry)
  renderer.py          OpenGL renderer: effects, bloom, glass-box composite, frame history, datamosh feedback

handcore/            the tracking "brain" (no camera, no GPU, no MediaPipe inside)
  pipeline.py          the main flow: ingest -> validate -> identity -> filter -> box
  validate.py          is this really a hand? (score, edges, geometry)
  identity.py          which hand is which, over time; acquire / lose rules
  one_euro.py kalman.py   the two filters          constraints.py   spike gate for single fingertips
  quad.py              builds and checks the four-corner box
  pose.py              openness / twist / speed of the whole hand
  predict.py           Presenter: latency compensation and spring smoothing
  view.py              opening animation, 3D box twist, the twist ratchet that picks the effect

td/shaders/          the GPU programs (GLSL): fx_glass, fx_moire, fx_slitscan, fx_pixelsort, fx_crt, fx_datamosh(+_mv),
                     composite (the glass box), plus older effects kept for reference (ascii, bitmap, neon, web, particles ...)
tests/               121 automated tests         tools/   recording, replay, measuring and preview tools
fixtures/preview/    older shader preview renders (the effect previews we made from a third-party video are not included)
models/              the neural-network files    logs/    written at run time (hand coordinates; delete after the exhibition)
```

Useful tools (all optional): `tools/real_jitter.py` (measures jitter on a recorded session), `tools/effects_preview_video.py` (renders every effect on a video and times them), `tools/bench_roi.py` and `tools/bench_models.py` (compare hand models), `tools/trace_summary.py` (summary of a live session).

---

## 14. Limitations and future work

**Limitations (be upfront about these)**
- The webcam's 30 fps limits tracking to 30 updates per second; fast hand motion is bridged by prediction.
- Hands turned fully edge-on, hidden behind each other, or cut by the frame edge are lost; the box hides rather than guesses.
- Built for one person with two hands, tested mostly with the team's own hands in a few rooms.
- Absolute accuracy against labelled data is **not measured**; jitter is about 0.5 px, above our 0.4 px target.
- Tested on one Windows laptop (integrated graphics). Performance on other machines is unknown.
- No point cloud or depth-map effect (no depth sensor).

**Future work**
- Record and label a proper test set to measure true accuracy.
- Use a 60 fps camera, or a depth camera (e.g. a RealSense/Kinect-style sensor) to unlock point-cloud and depth-map effects.
- Test with many different people, skin tones, rooms and webcams.
- Add sound that reacts to the gestures; add a way to choose an effect directly.
- Export the result to projection or VJ software.

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **Landmark** | A named point on the hand (e.g. "index fingertip"). MediaPipe gives 21 per hand. |
| **MediaPipe** | Google's free, open-source toolkit for tracking hands, faces and bodies in real time. |
| **Neural network / model** | A program learned from many example pictures; here it outputs hand landmarks. |
| **Inference** | Running an already-trained model on new data (as opposed to *training* it). |
| **Jitter** | Small random shaking of a measurement even when nothing moves. |
| **Kalman filter** | A method that predicts where something will be, then corrects the prediction with each new measurement. |
| **One Euro filter** | A smoothing filter that smooths strongly when you are still and weakly when you move fast, so it is steady *and* responsive. |
| **Latency** | The delay between a real action (moving your hand) and its effect on screen. |
| **Frame rate / fps / Hz** | How many pictures (or measurements) per second. |
| **Shader / GLSL** | A small program that runs on the graphics card for every pixel. GLSL is the language. |
| **Framebuffer / texture** | An image stored on the graphics card that shaders read from or draw into. |
| **Bloom** | A soft glow around bright areas (a blurred copy added back on top). |
| **Moire** | The large pattern you see when two fine patterns overlap at a slight angle. |
| **Slit scan** | A photography/video technique where different parts of the picture show different moments in time. |
| **Pixel sorting** | A glitch-art technique that sorts pixels (e.g. by brightness) along rows or columns. |
| **Datamosh** | A glitch-art effect from video compression errors where old picture content is dragged along by motion data. |
| **CRT** | Cathode-ray tube: the old, bulky kind of TV or monitor. |
| **Process / thread** | Two ways a program does several things at once; processes are separate programs with their own memory, threads share memory. |
| **GIL** | Python's "global interpreter lock": only one thread runs Python code at a time, which is why we use two processes. |
| **Ratchet** | A control that moves one step each time you push past a threshold (like a ratchet wrench). Our twist control works this way. |

---

## 16. Credits and licences

- **Hand tracking models and palm detector:** Google **MediaPipe** (open source; check the MediaPipe licence terms before redistributing the models).
- **Libraries:** OpenCV (picture handling), NumPy (maths), moderngl and GLFW (OpenGL window and rendering), Pillow (text atlas), PyYAML (settings), ai-edge-litert (runs the landmark network), pytest (tests). Each has its own open-source licence.
- **Inspiration:** a social-media portal video made with TouchDesigner and MediaPipe. We did not copy any code or assets from it; the idea and the look were our starting point, and everything in this project was built by us.
- **Effects** are our own implementations of well-known techniques from computer graphics and video art (frosted glass, moire, slit scan, pixel sorting, CRT simulation, datamoshing).
- **One Euro filter:** Casiez, Roussel and Vogel, *"1 Euro Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"*, CHI 2012 (we implemented the published idea ourselves).
- This project was developed with the help of an AI coding assistant (Claude by Anthropic) working with the team. `[Edit this line to say exactly how your team used it and what each member did, so your credits are accurate.]`

*Made for the school IT exhibition. Have fun, and please try the twist!*
