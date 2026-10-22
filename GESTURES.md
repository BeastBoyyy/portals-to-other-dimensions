# How to use it: no keys, just your hands

Run (no TouchDesigner needed): `.venv\Scripts\python app\portal_app.py`   (`--hud` for the debug overlay, `--cell 4` for fewer particles)

## The box
It is **always attached to the same fingers**: bottom corners = your **thumb tips**, top corners = your **index tips**, and the glass **lid**
(the angled "wings" along the top) reaches your **middle + ring** fingertips. Raising or curling fingers does not change that.

## Opening and closing: the whole hand
- **Pinch** (all five fingertips gathered, like holding a pinch of sand) = the box is **closed** (nothing drawn).
- **Open the hand** (fingers spread) = the box **opens**, unfolding out of your hands in about 0.2 s. **Pinch again** and it folds shut.
- Measured on your reference clip: fingertip spread / palm size is 0.2-0.4 when pinched and 0.8-1.2 when open, so the portal is closed below 0.35 and
  fully open above 0.80 (`gesture.spread_closed`, `gesture.spread_open`). **Both hands** must open for it to open.

## Twisting: the box twists in 3D and turns the effect dial
Rotate your wrists (palm turned from facing the camera toward edge-on). The angle of the palm to the camera is the **twist**:
- each hand's twist pushes *its* side of the box deeper, so the box visibly twists in perspective with your hands;
- the effect changes with a **relative twist ratchet**: from wherever your hands rest, rotate your wrists ~20 degrees further and hold it for a quarter of a second to step to the next effect (rotate back the other way for the previous one). A partial twist previews the neighbouring effect; the resting angle slowly re-centres itself, so no absolute angle matters.
- order (wraps around): **frosted glass** > **moire** > **slit scan** > **pixel sort** > **CRT** > **datamosh**:
  - **frosted glass**: etched glass with condensation droplets (each a tiny clear lens), a drifting glint; moving fast wipes the frost clear.
  - **moire**: layered line screens whose interference makes big flowing coloured fringes; the room bends the fringes.
  - **slit scan**: every pixel shows a different moment (up to ~1.5 s back) so movement stretches into ribbons; twisting rotates the time axis.
  - **pixel sort**: bright shapes melt into long sorted streaks; a brightness window sweeps through the picture.
  - **CRT**: curved tube, scan lines, phosphor mask, halation, rolling bar, sync tearing.
  - **datamosh**: codec-style smear: moving areas drag old pixels along in 16 px blocks until a recovery band sweeps through.

## Reactive: speed
Every effect reacts to hand speed: glass wipes clear and disperses colour, moire fringes tighten and race, slit scan reaches further back in time, pixel sort widens its window
and scrolls faster, CRT tears and converges badly, datamosh exaggerates motion vectors and stops healing. The portal box also glitches (band shifts + RGB split). Hold still and it settles.

## Tips
- Sit ~1 m back with both hands in the middle of the frame, palms toward the camera, front-lit.
- Twist estimates are noisy (+-10 deg): make a clear wrist rotation; the ratchet needs a deliberate rotation held briefly, so it will not flicker.
- Tune everything in `config.yaml` under `gesture:` (ratchet step / hold time, effects order, open/close thresholds, speed sensitivity).
- The reference clip's mosaic blocks over faces are post-production privacy blur and are NOT reproduced.
