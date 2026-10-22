# Build the whole portal network inside TouchDesigner.   (needs user verification in TD -- see MANUAL_STEPS.md)
#
# In TouchDesigner open the Textport (Alt+T) and run:
#     REPO = r'D:\TDMP\portal'
#     exec(open(REPO + r'\td\build_network.py').read())
#
# Idempotent: re-running reuses operators with the same names and re-applies parameters/text.
# Parameter names differ slightly between TD builds, so every assignment goes through setp(): anything TD rejects is
# collected and printed at the end as a short manual checklist instead of aborting the build.
import json
import os

REPO = globals().get('REPO') or os.path.dirname(os.path.dirname(os.path.abspath(project.folder)))
CFG = json.load(open(os.path.join(REPO, 'td', 'config.json')))
W, H = CFG['camera']['width'], CFG['camera']['height']
issues = []


def rd(*p):
    with open(os.path.join(REPO, *p), encoding='utf-8') as f:
        return f.read()


root = op('/project1')
base = root.op('portal') or root.create(baseCOMP, 'portal')


def make(kind, name):
    o = base.op(name)
    return o if o is not None else base.create(kind, name)


def setp(o, name, val, expr=False):
    try:
        p = getattr(o.par, name)
        if expr:
            p.expr = val
        else:
            p.val = val
    except Exception as e:
        issues.append('%s.%s = %r  (%s)' % (o.name, name, val, str(e)[:60]))


def text(name, body):
    d = make(textDAT, name)
    d.text = body
    return d


def wire(dst, *srcs):
    for i, s in enumerate(srcs):
        dst.inputConnectors[i].connect(s)


def glsl(name, shader, *inputs, vecs=()):
    t = make(glslTOP, name)
    d = text(name + '_pixel', rd('td', 'shaders', shader))
    setp(t, 'pixeldat', d.name)
    setp(t, 'outputresolution', 'custom')
    setp(t, 'resolutionw', W)
    setp(t, 'resolutionh', H)
    setp(t, 'format', 'rgba16float')
    wire(t, *inputs)
    # vecs: [(uniform_name, [expr x, y, z, w])]  -- expressions evaluated every frame
    for i, (uname, comps) in enumerate(vecs):
        setp(t, 'vec%dname' % i, uname)
        for c, e in zip('xyzw', comps):
            setp(t, 'vec%dvalue%s' % (i, c), str(e), expr=isinstance(e, str))
    return t


N = "op('portal_null')"          # the single source of truth for per-frame values

# ---- modules (DATs) ---------------------------------------------------------------------------------------------
text('protocol', rd('adapters', 'protocol.py'))
text('config_json', 'import json\nCFG = json.loads(%r)\n' % json.dumps(CFG))
text('portal_state', rd('td', 'scripts', 'portal_state.py'))
presenter_cb = text('presenter_cb', rd('td', 'scripts', 'presenter_chop.py'))
mask_cb = text('mask_cb', rd('td', 'scripts', 'mask_top.py'))
keys_cb = text('keys_cb', rd('td', 'scripts', 'keys.py'))
text('selfcheck', rd('td', 'selfcheck.py'))

# ---- camera -------------------------------------------------------------------------------------------------------
cam = make(videodeviceinTOP, 'cam')                 # pick the device manually if it is not the first camera
setp(cam, 'resolutionw', W)         # (no 'resolution' menu on 2025.32460; width/height pars are enough)
setp(cam, 'resolutionh', H)
cam_small = make(resolutionTOP, 'cam_small')        # RAW orientation, sent to the sidecar
wire(cam_small, cam)
setp(cam_small, 'outputresolution', 'custom')
setp(cam_small, 'resolutionw', 640)
setp(cam_small, 'resolutionh', 360)
cam_disp = make(flipTOP, 'cam_disp')                # display image: the ONE mirror decision (config camera.mirror)
wire(cam_disp, cam)
setp(cam_disp, 'flipx', bool(CFG['camera']['mirror']))

fw = make(executeDAT, 'frame_writer')
fw.text = rd('td', 'scripts', 'frame_writer.py')
setp(fw, 'active', True)
setp(fw, 'frameend', True)

# ---- sidecar link -------------------------------------------------------------------------------------------------
# (state packets are received by a socket thread inside portal_state.py; an old UDP In DAT would hold the port, so remove it)
for _old in ('udp_in', 'udp_callbacks'):
    if base.op(_old) is not None:
        base.op(_old).destroy()

# ---- CHOPs: presenter -> portal_null (single source of truth) -----------------------------------------------------
pres = make(scriptCHOP, 'presenter')
setp(pres, 'callbacks', presenter_cb.name)
portal_null = make(nullCHOP, 'portal_null')
wire(portal_null, pres)

keys = make(keyboardinCHOP, 'keys')
setp(keys, 'keys', '1 2 h')
kx = make(chopexecuteDAT, 'keys_exec')
kx.text = rd('td', 'scripts', 'keys.py')
setp(kx, 'chops', keys.name)
setp(kx, 'offtoon', True)

# ---- mask ----------------------------------------------------------------------------------------------------------
mask_in = make(scriptTOP, 'mask_in')
setp(mask_in, 'callbacks', mask_cb.name)
mask_disp = make(flipTOP, 'mask_disp')
wire(mask_disp, mask_in)
setp(mask_disp, 'flipx', bool(CFG['camera']['mirror']))

# ---- scenes -------------------------------------------------------------------------------------------------------
cell = CFG['td']['quality_cell_px']
scene_a = glsl('scene_a', 'scene_a.glsl', cam_disp, mask_disp, vecs=[
    ('uA', [N + "['time']", cell, 6.0, 1.0]),
    ('uB', [0.9, 1.1, 1.0, 0.0])])
scene_b = glsl('scene_b', 'scene_b.glsl', cam_disp, vecs=[
    ('uA', [N + "['time']", 0.42, 0.15, 0.0]),
    ('uB', [0.0, 1.0, 0.0, 0.0])])
scene_blend = glsl('scene_blend', 'scene_blend.glsl', scene_a, scene_b, vecs=[
    ('uMix', [N + "['scene_mix']", 0, 0, 0])])
scene_blur = make(blurTOP, 'scene_blur')            # cheap bloom source
wire(scene_blur, scene_blend)
setp(scene_blur, 'size', 14)

comp = glsl('composite', 'composite.glsl', cam_disp, scene_blend, scene_blur, vecs=[
    ('uQA', [N + "['q0x']", N + "['q0y']", N + "['q1x']", N + "['q1y']"]),
    ('uQB', [N + "['q2x']", N + "['q2y']", N + "['q3x']", N + "['q3y']"]),
    ('uP1', [N + "['alpha']", N + "['inset']", N + "['time']", 1.0]),     # last = tunnel on/off
    ('uP2', [N + "['par_x']", N + "['par_y']", 0.35, 2.0])])              # parallax, bloom gain, outline px

# ---- HUD + output --------------------------------------------------------------------------------------------------
hud = make(textTOP, 'hud_text')
setp(hud, 'text', "op('portal_state').module.hud_string()", expr=True)
setp(hud, 'outputresolution', 'custom')
setp(hud, 'resolutionw', W)
setp(hud, 'resolutionh', H)
setp(hud, 'fontsizex', 14)
setp(hud, 'alignx', 'left')
setp(hud, 'aligny', 'top')
hud_level = make(levelTOP, 'hud_level')            # opacity follows channel 'hud' (key H toggles it)
wire(hud_level, hud)
setp(hud_level, 'opacity', N + "['hud']", expr=True)
hud_comp = make(compositeTOP, 'hud_comp')
wire(hud_comp, comp, hud_level)
setp(hud_comp, 'operand', 'add')
out = make(nullTOP, 'out')
wire(out, hud_comp)

win = make(windowCOMP, 'win')
setp(win, 'winop', out.path)
setp(win, 'winw', W)
setp(win, 'winh', H)
setp(win, 'borders', False)

# ---- project timing -----------------------------------------------------------------------------------------------
try:
    project.cookRate = 60
    project.realTime = True
except Exception as e:
    issues.append('project timing: %s' % e)

print('portal network built in', base.path)
if issues:
    print('\nSet these by hand (see td/MANUAL_STEPS.md, section "Parameters TD rejected"):')
    for s in issues:
        print('  -', s)
else:
    print('all parameters applied.')
print('\nNext: start the sidecar (python adapters/mediapipe_sidecar.py), press F1 in the window, keys 1/2 scenes, H HUD.')
