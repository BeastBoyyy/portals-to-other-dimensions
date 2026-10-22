# Paste in the TouchDesigner Textport:  REPO = r'D:\TDMP\portal'; exec(open(REPO + r'\td\diagnose.py').read())
# Prints what is wrong with the camera -> frame writer -> sidecar chain. Changes nothing.
b = op('/project1/portal')
print('--- portal diagnose ---')
if b is None:
    print('NO /project1/portal : the build did not run in this project. Run build_network.py.')
else:
    def say(label, fn):
        try:
            print('%-34s %s' % (label, fn()))
        except Exception as e:
            print('%-34s ERROR %s' % (label, e))
    say('protocol has v2 shm name', lambda: 'portal_frame_v2' in b.op('protocol').text)
    say('frame_writer has size guard', lambda: 'MIN_SIDE' in b.op('frame_writer').text)
    say('frame_writer active / frameend', lambda: (b.op('frame_writer').par.active.eval(), b.op('frame_writer').par.frameend.eval()))
    say('cam device', lambda: b.op('cam').par.device.eval())
    say('cam size', lambda: (b.op('cam').width, b.op('cam').height))
    say('cam_small size', lambda: (b.op('cam_small').width, b.op('cam_small').height))
    say('cam_small numpy shape', lambda: b.op('cam_small').numpyArray(delayed=False).shape)
    say('cam_small mean brightness', lambda: float(b.op('cam_small').numpyArray(delayed=False)[..., :3].mean()))
    say('frames written so far', lambda: b.op('portal_state').module.frames_written)
    say('udp packets received', lambda: b.op('portal_state').module.n_recv)
    say('udp receiver thread alive', lambda: __import__('builtins')._portal_rx[1].is_alive())
    for n in ('frame_writer', 'portal_state', 'presenter', 'mask_in', 'scene_a', 'composite'):
        say('errors in ' + n, lambda n=n: (b.op(n).errors() or 'none'))
    print('--- picture chain: mean RGB per stage (0 = black) ---')
    def stat(n):
        o = b.op(n)
        a = o.numpyArray(delayed=False)
        return 'size %dx%d  mean %.3f  max %.3f' % (o.width, o.height, float(a[..., :3].mean()), float(a[..., :3].max()))
    for n in ('cam', 'cam_disp', 'mask_in', 'mask_disp', 'scene_a', 'scene_b', 'scene_blend', 'scene_blur', 'composite', 'hud_text', 'hud_level', 'hud_comp', 'out'):
        say('TOP ' + n, lambda n=n: stat(n))
    for n in ('scene_a', 'scene_b', 'scene_blend', 'composite'):
        say('warnings ' + n, lambda n=n: (b.op(n).warnings() or 'none'))
        say('errors ' + n, lambda n=n: (b.op(n).errors() or 'none'))
    say('win winop', lambda: b.op('win').par.winop.eval())
    say('portal_null alpha/valid/hud/mix', lambda: tuple(round(b.op('portal_null')[k].eval(), 3) for k in ('alpha', 'valid', 'hud', 'scene_mix')))
    say('portal_null q0..q3', lambda: [round(b.op('portal_null')[k].eval(), 3) for k in ('q0x', 'q0y', 'q1x', 'q1y', 'q2x', 'q2y', 'q3x', 'q3y')])
    say('composite input0 / glsl par', lambda: (b.op('composite').inputs[0].name, b.op('composite').par.pixeldat.eval()))
    print('Camera lines show 0 or a tiny size: pick the webcam in cam > Device. All good but frames written = 0: frame_writer inactive or erroring.')
