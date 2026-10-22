# CHOP Execute DAT on Keyboard In CHOP 'keys' (channels k1, k2, h)  -- needs user verification in TD
def onOffToOn(channel, sampleIndex, val, prev):
    name = channel.name
    S = op('portal_state').module
    if name in ('k1', 'k2'):
        S.key(name[1])
    elif name == 'kh':
        S.key('h')
    return
