# UDP In DAT callbacks (needs user verification in TD). Binary datagrams, one per camera frame.
def onReceive(dat, rowIndex, message, byteData, peer):
    op('portal_state').module.on_packet(byteData)
    return
