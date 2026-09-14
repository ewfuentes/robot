"""Offline experiment guard: forbid network connections in Python processes."""
import sys

def deny_network(event, args):
    if event == "socket.connect":
        raise RuntimeError("Network connections disabled for overnight experiments")

sys.addaudithook(deny_network)
