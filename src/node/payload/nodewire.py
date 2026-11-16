"""Branchy node wire protocol (shared by the daemon and the control client).

Frame:  !I header_len | header JSON (utf-8) | 32-byte HMAC-SHA256 | payload
The header always carries "n" = payload length. The MAC covers a direction tag,
a per-direction frame counter and the header bytes, keyed with a per-connection
session key, so the shared token never crosses the wire and frames cannot be
replayed or reordered. Payloads are NOT encrypted and are only MAC-protected
when the header carries "psha" (sha256 of the payload), which privileged ops
require. Intended for a trusted LAN / VPN.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import struct

PROTO = 1
MAX_HEADER = 1 << 16
MAX_PAYLOAD = 96 * 1024 * 1024
MAC_LEN = 32
SMALL = 1 << 16


class WireError(ConnectionError):
    pass


def _mac(key: bytes, *parts: bytes) -> bytes:
    return hmac.digest(key, b''.join(parts), 'sha256')


def recv_exact(sock, n: int) -> bytes:
    if n == 0:
        return b''
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if not r:
            raise WireError('peer disconnected')
        got += r
    return bytes(buf) if n < SMALL else buf


def tune(sock):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for opt, val in (('TCP_KEEPIDLE', 30), ('TCP_KEEPINTVL', 10), ('TCP_KEEPCNT', 6)):
        if hasattr(socket, opt):
            try:
                sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), val)
            except OSError:
                pass


def _send_plain(sock, obj):
    raw = json.dumps(obj, separators=(',', ':')).encode()
    sock.sendall(struct.pack('!I', len(raw)) + raw)


def _recv_plain(sock):
    n, = struct.unpack('!I', recv_exact(sock, 4))
    if not 1 <= n <= 4096:
        raise WireError('bad handshake frame')
    return json.loads(bytes(recv_exact(sock, n)))


class Channel:
    """Authenticated framed channel over a connected socket."""

    def __init__(self, sock, key: bytes, is_server: bool):
        self.sock = sock
        self.key = key
        self.tx_tag, self.rx_tag = (b'S', b'C') if is_server else (b'C', b'S')
        self.tx_seq = 0
        self.rx_seq = 0

    def send(self, header: dict, payload=b''):
        if len(payload) > MAX_PAYLOAD:
            raise ValueError('payload too large')
        raw = json.dumps(dict(header, n=len(payload)), separators=(',', ':')).encode()
        if len(raw) > MAX_HEADER:
            raise ValueError('header too large')
        self.tx_seq += 1
        mac = _mac(self.key, self.tx_tag, struct.pack('!Q', self.tx_seq), raw)
        head = struct.pack('!I', len(raw)) + raw + mac
        if len(payload) <= SMALL:
            self.sock.sendall(head + bytes(payload))
        else:
            self.sock.sendall(head)
            self.sock.sendall(payload)

    def recv(self):
        n, = struct.unpack('!I', recv_exact(self.sock, 4))
        if not 1 <= n <= MAX_HEADER:
            raise WireError('bad header size')
        raw = bytes(recv_exact(self.sock, n))
        mac = bytes(recv_exact(self.sock, MAC_LEN))
        self.rx_seq += 1
        if not hmac.compare_digest(mac, _mac(self.key, self.rx_tag, struct.pack('!Q', self.rx_seq), raw)):
            raise WireError('bad frame MAC')
        header = json.loads(raw)
        length = header.get('n', 0)
        if type(length) is not int or not 0 <= length <= MAX_PAYLOAD:
            raise WireError('bad payload length')
        payload = recv_exact(self.sock, length)
        if 'psha' in header and hashlib.sha256(payload).hexdigest() != header['psha']:
            raise WireError('payload hash mismatch')
        return header, payload


def server_handshake(sock, token: bytes, node_name: str) -> Channel:
    ns = os.urandom(16)
    _send_plain(sock, dict(hello='branchy-node', proto=PROTO, node=node_name, nonce=ns.hex()))
    reply = _recv_plain(sock)
    nc = bytes.fromhex(str(reply.get('nonce', '')))
    mac = bytes.fromhex(str(reply.get('mac', '')))
    if len(nc) != 16 or not hmac.compare_digest(mac, _mac(token, b'client', ns, nc)):
        _send_plain(sock, dict(ok=False, error='authentication failed'))
        raise WireError('authentication failed')
    _send_plain(sock, dict(ok=True, mac=_mac(token, b'server', ns, nc).hex()))
    return Channel(sock, _mac(token, b'session', ns, nc), True)


def client_handshake(sock, token: bytes) -> tuple[Channel, dict]:
    hello = _recv_plain(sock)
    if hello.get('hello') != 'branchy-node' or hello.get('proto') != PROTO:
        raise WireError(f'not a compatible branchy node: {hello}')
    ns = bytes.fromhex(hello['nonce'])
    nc = os.urandom(16)
    _send_plain(sock, dict(nonce=nc.hex(), mac=_mac(token, b'client', ns, nc).hex()))
    reply = _recv_plain(sock)
    if not reply.get('ok'):
        raise WireError('node rejected the token')
    if not hmac.compare_digest(bytes.fromhex(reply['mac']), _mac(token, b'server', ns, nc)):
        raise WireError('node failed to prove it knows the token')
    return Channel(sock, _mac(token, b'session', ns, nc), False), hello
