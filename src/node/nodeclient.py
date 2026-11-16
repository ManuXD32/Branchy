"""Controller-side client for Branchy nodes (runs on the coordinating machine)."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import socket
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAYLOAD_SRC = HERE / 'payload'
sys.path.insert(0, str(PAYLOAD_SRC))
import nodewire  # noqa: E402

from peerclient import CHUNK, HASHED, IDEMPOTENT, Node, NodeError  # noqa: E402,F401


def payload_tarball(src: Path = PAYLOAD_SRC) -> tuple[bytes, str]:
    """Deterministic tar.gz of the payload folder; its sha256[:12] is the code version."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as gz, tarfile.open(fileobj=gz, mode='w') as tar:
        for path in sorted(src.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            info = tarfile.TarInfo(path.relative_to(src).as_posix())
            data = path.read_bytes()
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()
    return raw, hashlib.sha256(raw).hexdigest()[:12]


def load_cluster(path=None, only=None, delay_ms=None):
    """cluster.json -> {name: Node}. `only` filters by name."""
    path = Path(path or HERE / 'cluster.json')
    spec = json.loads(path.read_text())
    token_file = Path(spec['token_file'])
    if not token_file.is_absolute():           # relative: next to the cluster file, else from the repository root
        token_file = next((base / token_file for base in (path.parent, HERE.parent.parent) if (base / token_file).exists()), token_file)
    token = token_file.read_text().strip()
    nodes = {}
    for name, entry in spec['nodes'].items():
        if only and name not in only:
            continue
        nodes[name] = Node(name, entry['host'], entry.get('port', 8730), token,
                           delay_ms=entry.get('delay_ms', 0.0) if delay_ms is None else delay_ms)
    return nodes
