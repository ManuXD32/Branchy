"""Memory and storage counters of the control group this process runs in (Linux cgroup v2), for the low-memory
experiments: how much the device holds in RAM and how much it reads back from storage. Empty dict where unavailable."""
from __future__ import annotations

from pathlib import Path

KEYS = ('anon', 'file', 'file_mapped', 'kernel', 'pgmajfault', 'workingset_refault_file')


def _cgroup():
    try:
        rel = [line.split('::', 1)[1].strip() for line in open('/proc/self/cgroup') if line.startswith('0::')][0]
        folder = Path('/sys/fs/cgroup') / rel.lstrip('/')
        return folder if (folder / 'memory.stat').is_file() else None
    except Exception:  # noqa: BLE001
        return None


def snapshot() -> dict:
    out = {}
    try:
        folder = _cgroup()
        if folder is not None:
            stat = dict(line.split()[:2] for line in (folder / 'memory.stat').read_text().splitlines())
            out.update({k: int(stat.get(k, 0)) for k in KEYS})
            out['current'] = int((folder / 'memory.current').read_text())
            limit = (folder / 'memory.max').read_text().strip()
            out['max'] = None if limit == 'max' else int(limit)
            read = 0
            if (folder / 'io.stat').is_file():
                for line in (folder / 'io.stat').read_text().splitlines():
                    # one line per block device; stacked devices (LVM on NVMe) report the same bytes twice: take the largest
                    read = max(read, sum(int(f[7:]) for f in line.split() if f.startswith('rbytes=')))
            out['io_read_bytes'] = read
        for line in open('/proc/self/io'):
            if line.startswith('read_bytes:'):
                out['proc_read_bytes'] = int(line.split()[1])
    except Exception:  # noqa: BLE001
        pass
    return out


def delta(before: dict, after: dict) -> dict:
    """Counters as differences, gauges as the later value."""
    out = {}
    for key, value in after.items():
        if key in ('pgmajfault', 'workingset_refault_file', 'io_read_bytes', 'proc_read_bytes') and value is not None:
            out[key] = value - before.get(key, 0)
        else:
            out[key] = value
    return out
