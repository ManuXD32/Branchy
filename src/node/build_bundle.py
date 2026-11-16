#!/usr/bin/env python3
"""Build dist/branchy-node.zip (stage 0 + initial payload) and dist/go.sh."""
import argparse
import io
import tarfile
import zipfile
from pathlib import Path

from nodeclient import payload_tarball

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', default='http://COORDINATOR_HOST:8765', help='URL the devices download from')
    ap.add_argument('--out', type=Path, default=ROOT / 'dist')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tarball, version = payload_tarball()
    with zipfile.ZipFile(args.out / 'branchy-node.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ('start.sh', 'stop.sh', 'supervisor.py'):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 11, 0, 0, 0))
            info.external_attr = 0o755 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (HERE / 'stage0' / name).read_bytes())
        import gzip
        with tarfile.open(fileobj=io.BytesIO(gzip.decompress(tarball)), mode='r') as tar:
            for member in tar:
                z.writestr(f'payload/{version}/{member.name}', tar.extractfile(member).read())
        z.writestr('payload/CURRENT', version)
        z.writestr('VERSION', version)
    go = (HERE / 'stage0' / 'go.sh').read_text().replace('__BASE__', args.base)
    (args.out / 'go.sh').write_text(go)
    print(f'bundle {args.out / "branchy-node.zip"} payload version {version} '
          f'({(args.out / "branchy-node.zip").stat().st_size} bytes)')


if __name__ == '__main__':
    main()
