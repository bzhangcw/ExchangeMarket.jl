#!/usr/bin/env python3
"""Download the click log needed for the JD Fisher experiment (standard library only)."""
import argparse
import hashlib
from pathlib import Path
import time
import urllib.request
import urllib.error

REVISION = 'cbad25bd35202e98e6ae7fc23de4913695c28927'
SOURCE = 'https://huggingface.co/datasets/a6687543/MSOM_Data_Driven_Challenge_2020'
FILES = {
    'JD_click_data.csv': (920317454, '7e1ea745f1a49a0a223832b97247964f817e485d32df0fb332b85fb3f363d728'),
}

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def download(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in FILES.items():
        target = directory / name
        if target.exists():
            if target.stat().st_size == size and sha256(target) == digest:
                print(f'Verified existing {target}', flush=True)
                continue
            raise RuntimeError(f'{target} exists but does not match the pinned source; move it aside before retrying')
        partial = directory / (name + '.part')
        url = f'{SOURCE}/resolve/{REVISION}/{name}?download=true'
        for attempt in range(6):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == size:
                break
            request = urllib.request.Request(url + f'&offset={offset}&attempt={attempt}',
                headers={'Range': f'bytes={offset}-', 'User-Agent': 'JD-Fisher-reproduction/1.0'})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status == 206:
                        content_range = response.headers.get('Content-Range', '')
                        if not content_range.startswith(f'bytes {offset}-') or not content_range.endswith(f'/{size}'):
                            raise RuntimeError('Server returned an unexpected byte range')
                    elif response.status == 200:
                        offset = 0  # Range not supported: replace only our incomplete .part file.
                    else:
                        raise RuntimeError(f'Unexpected HTTP status {response.status}')
                    last_report = offset
                    with partial.open('ab' if offset else 'wb') as stream:
                        while True:
                            block = response.read(4*1024*1024)
                            if not block:
                                break
                            stream.write(block)
                            offset += len(block)
                            if offset-last_report >= 64*1024*1024:
                                print(f'{name}: {offset}/{size} bytes', flush=True)
                                last_report = offset
                if partial.stat().st_size == size:
                    break
            except (OSError, urllib.error.URLError) as exc:
                print(f'Download attempt {attempt+1} interrupted: {type(exc).__name__}', flush=True)
                if attempt == 5:
                    raise
                time.sleep(2)
        if partial.stat().st_size != size or sha256(partial) != digest:
            raise RuntimeError(f'Checksum mismatch for {partial}; download was not accepted')
        partial.replace(target)
        print(f'Verified {target}: {size} bytes, SHA256 {digest}', flush=True)
    # Preserve the public source card locally alongside the data.
    card = directory / 'SOURCE_README.md'
    if not card.exists():
        with urllib.request.urlopen(f'{SOURCE}/resolve/{REVISION}/README.md', timeout=60) as response:
            content = response.read()
        temporary = card.with_suffix('.part')
        temporary.write_bytes(content)
        temporary.replace(card)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent/'jd')
    args = parser.parse_args()
    download(args.data_dir)
