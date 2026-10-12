#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
# Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
# CLONE_GATE: sparse-router-formal::scripts::headers
"""Check tracked source notices without scanning dependencies or build output."""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUFFIXES = {'.rs', '.c', '.h', '.lean', '.pas', '.py', '.sh', '.toml', '.yml'}

def check():
    # Include newly added source files before a commit, without reading ignored tools.
    output = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT)
    failures = []
    count = 0
    for name in set(output.decode().split('\0')):
        path = ROOT / name
        if not name or not path.is_file() or (path.suffix not in SUFFIXES and path.name != 'Makefile'):
            continue
        count += 1
        head = path.read_text(encoding='utf-8')[:2500]
        # Preserve pre-existing Sovereign Leviathan notices without rewriting their terms.
        licensed = 'SPDX-License-Identifier:' in head or 'License-ID: SL-AGPL3-001' in head
        if not licensed or 'Copyright' not in head or 'CLONE_GATE:' not in head:
            failures.append(name)
    for name in sorted(failures):
        print('Missing licensing/attribution notice:', name)
    print(f'Checked {count} source files; {len(failures)} missing notices.')
    return 1 if failures else 0

if __name__ == '__main__':
    sys.exit(check())
