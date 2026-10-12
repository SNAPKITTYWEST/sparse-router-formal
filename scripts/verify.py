#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
# Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
# CLONE_GATE: sparse-router-formal::scripts::release-check
"""Sequential CPU-only validation for Unix and Windows."""
import argparse
import os
import pathlib
import subprocess
import sys
ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cc', default='cc')
    parser.add_argument('--cc-arg', action='append', default=[])
    parser.add_argument('--cargo', default='cargo')
    parser.add_argument('--lake', default='lake')
    args = parser.parse_args()
    env = dict(os.environ, CARGO_BUILD_JOBS='1', LEAN_NUM_THREADS='1')
    manifest = str(ROOT / 'rust/tensor-hw/Cargo.toml')
    build = ROOT / 'build'
    build.mkdir(exist_ok=True)
    executable = build / ('test_router.exe' if os.name == 'nt' else 'test_router')
    commands = [
        [sys.executable, 'scripts/check_headers.py'],
        [sys.executable, 'scripts/check_proofs.py'],
        [args.cc, *args.cc_arg, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-UNDEBUG',
         '-I', 'c-reference', 'c-reference/tests/test_router.c', '-o', str(executable)],
        [str(executable)],
        [args.cargo, 'fmt', '--manifest-path', manifest, '--check'],
        [args.cargo, 'clippy', '--locked', '--manifest-path', manifest, '--all-targets', '--all-features', '--', '-D', 'warnings'],
        [args.cargo, 'test', '--locked', '--manifest-path', manifest, '--all-targets', '--all-features'],
        [args.cargo, 'test', '--locked', '--release', '--manifest-path', manifest],
        [args.cargo, 'run', '--locked', '--manifest-path', manifest, '--example', 'core_operations'],
        [args.lake, '-d', 'formal', 'build'],
    ]
    for command in commands:
        print('Running:', ' '.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    print('All local release checks passed. Licensing reconciliation and target CI remain required.')

if __name__ == '__main__':
    try:
        main()
    except (subprocess.CalledProcessError, OSError) as error:
        print('Verification failed:', error, file=sys.stderr)
        sys.exit(1)
