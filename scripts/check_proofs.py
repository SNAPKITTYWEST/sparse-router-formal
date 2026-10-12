#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
# Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
# CLONE_GATE: sparse-router-formal::scripts::proofs
import pathlib
import re
import sys

root = pathlib.Path(__file__).resolve().parents[1] / 'formal'
failures = []
for path in root.rglob('*.lean'):
    if '.lake' in path.parts:
        continue
    source = path.read_text(encoding='utf-8')
    source = re.sub(r'/\-.*?\-/', '', source, flags=re.S)
    source = re.sub(r'--[^\n]*', '', source)
    if not source.strip() or re.search(r'\b(sorry|admit|axiom)\b', source):
        failures.append(str(path.relative_to(root)))
print('Proof source check:', 'PASS' if not failures else ', '.join(failures))
sys.exit(bool(failures))
