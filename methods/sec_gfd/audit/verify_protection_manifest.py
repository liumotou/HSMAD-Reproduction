"""Verify an existing protected-artifact SHA256 manifest without altering it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from methods.project_paths import project_root

ROOT = project_root()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


before_path = ROOT / 'methods/sec_gfd/audit/protection_manifest_before.json'
before = json.loads(before_path.read_text(encoding='utf-8'))
current: dict[str, str | None] = {}
expected: dict[str, str] = {entry['path']: entry['sha256'] for entry in before['files']}
for relative in expected:
    path = ROOT / relative
    current[relative] = digest(path) if path.exists() else None
changed = {path: {'before': expected[path], 'after': current[path]}
           for path in expected if expected[path] != current[path]}
result = {
    'baseline_count': before['count'],
    'changed_count': len(changed),
    'changed': changed,
    'status': 'PASS' if not changed else 'FAIL',
}
(ROOT / 'methods/sec_gfd/audit/protection_manifest_after_weibo_formal.json').write_text(
    json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
print(json.dumps(result, sort_keys=True))
