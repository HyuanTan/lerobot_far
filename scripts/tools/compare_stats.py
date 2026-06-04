"""
uv run python compare_stats.py old_stats.json new_stats.json

uv run python scripts/tools/compare_stats.py /data/users/huoyuan/.cache/huggingface/lerobot/HollyTan/so101_pick-place-v2.2-100eps/meta/stats.json /data/users/huoyuan/.cache/huggingface/lerobot/HollyTan/so101_pick-place-v2.2-100eps_relative/meta/stats.json

uv run python scripts/tools/compare_stats.py /data/users/huoyuan/.cache/huggingface/lerobot/HollyTan/so101_pick-place-v2.4/meta/stats.json /data/users/huoyuan/.cache/huggingface/lerobot/HollyTan/so101_pick-place-v2.4_relative/meta/stats.json
"""
import json
import sys
import numpy as np

if len(sys.argv) != 3:
    print("Usage: python compare_stats.py old_stats.json new_stats.json")
    sys.exit(1)

old_path, new_path = sys.argv[1], sys.argv[2]

with open(old_path, "r") as f:
    old = json.load(f)

with open(new_path, "r") as f:
    new = json.load(f)


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            out.update(flatten(v, key))
    else:
        out[prefix] = d
    return out


old_f = flatten(old)
new_f = flatten(new)

keys = sorted(set(old_f) | set(new_f))

print("Changed numeric/list fields:")
print("=" * 100)

for k in keys:
    if k not in old_f:
        print(f"[ADDED] {k}: {new_f[k]}")
        continue

    if k not in new_f:
        print(f"[REMOVED] {k}: {old_f[k]}")
        continue

    a = old_f[k]
    b = new_f[k]

    if a == b:
        continue

    # Only show action / observation.state
    if not (
        k.startswith("action")
        or k.startswith("observation.state")
        or "action" in k
    ):
        continue

    try:
        aa = np.array(a, dtype=float)
        bb = np.array(b, dtype=float)

        if aa.shape == bb.shape:
            diff = bb - aa
            print(f"\n{k}")
            print(f"  old: {aa}")
            print(f"  new: {bb}")
            print(f"  diff: {diff}")
            print(f"  max_abs_diff: {np.max(np.abs(diff))}")
        else:
            print(f"\n{k}")
            print(f"  shape changed: {aa.shape} -> {bb.shape}")
            print(f"  old: {a}")
            print(f"  new: {b}")

    except Exception:
        print(f"\n{k}")
        print(f"  old: {a}")
        print(f"  new: {b}")