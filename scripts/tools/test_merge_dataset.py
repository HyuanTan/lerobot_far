from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

ds = LeRobotDataset("HollyTan/so101_pick-place-merge-v2.2-v2.3-v2.4_20hz")

print("fps:", ds.meta.fps)
print("features:", ds.meta.features)
print("length:", len(ds))

ts = []
for i in range(min(100, len(ds))):
    ts.append(float(ds[i]["timestamp"]))

print("first timestamps:", ts[:20])
print("diff:", np.diff(ts[:20]))