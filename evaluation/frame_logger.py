
import csv
import os

class FrameLogger:
    def __init__(self, filepath='frame_log.csv'):
        self.filepath = filepath
        self.fields = ['frame_id', 'x', 'y', 'z', 'pose_norm', 'flow_mag', 'depth_mean']
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.fields)

    def log(self, frame_id, pose, position, depth_map=None, flow_map=None):
        row = [frame_id, *position.round(3)]
        row.append(np.linalg.norm(pose[:3, 3]))
        if flow_map is not None:
            row.append(np.mean(np.linalg.norm(flow_map, axis=2)))
        else:
            row.append(None)
        if depth_map is not None:
            row.append(np.mean(depth_map))
        else:
            row.append(None)
        with open(self.filepath, 'a', newline='') as f:
            csv.writer(f).writerow(row)
