import numpy as np
from torch.utils.data import Dataset
from feeders import tools

# 1-indexed COCO17 edges for bone computations (x,y only)
coco17_pairs = [
    (1, 2), (1, 3), (2, 4), (3, 5),
    (6, 7),
    (6, 8), (8, 10),
    (7, 9), (9, 11),
    (6, 12), (7, 13),
    (12, 13),
    (12, 14), (14, 16),
    (13, 15), (15, 17),
]

class Feeder(Dataset):
    """
    Reads a single .npz with keys:
      x_train, y_train, x_test, y_test

    split='train' uses x_train/y_train
    split='test'  uses x_test/y_test

    Supports x_* as:
      A) (N, T, 102) where 102 = 2(person)*17(joint)*3(x,y,conf)
         -> reshaped to (N, 3, T, 17, 2)
      B) (N, 3, T, 17, 2) already.
    """
    def __init__(
        self,
        data_path,
        split="train",
        p_interval=1,
        random_choose=False,
        random_shift=False,
        random_move=False,
        random_rot=False,     # MUST stay False for (x,y,conf)
        window_size=-1,
        normalization=False,
        debug=False,
        use_mmap=False,
        bone=False,
        vel=False,
        num_person=2,
        num_point=17,
        in_channels=3,
    ):
        self.data_path = data_path
        self.split = split
        self.p_interval = p_interval
        self.window_size = window_size
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.debug = debug

        self.num_person = int(num_person)
        self.num_point = int(num_point)
        self.in_channels = int(in_channels)

        self.load_data()

        if normalization:
            self.get_mean_map()

    def load_data(self):
        d = np.load(self.data_path, allow_pickle=True)
        if self.split == "train":
            x = d["x_train"]
            y = d["y_train"]
            self.sample_name = [f"train_{i}" for i in range(len(x))]
        elif self.split == "test":
            x = d["x_test"]
            y = d["y_test"]
            self.sample_name = [f"test_{i}" for i in range(len(x))]
        else:
            raise NotImplementedError("split must be 'train' or 'test'")

        if self.debug:
            x = x[:100]
            y = y[:100]
            self.sample_name = self.sample_name[:100]

        # labels: one-hot -> class index
        self.label = np.where(y > 0)[1].astype(np.int64)

        # Convert x to (N,C,T,V,M)
        self.data = self._to_nctvm(x).astype(np.float32)

    def _to_nctvm(self, x):
        """
        Returns (N,C,T,V,M)
        """
        if x.ndim == 5:
            # already (N,C,T,V,M)
            N, C, T, V, M = x.shape
            if (C, V, M) != (self.in_channels, self.num_point, self.num_person):
                raise ValueError(f"x has shape {x.shape} but expected (N,{self.in_channels},T,{self.num_point},{self.num_person})")
            return x

        if x.ndim != 3:
            raise ValueError(f"Expected x dims 3 or 5, got {x.shape}")

        # x: (N, T, D)
        N, T, D = x.shape
        expected_D = self.num_person * self.num_point * self.in_channels
        if D != expected_D:
            raise ValueError(f"Expected last dim D={expected_D} but got D={D} in {x.shape}")

        # reshape to (N, T, M, V, C) then transpose to (N, C, T, V, M)
        x = x.reshape(N, T, self.num_person, self.num_point, self.in_channels)
        x = x.transpose(0, 4, 1, 3, 2)
        return x

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])   # (C,T,V,M)
        data_numpy[2] = np.clip(data_numpy[2], 0.0, 1.0)  # ensure confidence is in [0,1]
        label = int(self.label[index])

        # valid frames based on x,y only (ignore confidence)
        xy = data_numpy[:2]
        valid_frame_num = np.sum(xy.sum(0).sum(-1).sum(-1) != 0)

        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        xy = data_numpy[:2]  # (2,T,V,M)

        mask = (xy != 0)
        if mask.any():
            vals = xy[mask]
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med))) + 1e-6  # robust scale
            xy = (xy - med) / (mad * 1.4826)  # ~robust z-score
            data_numpy[:2] = np.clip(xy, -10.0, 10.0)  # prevent extreme tails
        # IMPORTANT: for (x,y,conf), random_rot is NOT meaningful
        if self.random_rot:
            raise ValueError("random_rot=True is not valid for (x,y,confidence). Set random_rot: False.")

        if self.bone:
            bone_data = np.zeros_like(data_numpy)
            # x,y bones; keep confidence unchanged for each joint
            bone_data[2:3] = data_numpy[2:3]
            for v1, v2 in coco17_pairs:
                bone_data[0:2, :, v1 - 1] = data_numpy[0:2, :, v1 - 1] - data_numpy[0:2, :, v2 - 1]
            data_numpy = bone_data

        if self.vel:
            vel_data = np.zeros_like(data_numpy)
            # x,y velocities; keep confidence as-is (or also diff it if you prefer)
            vel_data[2:3] = data_numpy[2:3]
            vel_data[0:2, :-1] = data_numpy[0:2, 1:] - data_numpy[0:2, :-1]
            vel_data[0:2, -1] = 0
            data_numpy = vel_data

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)