from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from feeders import tools

# 1-indexed pairs for bone modality (optional)
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
    Expects directory:
      data_path/
        train/<class_name>/*.npz
        test/<class_name>/*.npz

    Each npz:
      keypoints_3d: (T, P, 17, 3)  (P <= num_person)
    Returns:
      data_numpy: (C=3, T, V=17, M=num_person)
    """
    def __init__(self, data_path, split='train', p_interval=1,
                 random_choose=False, random_shift=False, random_move=False,
                 random_rot=False, window_size=-1, normalization=False,
                 debug=False, use_mmap=False, bone=False, vel=False,
                 num_person=2, key='keypoints_3d'):
        self.data_path = Path(data_path)
        self.split = split
        self.p_interval = p_interval
        self.window_size = window_size
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.num_person = num_person
        self.key = key
        self.debug = debug

        self._index_files_and_labels()

        if normalization:
            self.get_mean_map()

    def _index_files_and_labels(self):
        split_dir = self.data_path / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split dir: {split_dir}")

        # class names sorted => stable label ids
        class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        if len(class_dirs) == 0:
            raise RuntimeError(f"No class folders found in {split_dir}")

        self.class_names = [p.name for p in class_dirs]
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}

        self.files = []
        self.labels = []
        for cdir in class_dirs:
            lab = self.class_to_id[cdir.name]
            for fp in sorted(cdir.rglob("*.npz")):
                self.files.append(fp)
                self.labels.append(lab)

        if self.debug:
            self.files = self.files[:100]
            self.labels = self.labels[:100]

        self.labels = np.array(self.labels, dtype=np.int64)
        self.sample_name = [str(p.relative_to(self.data_path)) for p in self.files]

    def __len__(self):
        return len(self.labels)

    def _to_ctvm(self, x):
        # x: (T,P,17,3)
        if x.ndim != 4:
            raise ValueError(f"Expected (T,P,17,3) but got {x.shape}")
        T, P, V, C = x.shape
        if V != 17 or C != 3:
            raise ValueError(f"Expected V=17,C=3 but got V={V},C={C}")

        M = self.num_person
        if P < M:
            pad = np.zeros((T, M - P, V, C), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=1)
        elif P > M:
            x = x[:, :M]

        # (T,M,V,C) -> (C,T,V,M)
        return x.transpose(3, 0, 2, 1).astype(np.float32)

    def __getitem__(self, index):
        fp = self.files[index]
        label = int(self.labels[index])

        d = np.load(fp, allow_pickle=True)
        x = d[self.key]  # (T,P,17,3)
        data_numpy = self._to_ctvm(x)  # (C,T,V,M)

        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)

        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        if self.bone:
            bone_data = np.zeros_like(data_numpy)
            for v1, v2 in coco17_pairs:
                bone_data[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index