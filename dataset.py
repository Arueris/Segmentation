import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import zipfile

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

class RetCurvDataset(Dataset):
    def __init__(self,
                 pIds: list[str],
                 path_raw: str = r"\\?\UNC\zvsl.mvl6.uni-tuebingen.de\Employees\06_Datasets\RetCurv\Plex Elite",
                 path_seg: str = r"\\?\UNC\zvsl.mvl6.uni-tuebingen.de\Employees\06_Datasets\RetCurv\OCT_Segmented",
                 augment: bool = True,
                 hflip_p: float = 0.5,
                 normalize: str = "zscore",  # "none" | "minmax" | "zscore",
                 resize: tuple[int, int] | None = (512, 512)
                 ):
        self.pIds = pIds
        self.root_raw = Path(path_raw)
        self.root_seg = Path(path_seg)
        self.augment = augment
        self.hflip_p = float(hflip_p)
        self.normalize = normalize

        self.resize = resize    # it is not used just because of the pretrained encoder use this attribute from the other dataset

        zip_files = list(self.root_seg.rglob("*MLS_Layer_location_maps.zip"))

        self.samples = list()
        for pId in self.pIds:
            p_path = self.root_raw / pId
            cube_names = [x for x in os.listdir(p_path) if "_Cube" in x]
            for cube_name in cube_names:
                zip_file_matches = [x for x in zip_files if pId in str(x) and cube_name.strip(pId) in str(x)]

                if len(zip_file_matches) == 1:
                    for i in range(1024):
                        self.samples.append((p_path / cube_name, zip_file_matches[0], i))
    
    def __len__(self):
        return len(self.samples)
    
    def _load_image(self, path: Path, idx: int) -> np.ndarray:
        cube_path = next(path.glob("*_cube_z.img"))
        cube = np.memmap(cube_path, dtype=np.uint8, mode='r', shape=(1024, 1536, 1024))
        bscan = cube[idx]
        bscan = np.flip(bscan, axis=0)
        return bscan
    
    def _load_mask(self, zip_path: Path, idx: int) -> np.ndarray:
        with zipfile.ZipFile(zip_path, 'r') as z:
            files = z.namelist()
            bm_file = next(f for f in files if "BMSeg_location.png" in f)
            chor_file = next(f for f in files if "choroidSeg_location.png" in f)
            bm = np.array(Image.open(z.open(bm_file)))
            chor = np.array(Image.open(z.open(chor_file)))
        bm_anno = bm[idx]
        bm_anno = np.flip(bm_anno, axis=0)
        chor_anno = chor[idx]
        chor_anno = np.flip(chor_anno, axis=0)
        y = np.arange(1536)[:, None]
        mask = ((y >= bm_anno) & (y <= chor_anno)).astype(np.uint8)
        return mask
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return img
        if self.normalize == "minmax":
            mn, mx = float(img.min()), float(img.max())
            if mx > mn:
                return (img - mn) / (mx - mn)
            return img * 0.0
        if self.normalize == "zscore":
            img = (img - img.mean()) / (img.std() + 1e-8)
            return img
        raise ValueError(f"Unknown normalize='{self.normalize}'")

    def _augment_pair(self, img_t: torch.Tensor, mask_t: torch.Tensor):
        # img_t: (1,H,W) float32, mask_t: (1,H,W) uint8/float
        # Horizontal flip
        if random.random() < self.hflip_p:
            img_t = TF.hflip(img_t)
            mask_t = TF.hflip(mask_t)
        return img_t, mask_t
    
    def __getitem__(self, idx):
        cube_path, zip_path, bscan_idx = self.samples[idx]
        img = self._load_image(cube_path, bscan_idx)   # (H,W) uint8
        mask = self._load_mask(zip_path, bscan_idx)    # (H,W) uint8 {0,1}

        img = self._normalize_image(img)

        img_t = torch.from_numpy(img).unsqueeze(0).float()                 # (1,H,W) float32
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()               # (1,H,W) float32 {0,1}

        if self.augment:
            img_t, mask_t = self._augment_pair(img_t, mask_t)

        return img_t, mask_t


class OIMHSDataset(Dataset):
    def __init__(self,
                 participants: list[str],
                 path: str = r"F:/Python/SAM2/OCTDatasetOIMHS",
                 augment: bool = True,
                 max_rotate_deg: float = 0,
                 hflip_p: float = 0.5,
                 return_numpy: bool = False,
                 normalize: str = "zscore",  # "none" | "minmax" | "zscore",
                 resize: tuple[int, int] | None = (512, 512)
                 ):
        """
        Dataset für 2D OCT B-Scans + binäre Choroid-Maske.

        Args:
            participants: Liste der Participant-Ordnernamen.
            path: Root-Pfad mit Unterordnern Images/<participant> und Annotations/<participant>.
            augment: Ob Augmentationen angewendet werden (nur für Train=True).
            max_rotate_deg: Maximaler Rotationswinkel in Grad (Uniform [-max, +max]).
            hflip_p: Wahrscheinlichkeit für horizontalen Flip.
            return_numpy: Wenn True, gib (H,W) numpy arrays zurück, sonst Torch-Tensoren (C,H,W).
            normalize: Normalisierung für Bild: "none", "minmax", "zscore".
        """
        self.participants = participants
        self.root = Path(path)
        self.image_path = self.root / "Images"
        self.annotation_path = self.root / "Annotations"

        self.augment = augment
        self.max_rotate_deg = float(max_rotate_deg)
        self.hflip_p = float(hflip_p)
        self.return_numpy = return_numpy
        self.normalize = normalize

        self.resize = resize    # it is not used just because of the pretrained encoder use this attribute from the other dataset

        # --- Paare über Dateinamen matchen, damit Bild <-> Maske korrekt ---
        self.samples = []
        for participant in self.participants:
            img_dir = self.image_path / participant
            ann_dir = self.annotation_path / participant

            img_files = sorted(img_dir.glob("*.png"))
            # Map annotation by stem (filename without suffix)
            ann_map = {p.stem: p for p in ann_dir.glob("*.png")}

            for img_p in img_files:
                ann_p = ann_map.get(img_p.stem, None)
                if ann_p is not None:
                    self.samples.append((img_p, ann_p))
                # Wenn du Missing-Files debuggen willst, kannst du hier loggen.

        if len(self.samples) == 0:
            raise RuntimeError("Keine (Bild,Maske)-Paare gefunden. Prüfe Pfade/Teilnehmernamen/Struktur.")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: Path) -> np.ndarray:
        # OCT meist 1-kanalig -> "L"
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32)  # (H,W)
        return arr

    def _load_mask(self, path: Path) -> np.ndarray:
        # Maske als 0/255 oder 0/1 -> binär machen
        m = Image.open(path)
        arr = np.array(m, dtype=np.uint8)
        # Alles >0 als Choroid
        # arr = (arr > 0).astype(np.uint8)
        # yellow: (255,255,0) -> Choroid, alles andere Hintergrund
        arr = ((arr[:, :, 0] == 255) & (arr[:, :, 1] == 255) & (arr[:, :, 2] == 0)).astype(np.uint8)
        return arr

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return img
        if self.normalize == "minmax":
            mn, mx = float(img.min()), float(img.max())
            if mx > mn:
                return (img - mn) / (mx - mn)
            return img * 0.0
        if self.normalize == "zscore":
            img = (img - img.mean()) / (img.std() + 1e-8)
            return img
        raise ValueError(f"Unknown normalize='{self.normalize}'")

    def _augment_pair(self, img_t: torch.Tensor, mask_t: torch.Tensor):
        # img_t: (1,H,W) float32, mask_t: (1,H,W) uint8/float
        # Horizontal flip
        if random.random() < self.hflip_p:
            img_t = TF.hflip(img_t)
            mask_t = TF.hflip(mask_t)

        # Kleine Rotation
        if self.max_rotate_deg > 0:
            angle = random.uniform(-self.max_rotate_deg, self.max_rotate_deg)
            img_t = TF.rotate(img_t, angle=angle, interpolation=InterpolationMode.BILINEAR)
            mask_t = TF.rotate(mask_t, angle=angle, interpolation=InterpolationMode.NEAREST)

        return img_t, mask_t

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = self._load_image(img_path)   # (H,W) float32
        mask = self._load_mask(mask_path)  # (H,W) uint8 {0,1}

        img = self._normalize_image(img)

        if self.return_numpy:
            # Für numpy: optional augmentations in numpy wäre möglich, aber Torch ist sauberer.
            # Daher: wir machen Augmentations in Torch und konvertieren zurück.
            img_t = torch.from_numpy(img).unsqueeze(0)              # (1,H,W)
            mask_t = torch.from_numpy(mask).unsqueeze(0).float()    # (1,H,W)

            if self.augment:
                img_t, mask_t = self._augment_pair(img_t, mask_t)

            # zurück zu numpy (H,W)
            img_out = img_t.squeeze(0).numpy().astype(np.float32)
            mask_out = mask_t.squeeze(0).numpy().astype(np.float32)  # {0,1}
            return img_out, mask_out

        # Standard: Torch-Tensoren zurückgeben
        img_t = torch.from_numpy(img).unsqueeze(0)                 # (1,H,W) float32
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()       # (1,H,W) float32 {0,1}

        if self.augment:
            img_t, mask_t = self._augment_pair(img_t, mask_t)

        return img_t, mask_t
    


def get_dataloader(participants: list[str], train_portion: float = 0.8,
                   path = r"F:/Python/SAM2/OCTDatasetOIMHS", augment: bool = True, 
                   max_rotate_deg: float = 0, hflip_p: float = 0.5, 
                   return_numpy: bool = False, normalize: str = "none",
                   batch_size: int = 16, num_workers: int = 0):
    n_train = int(len(participants) * train_portion)
    dataset_train = OIMHSDataset(participants[:n_train],
                           path = path, 
                           augment = augment, 
                           max_rotate_deg = max_rotate_deg, 
                           hflip_p = hflip_p, 
                           return_numpy = return_numpy, 
                           normalize = normalize)
    dataset_test = OIMHSDataset(participants[n_train:], 
                         path = path, 
                         augment = False,  # Keine Augmentierungen für Validierung
                         max_rotate_deg = 0, 
                         hflip_p = 0, 
                         return_numpy = return_numpy, 
                         normalize = normalize)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def get_dataloader_encoder_pretraining(participants: list[str],
                                       portions: tuple[float, float, float] = (0.65, 0.05, 0.3), 
                                       path = r"F:/Python/SAM2/OCTDatasetOIMHS", 
                                       augment: bool = True, 
                                       max_rotate_deg: float = 0, 
                                       hflip_p: float = 0.5, 
                                       return_numpy: bool = False, 
                                       normalize: str = "none",
                                       batch_size: int = 16, 
                                       num_workers: int = 0):

    if sum(portions) != 1.0:
        raise ValueError("Portions must sum to 1.0")
    n = len(participants)
    n_pretrain = int(n * portions[0])
    n_train = int(n * portions[1])    

    dataset_pretrain = OIMHSDataset(participants[:n_pretrain],
                             path = path, 
                             augment = augment, 
                             max_rotate_deg = max_rotate_deg, 
                             hflip_p = hflip_p, 
                             return_numpy = return_numpy, 
                             normalize = normalize)
    dataset_train = OIMHSDataset(participants[n_pretrain:n_pretrain+n_train], 
                         path = path, 
                         augment = augment,  # Keine Augmentierungen für Validierung
                         max_rotate_deg = max_rotate_deg, 
                         hflip_p = hflip_p, 
                         return_numpy = return_numpy, 
                         normalize = normalize)
    dataset_test = OIMHSDataset(participants[n_pretrain+n_train:], 
                         path = path, 
                         augment = False,  # Keine Augmentierungen für Validierung
                         max_rotate_deg = 0, 
                         hflip_p = 0, 
                         return_numpy = return_numpy, 
                         normalize = normalize)
    pretrain_loader = torch.utils.data.DataLoader(dataset_pretrain, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return pretrain_loader, train_loader, test_loader


if __name__ == "__main__":
    import os
    path: str=Path(r"F:/Python/SAM2/OCTDatasetOIMHS")
    participants = ["1", "2"] #  os.listdir(path / "Images")
    dataset = OIMHSDataset(participants, path)
    
