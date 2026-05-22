import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp
from torchvision import transforms
import torch.nn.functional as F

# =========================
# DATASET
# =========================

class OCTNormalDrusenCNV(Dataset):
    def __init__(self, root, normalize="zscore", resize=(512, 512), normal: bool = True, drusen: bool = False, cnv: bool = False):
        self.root = Path(root) / "Images"
        self.image_paths = list()
        self.resize = resize

        if normal:
            normal_path = self.root / "NORMAL"
            normal_participants = sorted(list(normal_path.glob("*")))
            for participant in normal_participants:
                self.image_paths.extend(sorted(list((participant / "OD").glob("*.jpg"))))
                self.image_paths.extend(sorted(list((participant / "OS").glob("*.jpg"))))
        if drusen:
            drusen_path = self.root / "DRUSEN"
            drusen_participants = sorted(list(drusen_path.glob("*")))
            for participant in drusen_participants:
                self.image_paths.extend(sorted(list((participant / "OD").glob("*.jpg"))))
                self.image_paths.extend(sorted(list((participant / "OS").glob("*.jpg"))))
        if cnv:
            cnv_path = self.root / "CNV"
            cnv_participants = sorted(list(cnv_path.glob("*")))
            for participant in cnv_participants:
                self.image_paths.extend(sorted(list((participant / "OD").glob("*.jpg"))))
                self.image_paths.extend(sorted(list((participant / "OS").glob("*.jpg"))))


        self.normalize = normalize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")  # grayscale
        img = np.array(img).astype(np.float32)

        # normalization
        if self.normalize == "zscore":
            img = (img - img.mean()) / (img.std() + 1e-8)
        elif self.normalize == "minmax":
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        img = np.expand_dims(img, axis=0)  # (1,H,W)

        img = torch.tensor(img, dtype=torch.float32)
        img = transforms.Resize(self.resize)(img)  # resize to (1,512,512)

        return img, 0  # dummy label


def get_dataloader(path, batch_size=16, num_workers=4, normalize="zscore", normal: bool = True, drusen: bool = False, cnv: bool = False):
    dataset = OCTNormalDrusenCNV(path, normalize=normalize, resize=(512, 512), normal=normal, drusen=drusen, cnv=cnv)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader

# if __name__ == "__main__":
#     # test dataset and dataloader
#     data_path = r"datasets/OCTDatasetNormalDrusenCNV"
#     loader = get_dataloader(data_path, batch_size=4, normalize="zscore", normal=True, drusen=True, cnv=True)
#     print (f"Dataset size: {len(loader.dataset)} images")
#     for images, _ in loader:
#         print(images.shape)  # should be (B, 1, H, W)
#         break


# =========================
# MODEL
# =========================

def build_encoder(encoder_name="resnet34", in_channels=1):
    encoder = smp.encoders.get_encoder(
        name=encoder_name,
        in_channels=in_channels,
        depth=5,
        weights=None
    )
    return encoder


class EncoderAutoencoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        enc_ch = encoder.out_channels[-1]

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(enc_ch, 256, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 2, stride=2),
        )

    def forward(self, x):
        size = x.shape[2:]  # (H,W)
        feats = self.encoder(x)
        x = feats[-1]
        x = self.decoder(x)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x


# =========================
# TRAINING
# =========================

def pretrain_encoder(
    data_path,
    encoder_name="resnet34",
    batch_size=16,
    num_workers=4,
    num_epochs=50,
    lr=1e-3,
    device="cuda",
    save_path="pretrained_encoder.pth"
):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # data
    loader = get_dataloader(data_path, batch_size, num_workers)

    # model
    encoder = build_encoder(encoder_name=encoder_name, in_channels=1)
    model = EncoderAutoencoder(encoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Start pretraining on {len(loader.dataset)} images")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)

            # =========================
            # MASKED RECONSTRUCTION (KEY!)
            # =========================
            mask = (torch.rand_like(images) > 0.75).float()
            masked_input = images * (1 - mask)

            recon = model(masked_input)


            loss = ((recon - images) ** 2 * mask).mean()

            # =========================

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {running_loss / len(loader):.4f}")

    # save encoder only
    torch.save(model.encoder.state_dict(), save_path)
    print(f"Saved encoder to {save_path}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
#    pass
    encoders = ["resnet34", "resnet50", "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3"]
    for encoder_name in encoders:
        pretrain_encoder(
            data_path=r"datasets/OCTDatasetNormalDrusenCNV",  # your new device dataset
            encoder_name=encoder_name,
            batch_size=16,
            num_workers=4,
            num_epochs=10,
            lr=1e-3,
            device="cuda",
            save_path=f"trained_models/encoder_models/pretrained_{encoder_name}_oct.pth"
        )