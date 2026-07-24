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
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# =========================
# Set up logging
# =========================

import logging
import gc
import time

def setup_logging(log_file="encoder_training.log"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='a', encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger

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


def measure_inference_time_cuda(model, input_tensor, device="cuda", runs=100):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot measure inference time on GPU.")

    model.to(device)
    input_tensor = input_tensor.to(device)
    model.eval()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    torch.cuda.synchronize()

    timings = []

    with torch.no_grad():
        for _ in range(runs):
            starter.record()

            _ = model(input_tensor)

            ender.record()
            torch.cuda.synchronize()

            timings.append(starter.elapsed_time(ender))  # ms

    return sum(timings) / len(timings)


def count_params(model):
    return sum(p.numel() for p in model.parameters())

def create_patch_mask(
        images,
        grid_size=32,
        mask_ratio=0.75):
    
    assert 0 < mask_ratio < 1, "mask_ratio must be between 0 and 1"
    
    B, C, H, W = images.shape

    assert H % grid_size == 0, f"Image height {H} is not divisible by grid_size {grid_size}"
    assert W % grid_size == 0, f"Image width {W} is not divisible by grid_size {grid_size}"

    patch_h = H // grid_size
    patch_w = W // grid_size

    # low-resolution patch grid
    patch_mask = (
        torch.rand(
            B,
            1,
            grid_size,
            grid_size,
            device=images.device
        ) < mask_ratio
    ).float()

    # expand to image resolution
    mask = patch_mask.repeat_interleave(
        patch_h,
        dim=2
    ).repeat_interleave(
        patch_w,
        dim=3
    )

    return mask


def pretrain_encoder(
    data_path,
    encoder_name="resnet34",
    batch_size=16,
    num_workers=4,
    num_epochs=50,
    lr=1e-3,
    device="cuda",
    save_path="pretrained_encoder.pth",
    parameter_count_threshold=22e6,
    log_dir="runs/encoder_pretraining",
    logger=None,
    loader: torch.utils.data.DataLoader | None = None
):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # data
    if loader is None:
        if logger is not None:
            logger.info(f"Loading dataset from {data_path} with batch_size={batch_size}, num_workers={num_workers}")
        loader = get_dataloader(data_path, batch_size, num_workers)
    else:
        if logger is not None:
            logger.info(f"Using provided dataloader with batch_size={batch_size}, num_workers={num_workers}")
    if logger is not None:
        logger.info(f"Dataset size: {len(loader.dataset)} images")

    # model
    encoder = build_encoder(encoder_name=encoder_name, in_channels=1)
    if count_params(encoder) > parameter_count_threshold:
        if logger is not None:
            logger.warning(f"Encoder '{encoder_name}' has {count_params(encoder):.2e} parameters, which exceeds the threshold of {parameter_count_threshold:.2e}. Consider using a smaller encoder to avoid OOM issues.")
        return
    model = EncoderAutoencoder(encoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    run_name = f"pretrain_{encoder_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"{log_dir}/{run_name}")

    if logger is not None:
        logger.info(f"Start pretraining of encoder '{encoder_name}' on {len(loader.dataset)} images")
        logger.info(f"Hyperparameters: batch_size={batch_size}, num_epochs={num_epochs}, lr={lr}, device={device}")
        logger.info(f"Model parameter count: {count_params(model)}")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)

            # =========================
            # MASKED RECONSTRUCTION (KEY!)
            # =========================
            # mask = (torch.rand_like(images) > 0.75).float()
            mask = create_patch_mask(images, grid_size=32, mask_ratio=0.75)
            masked_input = images * (1 - mask)

            recon = model(masked_input)


            loss = ((recon - images) ** 2 * mask).sum() / mask.sum() # MSE loss only on masked regions

            # =========================

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        writer.add_scalar("Loss/train", running_loss / len(loader), epoch)

    input_tensor = torch.randn((1, 1) + loader.dataset.resize)
    inference_time = measure_inference_time_cuda(model, input_tensor, device=device)
    writer.add_scalar("InferenceTimeMS", inference_time, 0)
    writer.add_scalar("ParameterCount", count_params(model), 0)
    # save encoder only
    torch.save(model.encoder.state_dict(), save_path)
    if logger is not None:
        logger.info(f"Saved encoder to {save_path}")
    writer.flush()
    writer.close()

    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    if logger is not None:
        logger.info(f"Finished pretraining of encoder '{encoder_name}'.")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    from segmentation_models_pytorch.encoders import get_encoder_names
    encoders = get_encoder_names()
    # encoders = [
    #     "resnet18", "resnet34", "resnet50", "resnext50_32x4d", 
    #     "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3",
    #     "densenet121", "densenet169", "densenet201",
    #     "mobilenet_v2", 
    #     "vgg16"]

    logger = setup_logging()
    logger.info(f"Starting encoder pretraining-script.")
    for encoder_name in encoders:
        try:
            pretrain_encoder(
                data_path=r"datasets/OCTDatasetNormalDrusenCNV",  # your new device dataset
                encoder_name=encoder_name,
                batch_size=16,
                num_workers=4,
                num_epochs=20,
                lr=1e-3,
                device="cuda",
                save_path=f"trained_models/encoder_models/pretrained_{encoder_name}_oct.pth",
                logger=logger
            )
        except Exception as e:
            logger.error(f"Error during pretraining of encoder '{encoder_name}': {e}")