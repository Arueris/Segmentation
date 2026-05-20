import dataset
import model
import os
from pathlib import Path
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
import torchvision

def to_3ch(x):
    # x: (B,1,H,W) -> (B,3,H,W)
    return x.repeat(1, 3, 1, 1)

def make_overlay(gray, mask, color=(0, 1, 0), alpha=0.35):
    g = gray.clone()
    g = (g - g.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0])
    denom = (g.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] - 1e-8)
    g = g/denom

    base = to_3ch(g)  # (B,3,H,W)
    overlay = base.clone()

    r, g, b = color
    overlay[:, 0] = torch.where(mask[:, 0] > 0.5, (1-alpha)*base[:, 0] + alpha * r, base[:, 0])
    overlay[:, 1] = torch.where(mask[:, 0] > 0.5, (1-alpha)*base[:, 1] + alpha * g, base[:, 1])
    overlay[:, 2] = torch.where(mask[:, 0] > 0.5, (1-alpha)*base[:, 2] + alpha * b, base[:, 2])

    return overlay

def train_model(
        arch="unet",
        encoder_name="resnet34",
        encoder_weights=None,
        num_epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        path: str = r"F:/Python/SAM2/OCTDatasetOIMHS",
        train_portion=0.7,
        augment=True,
        max_rotate_deg=0,
        hflip_p=0.5,
        normalize="none",
        batch_size=16,
        num_workers=0,
        gpu: bool = True
):
    # setup writer for TensorBoard
    run_name = f"{arch}_{encoder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    # add parameters to writer
    hparams = {
        "arch": arch,
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "train_portion": train_portion,
        "augment": augment,
        "max_rotate_deg": max_rotate_deg,
        "hflip_p": hflip_p,
        "normalize": normalize,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "gpu": gpu
    }


    # prepare dataloaders
    path = Path(path)
    participants = os.listdir(path / "Images")
    train_loader, test_loader = dataset.get_dataloader(
        participants, 
        train_portion=train_portion,
        path=path,
        augment=augment,
        max_rotate_deg=max_rotate_deg,
        hflip_p=hflip_p,
        return_numpy=False,
        normalize=normalize,
        batch_size=batch_size,
        num_workers=num_workers
    )

    #load model
    network = model.build_smp_model(
        arch=arch, 
        encoder_name=encoder_name, 
        encoder_weights=encoder_weights,
        in_channels=1,
        classes=1,
        activation=None)

    # define loss function and optimizer
    bce = torch.nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def loss_fn(logits, target):
        return 0.5 * bce(logits, target) + 0.5 * dice(logits, target)
    
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    network.to(device)

    # training loop
    best_val_loss = float('inf')
    best_val_dice = 0.0
    best_val_iou = 0.0
    best_val_train_loss = 0.0
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(network, train_loader, loss_fn, optimizer, device, desc=f"[{arch}_{encoder_name}] Train Epoch {epoch+1}/{num_epochs}", writer=writer, epoch=epoch)
        val_loss, val_dice, val_iou = val_one_epoch(network, test_loader, loss_fn, device, desc=f"[{arch}_{encoder_name}] Validation Epoch {epoch+1}/{num_epochs}", writer=writer, epoch=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_dice = val_dice
            best_val_iou = val_iou
            best_val_train_loss = train_loss
            torch.save(network.state_dict(), f"trained_models/{run_name}_best_model.pth")

    metrics = {
        "hparam/val_loss": float(best_val_loss),
        "hparam/val_dice": float(best_val_dice),
        "hparam/val_iou": float(best_val_iou),
        "hparam/train_loss_at_best_val": float(best_val_train_loss)
    }
    writer.add_hparams(hparams, metrics)
    writer.flush()
    writer.close()

    network.to("cpu")

    del network
    del optimizer
    torch.cuda.empty_cache()

def train_one_epoch(network, dataloader, loss_fn, optimizer, device, desc, writer, epoch):
    network.train()
    epoch_loss = 0.0
    n_batches = 0
    for images, masks in tqdm(dataloader, desc=desc):
        images = images.to(device)  # (B,1,H,W)
        masks = masks.to(device)    # (B,1,H,W)

        optimizer.zero_grad()
        logits = network(images)    # (B,1,H,W)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)  # accumulate total loss over all samples
        n_batches += images.size(0)
    avg_loss = epoch_loss / n_batches
    writer.add_scalar("Loss/train", avg_loss, epoch)
    return avg_loss

def val_one_epoch(network, dataloader, loss_fn, device, desc, writer, epoch):
    network.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    n_batches = 0
    threshold=0.5

    eps = 1e-7

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=desc):
            images = images.to(device)
            masks = masks.to(device)

            logits = network(images)
            loss = loss_fn(logits, masks)
            

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            preds_f = preds.view(preds.size(0), -1)
            masks_f = masks.view(masks.size(0), -1)

            intersection = (preds_f * masks_f).sum(dim=1)
            pred_sum = preds_f.sum(dim=1)
            mask_sum = masks_f.sum(dim=1)

            dice_score = (2.0 * intersection + eps) / (pred_sum + mask_sum + eps)

            union = pred_sum + mask_sum - intersection
            iou = (intersection + eps) / (union + eps)

            running_loss += loss.item() * images.size(0)
            running_dice += dice_score.sum().item()
            running_iou += iou.sum().item()
            n_batches += images.size(0)
    
    avg_loss = running_loss / n_batches
    avg_dice = running_dice / n_batches
    avg_iou = running_iou / n_batches

    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Metrics/val_dice", avg_dice, epoch)
    writer.add_scalar("Metrics/val_iou", avg_iou, epoch)

    k = min(4, images.size(0))
    img_k = images[:k].detach().cpu()
    gt_k = masks[:k].detach().cpu()
    pred_k = preds[:k].detach().cpu()

    ov_gt = make_overlay(img_k, gt_k, color=(0,1,0))        # green for GT
    ov_pred = make_overlay(img_k, pred_k, color=(1,0,0))    # red for prediction

    grid_img = torchvision.utils.make_grid(to_3ch(img_k), nrow=k)
    grid_gt = torchvision.utils.make_grid(to_3ch(gt_k), nrow=k)
    grid_pred = torchvision.utils.make_grid(to_3ch(pred_k), nrow=k)
    grid_ov_gt = torchvision.utils.make_grid(ov_gt, nrow=k)
    grid_ov_pred = torchvision.utils.make_grid(ov_pred, nrow=k)

    writer.add_image("Images/val_input", grid_img, epoch)
    writer.add_image("Masks/val_gt", grid_gt, epoch)
    writer.add_image("Masks/val_pred", grid_pred, epoch)
    writer.add_image("Overlays/val_gt", grid_ov_gt, epoch)
    writer.add_image("Overlays/val_pred", grid_ov_pred, epoch)

    return avg_loss, avg_dice, avg_iou


def check_completed_experiments(log_file="training.log"):
    completed = set()
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if "Finished training with architecture" in line:
                    parts = line.split("Finished training with architecture '")[1].split("' and encoder '")
                    arch = parts[0]
                    encoder = parts[1].split("' and normalization '")[0]
                    normalize = parts[1].split("' and normalization '")[1].split("' in ")[0]
                    completed.add((arch, encoder, normalize))
    return completed

if __name__ == "__main__":
    import logging
    import gc
    import time
    log_file = "training.log"
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
    completed_experiments = check_completed_experiments(log_file)
    archs = ["unet", "unet++", "deeplabv3+", "fpn", "pspnet", "segformer"] # 
    encoders = ["resnet34", "resnet50", "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3"]
    normalize_options = ["none", "minmax", "zscore"]
    logger.info(f"Starting training with {len(archs)*len(encoders)*len(normalize_options)} total experiments (after filtering completed ones)")
    for arch in archs:
        for encoder in encoders:
            for normalize in normalize_options:
                if (arch, encoder, normalize) in completed_experiments:
                    logger.info(f"Skipping already completed experiment with architecture '{arch}' and encoder '{encoder}' and normalization '{normalize}'")
                    continue
                if arch == "unet++" and encoder == "resnet50":
                    logger.info(f"Skipping incompatible combination of architecture '{arch}' and encoder '{encoder}'")
                    continue
                logger.info(f"Start training with architecture '{arch}' and encoder '{encoder}' and normalization '{normalize}'")
                try:
                    start = time.time()
                    train_model(
                        arch=arch, 
                        encoder_name=encoder, 
                        encoder_weights=None, 
                        num_epochs=20, 
                        lr=1e-3, 
                        weight_decay=1e-4, 
                        path=r"datasets/OCTDatasetOIMHS", 
                        train_portion=0.7, 
                        augment=True, 
                        max_rotate_deg=0, 
                        hflip_p=0.5, 
                        normalize=normalize, 
                        batch_size=16, 
                        num_workers=4, 
                        gpu=True)
                    logger.info(f"Finished training with architecture '{arch}' and encoder '{encoder}' and normalization '{normalize}' in {(time.time() - start)/60:.2f} minutes")
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error at {arch} and {encoder} and Normalisierung {normalize}: {e}")