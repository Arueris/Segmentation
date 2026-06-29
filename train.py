import logging

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
        encoder_weights_path=None,
        num_epochs: int = 10,
        num_freeze_encoder_epochs: int = 0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        path: str = r"F:/Python/SAM2/OCTDatasetOIMHS",
        train_portion=0.7,
        train_loader = None,
        test_loader = None,
        augment=True,
        max_rotate_deg=0,
        hflip_p=0.5,
        normalize="none",
        batch_size=16,
        num_workers=0,
        gpu: bool = True,
        log_dir: str = "runs",
        logger: logging.Logger | None = None,
        save_dir_model: str | None = None
):
    # setup writer for TensorBoard
    run_name = f"{arch}_{encoder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"{log_dir}/{run_name}")

    # add parameters to writer
    hparams = {
        "arch": arch,
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "encoder_weights_path": str(encoder_weights_path) if encoder_weights_path is not None else None,
        "num_epochs": num_epochs,
        "num_freeze_encoder_epochs": num_freeze_encoder_epochs,
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
    if train_loader is None and test_loader is None:
        logger.info(f"Preparing dataloaders with train portion {train_portion}, augment={augment}, max_rotate_deg={max_rotate_deg}, hflip_p={hflip_p}, normalize={normalize}, batch_size={batch_size}, num_workers={num_workers}")
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
    else:
        logger.info(f"Using provided dataloaders. Length of train dataset: {len(train_loader.dataset)}, length of test dataset: {len(test_loader.dataset)}")
    logger.info(f"Length of train dataset: {len(train_loader.dataset)}, length of test dataset: {len(test_loader.dataset)}")

    #load model
    network = model.build_smp_model(
        arch=arch, 
        encoder_name=encoder_name, 
        encoder_weights=encoder_weights,
        in_channels=1,
        classes=1,
        encoder_weights_path=encoder_weights_path,
        activation=None)

    if encoder_weights_path is not None and num_freeze_encoder_epochs > 0:
        if logger is not None:
            logger.info(f"Freezing encoder for the first {num_freeze_encoder_epochs} epochs.")
        for param in network.encoder.parameters():
            param.requires_grad = False

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
        if epoch == num_freeze_encoder_epochs and encoder_weights_path is not None:
            if logger is not None:
                logger.info(f"Unfreezing encoder after {num_freeze_encoder_epochs} epochs.")
            for param in network.encoder.parameters():
                param.requires_grad = True
        train_loss = train_one_epoch(network, train_loader, loss_fn, optimizer, device, desc=f"[{arch}_{encoder_name}] Train Epoch {epoch+1}/{num_epochs}", writer=writer, epoch=epoch)
        val_loss, val_dice, val_iou = val_one_epoch(network, test_loader, loss_fn, device, desc=f"[{arch}_{encoder_name}] Validation Epoch {epoch+1}/{num_epochs}", writer=writer, epoch=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_dice = val_dice
            best_val_iou = val_iou
            best_val_train_loss = train_loss
            if save_dir_model is not None:
                # create save directory if it doesn't exist
                os.makedirs(save_dir_model, exist_ok=True)
                torch.save(network.state_dict(), f"{save_dir_model}/{run_name}_best_model.pth")
            else:
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
    if writer is not None:
        writer.add_scalar("Loss/train", avg_loss, epoch)
    return avg_loss

def val_one_epoch(network, dataloader, loss_fn, device, desc, writer, epoch, save_images=False):
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

    if writer is not None:
        writer.add_scalar("Loss/val", avg_loss, epoch)
        writer.add_scalar("Metrics/val_dice", avg_dice, epoch)
        writer.add_scalar("Metrics/val_iou", avg_iou, epoch)

    k = min(4, images.size(0))
    img_k = images[:k].detach().cpu()
    gt_k = masks[:k].detach().cpu()
    pred_k = preds[:k].detach().cpu()

    ov_gt = make_overlay(img_k, gt_k, color=(0,1,0))        # green for GT
    ov_pred = make_overlay(img_k, pred_k, color=(1,0,0))    # red for prediction

    if save_images:
        grid_img = torchvision.utils.make_grid(to_3ch(img_k), nrow=k)
        grid_gt = torchvision.utils.make_grid(to_3ch(gt_k), nrow=k)
        grid_pred = torchvision.utils.make_grid(to_3ch(pred_k), nrow=k)
        grid_ov_gt = torchvision.utils.make_grid(ov_gt, nrow=k)
        grid_ov_pred = torchvision.utils.make_grid(ov_pred, nrow=k)

        if writer is not None:
            writer.add_image("Images/val_input", grid_img, epoch)
            writer.add_image("Masks/val_gt", grid_gt, epoch)
            writer.add_image("Masks/val_pred", grid_pred, epoch)
            writer.add_image("Overlays/val_gt", grid_ov_gt, epoch)
            writer.add_image("Overlays/val_pred", grid_ov_pred, epoch)

    return avg_loss, avg_dice, avg_iou




def execute_train_segmentation_models_from_scratch():
    import logging
    import gc
    import time
    log_file = "logs/training.log"
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

def execute_train_segmentation_models_scratch_vs_pretrained():
    import logging
    import gc
    import time
    from segmentation_models_pytorch.encoders import get_encoder_names

    def check_completed_experiments(log_file="training_scratch_vs_pretrained.log"):
        completed = set()
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                pretrained = False
                for line in f:
                    if "Finished training with architecture" in line:
                        parts = line.split("Finished training with architecture '")[1].split("' and encoder '")
                        arch = parts[0]
                        encoder = parts[1].split("' and normalization '")[0]
                        normalize = parts[1].split("' and normalization '")[1].split("' in ")[0]
                        pretrained_path = Path(f"trained_models/encoder_models/pretrained_{encoder}_oct.pth") if pretrained else None
                        completed.add((arch, encoder, normalize, pretrained_path))
                    else:
                        if "Start training" in line:
                            pretrained = "pretrained encoder" in line
        return completed

    log_file = f"logs/training_scratch_vs_pretrained.log"
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if os.path.exists(log_file):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.info(f"Log file '{log_file}' already exists. Checking completed experiments to avoid duplicates.")
        completed_experiments = check_completed_experiments(log_file)
        logger.info(f"Found {len(completed_experiments)} completed experiments in log file.")
    else:
        completed_experiments = list()
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
    archs = ["unet",  "deeplabv3+", "fpn", "pspnet", "segformer"] # "unet++", takes to long with "wrong" encoder.
    encoders = get_encoder_names()
    normalize_options = ["zscore"] # "none", "minmax", 

    # prepare hyperparameter combinations and check completed experiments
    combinations = list()
    logger.info(f"Preparing hyperparameter combinations.")
    skipped_encoder = ["vgg19", "xception", "densenet201", "densenet169", "dpn68b", "resnet50"]
    for arch in archs:
        for encoder in encoders:
            if "timm" in encoder:
                logger.info(f"Skipping encoder '{encoder}' from timm library.")
                continue
            if encoder in skipped_encoder:
                logger.info(f"Skipping encoder '{encoder}'.")
                continue
            
            encoder_pretrained_path = Path(f"trained_models/encoder_models/pretrained_{encoder}_oct.pth")
            if not encoder_pretrained_path.exists():
                logger.warning(f"Pretrained weights for encoder '{encoder}' not found at '{encoder_pretrained_path}'. Skipping pretrained experiment for this encoder.")
                continue
            for normalize in normalize_options:
                scratch_combo = (arch, encoder, normalize, None)
                if scratch_combo in completed_experiments:
                    logger.info(f"Skipping already completed scratch experiment with architecture '{arch}' and encoder '{encoder}' and normalization '{normalize}'")
                else:
                    combinations.append(scratch_combo)
                pretrained_combo = (arch, encoder, normalize, encoder_pretrained_path)
                if pretrained_combo in completed_experiments:
                    logger.info(f"Skipping already completed pretrained experiment with architecture '{arch}' and encoder '{encoder}' and normalization '{normalize}'")
                else:
                    combinations.append((arch, encoder, normalize, f"trained_models/encoder_models/pretrained_{encoder}_oct.pth"))  # pretrained

    n_models = len(combinations)
    current_model_n = 1
    logger.info(f"Starting training with {n_models} total experiments.")
    
    for arch, encoder, normalize, encoder_path in combinations:
        if encoder_path is None:
            logger.info(f"Start training from scratch with architecture '{arch}' and encoder '{encoder}' and normalization '{normalize}'")
        else:
            logger.info(f"Start training with pretrained encoder from '{encoder_path}' with architecture '{arch}' and encoder '{encoder}' and normalization '{normalize}'")
        try:
            start = time.time()
            train_model(
                arch=arch, 
                encoder_name=encoder, 
                encoder_weights=None, 
                encoder_weights_path=str(encoder_path) if encoder_path is not None else None,
                num_epochs=20, 
                num_freeze_encoder_epochs=10 if encoder_path is not None else 0,
                lr=1e-3, 
                weight_decay=1e-4, 
                path=r"datasets/OCTDatasetOIMHS", 
                train_portion=0.05, 
                augment=True, 
                max_rotate_deg=0, 
                hflip_p=0.5, 
                normalize=normalize, 
                batch_size=16, 
                num_workers=0, 
                gpu=True,
                log_dir="runs/scratchVSpretrained",
                save_dir_model="trained_models/scratchVSpretrained",
                logger=logger)
            logger.info(f"Finished training with architecture '{arch}' and encoder '{encoder}' and normalization '{normalize}' in {(time.time() - start)/60:.2f} minutes")
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error at {arch} and {encoder} and Normalisierung {normalize}: {e}")
        logger.info(f"Completed experiment {current_model_n}/{n_models}")
        current_model_n += 1


def execute_train_unsupervised_pretrain_vs_from_scratch():
    import logging
    import gc
    import time
    import pretrain_encoder
    from segmentation_models_pytorch.encoders import get_encoder_names

    def check_completed_experiments(log_file="training_scratch_vs_pretrained_OIMHSDataset.log"):
        completed = set()
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                pretrained = False
                for line in f:
                    if "Finished training with architecture" in line:
                        parts = line.split("Finished training with architecture '")[1].split("' and encoder '")
                        arch = parts[0]
                        encoder = parts[1].split("' and normalization '")[0]
                        normalize = parts[1].split("' and normalization '")[1].split("' in ")[0]
                        pretrained_path = Path(f"trained_models/encoder_models/pretrained_{encoder}_oct_OIMHSDataset.pth") if pretrained else None
                        completed.add((arch, encoder, normalize, pretrained_path))
                    else:
                        if "Start training" in line:
                            pretrained = "pretrained encoder" in line
        return completed

    log_file = f"logs/training_scratch_vs_pretrained_OIMHSDataset.log"
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if os.path.exists(log_file):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.info(f"Log file '{log_file}' already exists. Checking completed experiments to avoid duplicates.")
        completed_experiments = check_completed_experiments(log_file)
        logger.info(f"Found {len(completed_experiments)} completed experiments in log file.")
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
    archs = ["unet", "unet++", "deeplabv3+", "fpn", "pspnet", "segformer"] # "unet++", takes to long with "wrong" encoder.
    encoders = get_encoder_names()
    normalize_options = ["zscore"] # "none", "minmax", 
    unetpp_encoders = ["resnet34", "resnet18", "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3"]
    combinations = list()
    logger.info(f"Preparing hyperparameter combinations.")
    skipped_encoder = ["vgg19", "xception", "densenet201", "densenet169", "dpn68b", "resnet50"]
    encoder_pretrained_pathes = list()
    for arch in archs:
        for encoder in encoders:
            if "timm" in encoder:
                logger.info(f"Skipping encoder '{encoder}' from timm library.")
                continue
            if encoder in skipped_encoder:
                logger.info(f"Skipping encoder '{encoder}'.")
                continue
            if "resnext" in encoder:
                logger.info(f"Skipping encoder '{encoder}' because it is a resnext model which is likely to have too many parameters.")
                continue
            
            encoder_pretrained_path = Path(f"trained_models/encoder_models/pretrained_{encoder}_oct_OIMHSDataset.pth")

            if encoder_pretrained_path not in encoder_pretrained_pathes:
                encoder_pretrained_pathes.append(encoder_pretrained_path)

            for normalize in normalize_options:
                if arch == "unet++" and encoder not in unetpp_encoders:
                    logger.info(f"Skipping incompatible combination of architecture '{arch}' and encoder '{encoder}'")
                    continue
                # scratch_combo = (arch, encoder, normalize, None)
                # if scratch_combo in completed_experiments:
                #     logger.info(f"Skipping already completed scratch experiment with architecture '{arch}' and encoder '{encoder}' and normalization '{normalize}'")
                # else:
                # combinations.append(scratch_combo)
                pretrain_path_combo = (arch, encoder, normalize, encoder_pretrained_path)
                # if pretrained_combo in completed_experiments:
                #     logger.info(f"Skipping already completed pretrained experiment with architecture '{arch}' and encoder '{encoder}' and normalization '{normalize}'")
                # else:
                combinations.append(pretrain_path_combo)  # pretrained

    n_models = len(combinations) * 2  # each combination has a pretrained and a from-scratch experiment
    current_model_n = 1

    # prepare dataloaders
    logger.info(f"Initialize dataloaders.")
    datapath = r"datasets/OCTDatasetOIMHS"
    path = Path(datapath)
    participants = os.listdir(path / "Images")
    pretrain_loader, train_loader, test_loader = dataset.get_dataloader_encoder_pretraining(
        participants=participants,
        portions=(0.65, 0.05, 0.3),
        path=datapath,
        augment=True,
        max_rotate_deg=0,
        hflip_p=0.5,
        return_numpy=False,
    )
    logger.info(f"Length of pretrain dataset: {len(pretrain_loader.dataset)}, length of train dataset: {len(train_loader.dataset)}, length of test dataset: {len(test_loader.dataset)}")


    logger.info(f"Start pretraining {len(encoder_pretrained_pathes)} encoders for the pretrained experiments.")
    for encoder_pretrained_path in encoder_pretrained_pathes:
        encoder_name = encoder_pretrained_path.stem.split("pretrained_")[1].split("_oct")[0]
        if encoder_pretrained_path.exists():
            logger.info(f"Pretrained weights for encoder '{encoder_name}' already exist at '{encoder_pretrained_path}'. Skipping pretraining for this encoder.")
            continue
        logger.info(f"Pretraining encoder '{encoder_name}' with path '{encoder_pretrained_path}'")
        try:
            start = time.time()
            pretrain_encoder.pretrain_encoder(
                data_path = "",
                encoder_name=encoder_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                save_path=encoder_pretrained_path,
                log_dir = f"runs/encoder_pretraining_OIMHSDataset,",
                logger = logger,
                loader = pretrain_loader
            )
            logger.info(f"Finished pretraining encoder '{encoder_name}' in {(time.time() - start)/60:.2f} minutes")
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during pretraining of encoder '{encoder_name}': {e}")

    ###### new combinations, because of errors in the train script
    combinations = [
        ("deeplabv3+", "mit_b0", "zscore", "placeholder"),
        ("deeplabv3+", "mit_b1", "zscore", "placeholder"),
        ("deeplabv3+", "mobilenet_v2", "zscore", "placeholder"),
        ("deeplabv3+", "mobileone_s0", "zscore", "placeholder"),
        ("unet", "mit_b0", "zscore", "placeholder"),
        ("unet", "mit_b1", "zscore", "placeholder"),
        ("unet", "mobilenet_v2", "zscore", "placeholder"),
        ("unet", "mobileone_s0", "zscore", "placeholder"),
        ("unet", "mobileone_s1", "zscore", "placeholder"),
        ("unet", "mobileone_s2", "zscore", "placeholder"),
        ("unet", "mobileone_s3", "zscore", "placeholder"),
        ("unet", "mobileone_s4", "zscore", "placeholder")
    ]
    n_models = len(combinations)
    
    logger.info(f"Starting training with {n_models} total experiments.")

    for arch, encoder, normalize, encoder_path in combinations:
        # if not encoder_path.exists():
        #     logger.warning(f"Encoder weights for '{encoder_path}' do not exist. Skipping this experiment.")
        #     current_model_n += 2 # because we skip both the pretrained and the from-scratch experiment for this combination
        #     continue
        for pretrained_encoder in [None]: # encoder_path
            # if (arch, encoder, normalize, pretrained_encoder) in completed_experiments:
                # if pretrained_encoder is None:
                #     logger.info(f"Skipping already completed scratch experiment with architecture '{arch}' and encoder '{encoder}' and normalization '{normalize}'")
                # else:
                #     logger.info(f"Skipping already completed pretrained experiment with architecture '{arch}' and encoder '{encoder}' and normalization '{normalize}'")
                # continue
            if pretrained_encoder is None:
                logger.info(f"Start training from scratch with architecture '{arch}' and encoder '{encoder}' and normalization '{normalize}'")
            else:
                logger.info(f"Start training with pretrained encoder from '{pretrained_encoder}' with architecture '{arch}' and encoder '{encoder}' and normalization '{normalize}'")
            try:
                start = time.time()
                train_model(
                    arch=arch, 
                    encoder_name=encoder, 
                    encoder_weights=None, 
                    encoder_weights_path= pretrained_encoder, #str(encoder_path) if encoder_path is not None else None,
                    num_epochs=20, 
                    num_freeze_encoder_epochs=10 if pretrained_encoder is not None else 0,
                    lr=1e-3, 
                    weight_decay=1e-4, 
                    # path=r"datasets/OCTDatasetOIMHS", 
                    # train_portion=0.05,  # not needed since we already have separate train and test loaders
                    train_loader=train_loader,
                    test_loader=test_loader, 
                    augment=True, 
                    max_rotate_deg=0, 
                    hflip_p=0.5, 
                    normalize=normalize, 
                    batch_size=16, 
                    num_workers=0, 
                    gpu=True,
                    log_dir="runs/scratchVSpretrainedOIMHS",
                    save_dir_model="trained_models/scratchVSpretrainedOIMHS",
                    logger=logger)
                logger.info(f"Finished training with architecture '{arch}' and encoder '{encoder}' and normalization '{normalize}' in {(time.time() - start)/60:.2f} minutes")
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error at {arch} and {encoder} and Normalisierung {normalize}: {e}")
            logger.info(f"Completed experiment {current_model_n}/{n_models}")
            current_model_n += 1

if __name__ == "__main__":
    # execute_train_segmentation_models_scratch_vs_pretrained()
    execute_train_unsupervised_pretrain_vs_from_scratch()