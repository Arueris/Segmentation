import logging

import numpy as np
import torch
import segmentation_models_pytorch as smp

def build_smp_model(
    arch: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: str | None = None,   # None | "imagenet" | "ssl" | "swsl" | ...
    in_channels: int = 1,                 # OCT: 1
    classes: int = 1,                     # binary: 1 logit channel
    activation: str | None = None,        # None -> logits (recommended)
    encoder_weights_path: str | None = None,  # Pfad zu lokal gespeicherten Encoder-Gewichten (optional)
    logger: logging.Logger | None = None
):
    """
    Factory für SMP-Modelle.
    arch: "unet", "unet++", "deeplabv3+", "fpn", "pspnet", "segformer", ...
    kwargs: durchreichen für spezifische Modelle (z.B. encoder_depth, decoder_channels, ...)
    """

    # Aliases / Normalisierung
    a = arch.strip().lower().replace(" ", "")
    alias = {
        "unet": "unet",
        "u-net": "unet",
        "unet++": "unetplusplus",
        "unetplusplus": "unetplusplus",
        "deeplabv3+": "deeplabv3plus",
        "deeplabv3plus": "deeplabv3plus",
        "deeplab": "deeplabv3plus",
        "segformer": "segformer",
        "fpn": "fpn",
        "pspnet": "pspnet",
        "linknet": "linknet",
        "pan": "pan",
        "manet": "manet",
        "upernet": "upernet",
        "dpt": "dpt",
    }
    if a not in alias:
        raise ValueError(f"Unbekannte Architektur '{arch}'. Möglich: {sorted(alias.keys())}")

    arch_name = alias[a]

    # SMP create_model: einheitliches API für viele Architekturen/Encoder [1](https://smp.readthedocs.io/en/latest/quickstart.html)[2](https://pypi.org/project/segmentation-models-pytorch/)
    model = smp.create_model(
        arch=arch_name,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation
    )

    if encoder_weights_path is not None:
        if logger is not None:
            logger.info(f"Load encoder weights from '{encoder_weights_path}'...")
        else:
            print(f"Load encoder weights from '{encoder_weights_path}'...")
        encoder_state_dict = torch.load(encoder_weights_path, map_location="cpu")
        model.encoder.load_state_dict(encoder_state_dict)
        if logger is not None:
            logger.info("Encoder weights loaded.")
        else:
            print("Encoder weights loaded.")

    return model