import torch


def build_scheduler(optimizer):
    """
    Returns a ReduceLROnPlateau scheduler.

    Reduces the learning rate when the validation
    loss stops improving.
    """

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=1e-4,
        cooldown=1,
        min_lr=1e-6,
    )

    return scheduler