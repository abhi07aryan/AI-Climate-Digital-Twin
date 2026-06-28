import torch

from climate_twin.training.scheduler import build_scheduler

model = torch.nn.Linear(10, 1)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

scheduler = build_scheduler(optimizer)

losses = [
    0.50,
    0.40,
    0.35,
    0.35,
    0.35,
    0.35,
    0.35,
    0.35,
]

for epoch, loss in enumerate(losses):

    scheduler.step(loss)

    print(
        epoch + 1,
        optimizer.param_groups[0]["lr"]
    )