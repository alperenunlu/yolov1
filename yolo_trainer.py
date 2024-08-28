import torch
from tqdm import tqdm


def train_one_epoch(model, optimizer, criterion, scheduler, loader, device):
    model.train()
    running_loss = 0.0
    running_idx = 0

    loop = tqdm(loader)
    for data in loop:
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if (running_idx + 1) % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        running_idx += 1

        loop.set_postfix(loss=f"{running_loss / running_idx:.4f}")

    if (running_idx + 1) % 8 != 0:
        optimizer.step()
        optimizer.zero_grad()

    return running_loss / running_idx


@torch.no_grad()
def evaluate(model, criterion, loader, device):
    model.eval()
    running_loss = 0.0
    running_idx = 0

    loop = tqdm(loader)
    for data in loop:
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        running_idx += 1

        loop.set_postfix(loss=f"{running_loss / running_idx:.4f}")

    return running_loss / running_idx


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    return model, optimizer, epoch, loss
