import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
from yolo_utils import yolo_output_to_xyxy, yolo_target_to_xyxy


def train_one_epoch(model, optimizer, criterion, scheduler, loader, device):
    model.train()
    running_loss = 0.0
    running_idx = 0
    metric = MeanAveragePrecision()

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

        # pred_boxes = []
        # target_boxes = []
        # for out, tar in zip(outputs, targets):
        #     coords, labels, confidences = yolo_output_to_xyxy(out, 0.5)
        #     target_coords, target_labels, _ = yolo_target_to_xyxy(tar)
        #     pred_boxes.append({"boxes": coords, "labels": labels, "scores": confidences})
        #     target_boxes.append({"boxes": target_coords, "labels": target_labels})
        #
        # metric_values = metric(pred_boxes, target_boxes)
        #
        postfix = {
            "loss": f"{running_loss / running_idx:.4f}",
            # "map": metric_values["map"],
            # "map50": metric_values["map_50"],
        }
        loop.set_postfix(postfix)

    if (running_idx + 1) % 8 != 0:
        optimizer.step()
        optimizer.zero_grad()

    map = metric.compute()["map"]

    return running_loss / running_idx, map


@torch.no_grad()
def evaluate(model, criterion, loader, device):
    model.eval()
    running_loss = 0.0
    running_idx = 0
    metric = MeanAveragePrecision()

    loop = tqdm(loader)
    for data in loop:
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        running_idx += 1

        pred_boxes = []
        target_boxes = []
        for out, tar in zip(outputs, targets):
            coords, labels, confidences = yolo_output_to_xyxy(out, 0.5)
            target_coords, target_labels, _ = yolo_target_to_xyxy(tar)
            pred_boxes.append({"boxes": coords, "labels": labels, "scores": confidences})
            target_boxes.append({"boxes": target_coords, "labels": target_labels})

        metric_values = metric(pred_boxes, target_boxes)
        postfix = {
            "loss": f"{running_loss / running_idx:.4f}",
            "map": metric_values["map"],
            "map50": metric_values["map_50"],
        }
        loop.set_postfix(postfix)

    map = metric.compute()["map"]

    return running_loss / running_idx, map


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
