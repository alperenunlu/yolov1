import torch
import torch.optim as optim

from yolo_model import YOLOV1_tiny, YOLOv1ResNet
from voc_data import train_loader, test_loader
from yolo_trainer import train_one_epoch, evaluate, save_checkpoint, load_checkpoint
from yolo_loss import YOLOLoss
import time
import os
import glob

import warnings

warnings.filterwarnings("ignore")


num_epochs = 20
RESUME = False

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = YOLOv1ResNet().to(device)
# model = torch.compile(model)
optimizer = optim.Adam(model.parameters(), weight_decay=0.0005)
criterion = YOLOLoss()


if __name__ == "__main__":
    if RESUME:
        latest_dir = max(glob.glob("checkpoints_*"), key=os.path.getctime)
        latest = max(glob.glob(f"{latest_dir}/*.pth"), key=os.path.getctime)
        if latest:
            model, optimizer, last_epoch, loss = load_checkpoint(
                model, optimizer, latest
            )
            print(f"Loaded checkpoint from {latest}")
            print(f"Resuming from epoch {last_epoch + 1}")
        else:
            print("No checkpoints found. Starting from scratch")

    model = torch.jit.trace(model, torch.empty(8, 3, 448, 448, device=device))

    directory = f"checkpoints_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    os.makedirs(directory, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_map = train_one_epoch(
            model, optimizer, criterion, None, train_loader, device
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            train_loss,
            os.path.join(directory, f"checkpoint_{epoch + 1}.pth"),
        )

        test_loss, test_map = evaluate(model, criterion, test_loader, device)
        print(f"Train Loss: {train_loss:.4f}, Train mAP: {train_map:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test mAP: {test_map:.4f}")
