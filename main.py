import os
import torch
import torch.optim as optim
from voc_data import VOCDataModule
from yolo_model import YOLO_V1
from yolo_loss import YOLOLoss
from yolo_utils import yolo_pred_to_dict
from torchmetrics.detection import MeanAveragePrecision
from accelerate import Accelerator
from config_parser import load_config
from tqdm.auto import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Training with Checkpointing and Logging"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="epoch",
        help="Save model every N steps or at the end of an epoch",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to resume training from a checkpoint",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs` and relevent project information",
    )
    return parser.parse_args()


def train(args):
    config = load_config("yolo_config.yaml")
    # acc_kwargs = dict(mixed_precision="bf16", dynamo_backend="inductor")
    acc_kwargs = dict()
    acc_kwargs.update(dict(project_dir=args.project_dir, log_with="all"))

    voc_data = VOCDataModule(config)
    train_loader, valid_loader = voc_data.get_dataloaders()
    model = YOLO_V1(config)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
    )
    criterion = YOLOLoss(config)
    map_metric = MeanAveragePrecision(backend="faster_coco_eval")

    accelerator = Accelerator(**acc_kwargs)
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )

    starting_epoch = 0
    overall_step = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint:
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_loader)
            resume_step -= starting_epoch * len(train_loader)

    epoch_pbar = tqdm(range(starting_epoch, config.NUM_EPOCHS), desc="Epochs")
    overall_step = 0
    map_50 = dict(
        train=None,
        valid=None,
    )
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            active_loader = accelerator.skip_first_batches(train_loader, resume_step)
            overall_step += resume_step
            train_pbar = tqdm(
                active_loader, desc="Training", leave=False, initial=resume_step
            )
        else:
            train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in train_pbar:
            images, yolo_target, labels_dict = batch
            yolo_output = model(images)
            loss = criterion(yolo_output, yolo_target)

            total_loss += loss.detach().float()

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            yolo_output, labels_dict = accelerator.gather_for_metrics(
                (yolo_output, labels_dict)
            )
            map_metric.update(yolo_pred_to_dict(yolo_output, config), labels_dict)

            train_pbar.set_postfix(loss=loss.item())

            if isinstance(args.checkpointing_steps, int):
                output_dir = f"step_{overall_step}"
                if overall_step % args.checkpointing_steps == 0:
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

        scheduler.step()
        metric_dict = map_metric.compute()
        map_50["train"] = metric_dict["map_50"]
        epoch_pbar.set_postfix(map_50)
        map_metric.reset()

        model.eval()
        valid_pbar = tqdm(valid_loader, desc="Validation", leave=False)
        for batch in valid_pbar:
            images, yolo_target, labels_dict = batch
            with torch.no_grad():
                yolo_output = model(images)

            yolo_output, labels_dict = accelerator.gather_for_metrics(
                (yolo_output, labels_dict)
            )
            map_metric.update(yolo_pred_to_dict(yolo_output, config), labels_dict)
        metric_dict = map_metric.compute()
        map_50["valid"] = metric_dict["map_50"]
        epoch_pbar.set_postfix(map_50)
        map_metric.reset()

        if args.checkpointing_steps == "epoch":
            checkpoint_path = os.path.join(args.output_dir, f"epoch_{epoch}")
            accelerator.save_state(checkpoint_path)

        accelerator.log(
            dict(
                train_map_50=map_50["train"],
                valid_map_50=map_50["valid"],
                train_loss=total_loss / len(train_loader),
                epoch=epoch,
            ),
            step=overall_step,
        )

    accelerator.end_training()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
