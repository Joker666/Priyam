import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import datasets, transforms
from torchvision.models import resnet50
from tqdm import tqdm
from utils import LocalLogger

### Parse Training Arguments ###
parser = argparse.ArgumentParser(description="Arguments for Image Classification Training")
parser.add_argument("--experiment_name", help="Name of Experiment being Launched", required=True, type=str)
parser.add_argument(
    "--path_to_data",
    help="Path to ImageNet root folder which should contain \train and \validation folders",
    required=True,
    type=str,
)
parser.add_argument(
    "--working_directory",
    help="Working Directory where checkpoints and logs are stored, inside a \
                    folder labeled by the experiment name",
    required=True,
    type=str,
)
parser.add_argument("--epochs", help="Number of Epochs to Train", default=90, type=int)
parser.add_argument(
    "--save_checkpoint_interval", help="After how many epochs to save model checkpoints", default=10, type=int
)
parser.add_argument("--num_classes", help="How many classes is our network predicting?", default=1000, type=int)
parser.add_argument(
    "--batch_size",
    help="Effective batch size. If split_batches is false, batch size is \
                         multiplied by number of GPUs utilized ",
    default=64,
    type=int,
)
parser.add_argument(
    "--gradient_accumulation_steps", help="Number of Gradient Accumulation Steps for Training", default=1, type=int
)
parser.add_argument("--learning_rate", help="Starting Learning Rate for StepLR", default=0.1, type=float)
parser.add_argument("--weight_decay", help="Weight decay for optimizer", default=1e-4, type=float)
parser.add_argument("--momentum", help="Momentum parameter for SGD optimizer", default=0.9, type=float)
parser.add_argument("--step_lr_decay", help="Decay for Step LR", default=0.1, type=float)
parser.add_argument("--lr_step_size", help="Number of epochs for every step", default=30, type=int)
parser.add_argument(
    "--lr_warmup_start_factor",
    help="Learning rate start factor (i.e if learning rate is 0.1 and start factor is 0.01, then lr warm-up from 0.1*0.01 to 0.1)",
    default=0.1,
    type=float,
)
parser.add_argument(
    "--bias_weight_decay", help="Apply weight decay to bias", default=False, action=argparse.BooleanOptionalAction
)
parser.add_argument(
    "--norm_weight_decay",
    help="Apply weight decay to normalization weight and bias",
    default=False,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument("--max_grad_norm", help="Maximum norm for gradient clipping", default=1.0, type=float)
parser.add_argument("--img_size", help="Width and Height of Images passed to model", default=224, type=int)
parser.add_argument("--num_workers", help="Number of workers for DataLoader", default=32, type=int)
parser.add_argument(
    "--resume_from_checkpoint",
    help="Checkpoint folder for model to resume training from, inside the experiment folder",
    default=None,
    type=str,
)
args = parser.parse_args()

### Init the Accelerator ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(
    log_with=None,
    project_dir=path_to_experiment,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
)

### Init Logger ###
local_logger = LocalLogger(path_to_experiment)

experiment_config = {
    "epochs": args.epochs,
    "effective_batch_size": args.batch_size * accelerator.num_processes,
    "learning_rate": args.learning_rate,
}

accelerator.init_trackers(project_name=args.experiment_name, config=experiment_config)


### Accuracy Metric ###
accuracy_fn = Accuracy(task="multiclass", num_classes=args.num_classes)

### Load Model ###
model = resnet50()
if args.num_classes != 1000:
    model.fc = nn.Linear(2048, args.num_classes)

### Set Transforms for Training and Testing ###
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=(args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

### Load Dataset ###
path_to_train_data = os.path.join(args.path_to_data, "train")
path_to_valid_data = os.path.join(args.path_to_data, "val")
trainset = datasets.ImageFolder(path_to_train_data, transform=train_transforms)
testset = datasets.ImageFolder(path_to_valid_data, transform=test_transform)

### Set up DataLoaders ###
mini_batch_size = args.batch_size // args.gradient_accumulation_steps
train_dataloader = DataLoader(
    trainset, batch_size=mini_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
)
test_dataloader = DataLoader(
    testset, batch_size=mini_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
)

### Loss Function ###
loss_fn = nn.CrossEntropyLoss()

### Optimizer ###
if (not args.bias_weight_decay) and (not args.norm_weight_decay):
    accelerator.print("No weight decay for bias and normalization layer")

    weight_decay_params = []
    no_weight_decay_params = []

    for name, param in model.named_parameters():
        if "bias" in name and not args.bias_weight_decay:
            no_weight_decay_params.append(param)
        elif "bn" in name and not args.norm_weight_decay:
            no_weight_decay_params.append(param)
        else:
            weight_decay_params.append(param)

    optimizer = torch.optim.SGD(
        [
            {"params": weight_decay_params, "weight_decay": args.weight_decay},
            {"params": no_weight_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        momentum=args.momentum,
    )
else:
    accelerator.print("Weight decay for bias and normalization layer")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

### Learning Rate Scheduler ###
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.step_lr_decay)

### Prepare Model for Training ###
model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, test_dataloader, lr_scheduler
)
accelerator.register_for_checkpointing(lr_scheduler)

### Checkpointing ###
if args.resume_from_checkpoint:
    accelerator.print(f"Resuming from checkpoint {args.resume_from_checkpoint}")
    path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
    accelerator.load_state(path_to_checkpoint)
    starting_epoch = int(path_to_checkpoint.split("_")[-1])
else:
    starting_epoch = 0

all_train_losses, all_test_losses = [], []
all_train_accs, all_test_accs = [], []

### Training Loop ###
for epoch in range(starting_epoch, args.epochs):
    accelerator.print(f"Epoch {epoch}")

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    accumulated_loss = 0
    accumulated_acc = 0

    progress_bar = tqdm(
        range(len(train_dataloader) // args.gradient_accumulation_steps), disable=not accelerator.is_local_main_process
    )

    model.train()
    for images, labels in train_dataloader:
        images, labels = images.to(accelerator.device), labels.to(accelerator.device)

        with accelerator.accumulate(model):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            accumulated_loss += loss.item() / args.gradient_accumulation_steps
            predicted = outputs.argmax(axis=1)
            accumulated_acc += accuracy_fn(predicted, labels).item() / args.gradient_accumulation_steps

            ### Compute Gradients ###
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            loss_gathered = accelerator.gather_for_metrics(accumulated_loss)
            acc_gathered = accelerator.gather_for_metrics(accumulated_acc)

            train_loss.append(torch.mean(loss_gathered).item())
            train_acc.append(torch.mean(acc_gathered).item())

            accumulated_loss = 0
            accumulated_acc = 0

            progress_bar.update(1)

    model.eval()
    for images, labels in tqdm(test_dataloader, disable=not accelerator.is_local_main_process):
        images, labels = images.to(accelerator.device), labels.to(accelerator.device)

        with torch.no_grad():
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            loss_gathered = accelerator.gather_for_metrics(loss)
            predicted = outputs.argmax(axis=1)
            acc_gathered = accelerator.gather_for_metrics(accuracy_fn(predicted, labels))

            test_loss.append(torch.mean(loss_gathered).item())
            test_acc.append(torch.mean(acc_gathered).item())

    epoch_train_loss = np.mean(train_loss)
    epoch_test_loss = np.mean(test_loss)
    epoch_train_acc = np.mean(train_acc)
    epoch_test_acc = np.mean(test_acc)

    all_train_losses.append(epoch_train_loss)
    all_test_losses.append(epoch_test_loss)
    all_train_accs.append(epoch_train_acc)
    all_test_accs.append(epoch_test_acc)

    accelerator.print("Training Accuracy: ", epoch_train_acc, "Training Loss:", epoch_train_loss)
    accelerator.print("Testing Accuracy: ", epoch_test_acc, "Testing Loss:", epoch_test_loss)

    ### Log with Local Logger ###
    if accelerator.is_main_process:
        local_logger.log(
            epoch=epoch,
            train_loss=epoch_train_loss,
            test_loss=epoch_test_loss,
            train_acc=epoch_train_acc,
            test_acc=epoch_test_acc,
        )

    accelerator.log(
        {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "test_loss": epoch_test_loss,
            "test_acc": epoch_test_acc,
            "learning_rate": lr_scheduler.get_last_lr()[0],
        }
    )

    lr_scheduler.step(epoch)

    ## Checkpointing ###
    if epoch % args.save_checkpoint_interval == 0:
        accelerator.save_state(os.path.join(path_to_experiment, f"checkpoint_{epoch}"))

accelerator.end_training()
