from dataset import create_dataset
from model.UNet import UNet
from utils.engine import GaussianDiffusionTrainer
from utils.tools import train_one_epoch, load_yaml, plot_losses, validate_one_epoch
import torch
from utils.callbacks import ModelCheckpoint


def train(config):
    consume = config["consume"]
    if consume:
        cp = torch.load(config["consume_path"])
        config = cp["config"]
    print(config)

    device = torch.device(config["device"])
    train_loader, val_loader = create_dataset(**config["Dataset"])
    start_epoch = 1

    model = UNet(**config["Model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)

    model_checkpoint = ModelCheckpoint(**config["Callback"])

    if consume:
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        scheduler.load_state_dict(cp["scheduler"])
        model_checkpoint.load_state_dict(cp["model_checkpoint"])
        start_epoch = cp["start_epoch"] + 1

    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, config["epochs"] + 1):
        train_loss = train_one_epoch(trainer, train_loader, optimizer, device, epoch)
        print(f'Epoch [{epoch}], Train Loss: {train_loss:.4f}')

        train_losses.append(train_loss)

        val_loss = validate_one_epoch(trainer, val_loader, device)
        print(f'Epoch [{epoch}], Validation Loss: {val_loss:.4f}')

        val_losses.append(val_loss)

        scheduler.step(val_loss)

        model_checkpoint.step(val_loss, model=model.state_dict(), config=config,
                              optimizer=optimizer.state_dict(), start_epoch=epoch,
                              model_checkpoint=model_checkpoint.state_dict())

    plot_losses(train_losses, val_losses, "./loss/loss_plot.png")


if __name__ == "__main__":
    config = load_yaml("config.yml", encoding="utf-8")
    train(config)
