import torch.optim as optim
import torch.nn as nn
import torch
import models
import tqdm
import argparse
import os
from torch.autograd import Variable
from dataset import ImageNetDataset
from torch.utils.data import DataLoader


def train(dataset_dir, num_classes, epoch=100, batch_size=24, lr=1e-3,
          dev="cuda" if torch.cuda.is_available() else "gpu",
          ckpt_path=""):
    dev = torch.device(dev)

    model = models.create_stpnet(num_classes).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    e = 0
    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        e = ckpt["epoch"]

    trainset = ImageNetDataset(dataset_dir)
    dataloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    ce_loss = nn.CrossEntropyLoss()

    for e in range(e, epoch):
        bar = tqdm.tqdm(enumerate(dataloader), dynamic_ncols=True, ascii=True, desc=f"[{e}/{epoch}]")
        for b, (imgs, labels) in bar:
            imgs = Variable(imgs.to(dev), requires_grad=False)
            labels = Variable(labels.to(dev), requires_grad=False)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = ce_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            precision = (torch.argmax(outputs, dim=1) == labels).mean()
            bar.set_postfix_str("loss: %.3f, precision: %.2f%%" % (loss.item(), precision.item() * 100))

        os.mkdir("ckpts")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": e,
        }, f"ckpts/StpNet{num_classes}_ckpt.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=24, help="size of each image batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--dev", type=str, default="cuda", help="device for training")
    parser.add_argument("--num_classes", type=int, default=1000, help="size of each image batch")
    parser.add_argument("--dataset", type=str, help="path to data config file")
    parser.add_argument("--ckpt_path", type=str, help="if specified starts from checkpoint model")
    opt = parser.parse_args()
    print(opt)

    train(opt.dataset, opt.num_classes, opt.epoch, opt.batch_size, opt.lr, opt.dev, opt.ckpt_path)
