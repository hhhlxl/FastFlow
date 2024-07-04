import argparse
import os

import torch
import yaml
from ignite.contrib import metrics

import constants as const
import dataset
import fastflow
import utils
import  numpy as np


def build_train_data_loader(args, config):
    train_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )


def build_test_data_loader(args, config):
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def build_model(config):
    model = fastflow.FastFlow(
        # backbone_name=resnet18
        backbone_name=config["backbone_name"],
        # flow_step=8
        flow_steps=config["flow_step"],
        # input_size=256
        input_size=config["input_size"],
        # conv3*3_only=true
        conv3x3_only=config["conv3x3_only"],
        # hidden_ratio=1.0
        hidden_ratio=config["hidden_ratio"],
    )
    # 打印网络结构
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


# 优化器是Adam，LR = 1e-3，WEIGHT_DECAY = 1e-5
def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )


def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    # 这是一个类，用于记录和计算平均损失值
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        # forward
        data = data.cuda()
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )


def eval_once(dataloader, model):
    model.eval()
    auroc_metric_px = metrics.ROC_AUC()
    auroc_metric_im = metrics.ROC_AUC()
    for data, targets in dataloader:
        # targets是mask图片
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        outputs_flatten = outputs.flatten()
        # target就是标注的mask
        targets_flatten = targets.flatten()
        # 这是像素级的准确值吧？
        auroc_metric_px.update((outputs_flatten, targets_flatten))

        # 图像级
        outputs = outputs.numpy()
        targets = targets.cpu().numpy().astype(int)
        gt_list_sp = torch.Tensor([np.max(targets[i]) for i in range(targets.shape[0])])
        pr_list_sp = torch.Tensor([np.max(outputs[i]) for i in range(outputs.shape[0])])
        print(gt_list_sp)
        print(pr_list_sp)
        auroc_metric_im.update((pr_list_sp, gt_list_sp))
    auroc_px = auroc_metric_px.compute()
    auroc_im = auroc_metric_im.compute()
    print("Pixel-AUROC: {}".format(auroc_px))
    print("Image-AUROC: {}".format(auroc_im))


def train(args):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    # checkpoint的保存路径
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 加载config数据
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    optimizer = build_optimizer(model)

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch)
        # 每隔一个EVAL_INTERVAL测试一次
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_once(test_dataloader, model)
        # 每隔一个CHECKPOINT_INTERVAL保存一次权重
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )


def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    eval_once(test_dataloader, model)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        required=True,
        help="category name in mvtec",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
