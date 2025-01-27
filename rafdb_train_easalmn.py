import os
import random
import sys

from torchvision import transforms, datasets
from tqdm import tqdm
import argparse
import numpy as np

import torch

from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from models.EASALMN import EASALMN
from sam import SAM
from utils.loss import BCEWithLogitsLoss, SDL, ExpressionProxyLoss
from utils.visualization import plot_confusion_matrix, plot_tsne
from utils.utils_config import get_config

import wandb

eps = sys.float_info.epsilon

# This is the original emotion labels.
class_names = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']


def run_training(args):
    set_seed()
    # get config
    cfg = get_config(args.config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    # Sign in to wandb
    try:
        wandb.login(key=cfg.wandb_key)
    except Exception as e:
        print("WandB Key must be provided in config file.")
        print(f"Config Error: {e}")
    # Initialize wandb
    try:
        wandb.init(
            project=cfg.wandb_project,
            resume=cfg.wandb_resume,
            name="Ablation_Studies_RAF-DB_EASALMN_ce+pos+neg+neu+sdl+no_init",
            notes=cfg.notes)
    except Exception as e:
        print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
        print(f"Config Error: {e}")

    model = EASALMN(cfg)
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomRotation(5),
            transforms.RandomCrop(112, padding=8)
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])

    train_dataset = datasets.ImageFolder(f'{cfg.dir}/train', transform=data_transforms)

    print('Whole train set size:', train_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               num_workers=cfg.num_workers,
                                               shuffle=True,
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    val_dataset = datasets.ImageFolder(f'{cfg.dir}/val', transform=data_transforms_val)

    print('Validation set size:', val_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.batch_size,
                                             num_workers=cfg.num_workers,
                                             shuffle=False,
                                             pin_memory=True)

    ce_loss = torch.nn.CrossEntropyLoss()

    if cfg.expression_proxy_loss_enable:
        if cfg.sdl_loss_enable:
            expression_proxy_loss = None
        else:
            expression_proxy_loss = ExpressionProxyLoss(num_classes=7, embed_dim=512, queue=None, probe=None,
                                                    negative_expression_map={0: [3, 5], 1: [5], 2: [1, 3],
                                                                             3: [4],
                                                                             4: [3], 5: [1],
                                                                             6: []},
                                                    negative_expression_weight_map={0: [1 / 2, 1 / 2], 1: [1],
                                                                                    2: [1 / 2, 1 / 2],
                                                                                    3: [1], 4: [1], 5: [1],
                                                                                    6: []},
                                                    neutral_label=6,
                                                    lambda_pos=0.001,
                                                    lambda_neg=0.001,
                                                    lambda_neu=0.001)

    if cfg.sdl_loss_enable:
        sdl = SDL(cfg.num_classes, dim=512, k=cfg.k, size=cfg.queue_size).to(device)
        sdl_loss_fn = BCEWithLogitsLoss()

    optimizer = SAM(model.parameters(), torch.optim.Adam, lr=cfg.lr, rho=0.05, adaptive=False, )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = ExponentialLRWarmup(optimizer=optimizer, warmup_epochs=5,
    #                                 total_epochs=cfg.num_epochs)

    best_acc = 0
    patience = 0
    for epoch in tqdm(range(1, cfg.num_epochs + 1)):
        if patience == 10:
            print('Early stopping. The best acc has remained for 10 epochs.')
            break

        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in train_loader:
            iter_cnt += 1

            imgs = imgs.to(device)
            targets = targets.to(device)

            def closure():
                out, feat = model(imgs)
                loss = ce_loss(out, targets)

                if cfg.expression_proxy_loss_enable and expression_proxy_loss is not None:
                    loss += expression_proxy_loss(feat, targets)

                if cfg.sdl_loss_enable and epoch > cfg.sdl_update_freq:
                    soft_targets = sdl(feat, out, targets)  # 计算 soft_targets
                    sdl_loss = sdl_loss_fn(out, soft_targets, epoch)
                    loss += sdl_loss

                loss.backward()
                return loss

            out, feat = model(imgs)
            loss = ce_loss(out, targets)

            if cfg.expression_proxy_loss_enable and expression_proxy_loss is not None:
                loss += expression_proxy_loss(feat, targets)

            if cfg.sdl_loss_enable and epoch > cfg.sdl_update_freq:
                soft_targets = sdl(feat, out, targets)  # 计算 soft_targets
                sdl_loss = sdl_loss_fn(out, soft_targets, epoch)
                loss += sdl_loss

            loss.backward()
            optimizer.step(closure)
            optimizer.zero_grad()

            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        # 更新sdl队列
        with torch.no_grad():
            if cfg.sdl_loss_enable and epoch % cfg.sdl_update_freq == 0:
                sdl.update(feat, out, targets)

        if cfg.expression_proxy_loss_enable and cfg.sdl_loss_enable and epoch == cfg.sdl_update_freq:
            queue, probe = sdl.get_queue_probe()
            expression_proxy_loss = ExpressionProxyLoss(num_classes=7, embed_dim=512, queue=queue, probe=probe,
                                                        negative_expression_map={0: [3, 5], 1: [5], 2: [1, 3],
                                                                                 3: [4],
                                                                                 4: [3], 5: [1],
                                                                                 6: []},
                                                        negative_expression_weight_map={0: [1 / 2, 1 / 2], 1: [1],
                                                                                        2: [1 / 2, 1 / 2],
                                                                                        3: [1], 4: [1], 5: [1],
                                                                                        6: []},
                                                        neutral_label=6,
                                                        lambda_pos=0.001,
                                                        lambda_neg=0.001,
                                                        lambda_neu=0.001)

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss / iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. LR %.8f. Loss: %.3f.' % (
            epoch, acc, optimizer.param_groups[0]['lr'], running_loss,))
        wandb.log({
            "epoch": epoch,
            "Training Accuracy": acc,
            'Training LR': optimizer.param_groups[0]['lr'],
            'Training Loss': running_loss
        })

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0

            ## for calculating balanced accuracy
            y_true = []
            y_pred = []
            features = []
            labels = []

            model.eval()
            for (imgs, targets) in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)

                out, feature = model(imgs)

                features.append(feature)
                labels.append(targets)

                loss = ce_loss(out, targets)
                running_loss += loss
                _, predicts = torch.max(out, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += imgs.size(0)

                y_true.append(targets.cpu().numpy())
                y_pred.append(predicts.cpu().numpy())

                if iter_cnt == 0:
                    all_predicted = predicts
                    all_targets = targets
                else:
                    all_predicted = torch.cat((all_predicted, predicts), 0)
                    all_targets = torch.cat((all_targets, targets), 0)
                iter_cnt += 1
            running_loss = running_loss / iter_cnt
            scheduler.step()
            features = torch.cat(features, dim=0).cpu().numpy()
            labels = torch.cat(labels, dim=0).cpu().numpy()

            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            best_acc = max(acc, best_acc)

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)

            tqdm.write(
                "[Epoch %d] Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f" % (epoch, acc, balanced_acc, running_loss))
            tqdm.write("best_acc:" + str(best_acc))
            wandb.log({
                "epoch": epoch,
                "best_acc": best_acc,
                "Validation Accuracy": acc,
                # "Validation Balanced Accuracy": balanced_acc,
                "Validation Loss": running_loss
            })

            if acc == best_acc:
                patience = 0
                if acc > 0.9195:
                    torch.save(model.state_dict(),
                               os.path.join("checkpoints", "Ablation_Studies", "ExpressionLoss", "ce+pos+neg+neu+sdl",
                                            "RAF-DB", "No Init",
                                            f"RAF-DB_EASALMN_CE_POS_NEG_NEU_SDL_NO_INIT_Epoch_{epoch}_Acc_{acc:.4f}.pth"))
                    tqdm.write('Model saved.')

                    # Compute confusion matrix
                    matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
                    np.set_printoptions(precision=2)
                    plt.figure(figsize=(10, 8))
                    # Plot normalized confusion matrix
                    plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                                          title=f"EASALMN On RAF-DB Confusion Matrix (acc: {acc * 100:.2f}%)",
                                          path=os.path.join("checkpoints", "Ablation_Studies", "ExpressionLoss",
                                                            "ce+pos+neg+neu+sdl", "RAF-DB", "No Init",
                                                            f"RAF-DB_EASALMN_CE_POS_NEG_NEU_SDL_NO_INIT_Epoch_{epoch}_Acc_{acc:.4f}.png"))
                    # Plot t-SNE
                    plot_tsne(method_name="EASALMN", dataset_name="RAF-DB",
                              num_classes=cfg.num_classes, expression_names=class_names, features=features,
                              labels=labels,
                              path=os.path.join("checkpoints", "Ablation_Studies", "ExpressionLoss",
                                                "ce+pos+neg+neu+sdl", "RAF-DB", "No Init",
                                                f"RAF-DB_t-SNE_CE_POS_NEG_NEU_SDL_NO_INIT_Epoch_{epoch}_Acc_{acc:.4f}.png"))
            else:
                patience += 1

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()


def set_seed(seed=None):
    if seed is not None:
        # random.seed(seed)
        # np.random.seed(seed)
        # os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python的hash随机种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(11484500242548037272)
        torch.cuda.manual_seed(5982949918632672)
        torch.cuda.manual_seed_all(5982949918632672)
        current_seed = torch.initial_seed()
        current_cuda_seed = torch.cuda.initial_seed()
        # print(f"GPU随机种子是: {current_cuda_seed}")
        print(f"当前CPU随机种子是：{current_seed}, GPU随机种子是: {current_cuda_seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_easalmn", help="py config file")
    args = parser.parse_args()
    run_training(args)
