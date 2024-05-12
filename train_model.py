import os
import copy
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 今回は他のファイルに記述したネットワークとデータセットと損失関数などをロードしてきています
# 必要に応じてこれらのクラスや関数を配置してください
from model_operator import VGG16_RoI, MultiTaskLoss
from create_dataset import CustomFinetuneDataset
from arrange_data import check_dir


def load_data(data_root_dir, batch_size=128):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((227, 227)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    data_loaders = {}
    data_sizes = {}
    for name in ["train", "val"]:
        data_dir = os.path.join(data_root_dir, name)
        data_set = CustomFinetuneDataset(data_dir, transform=transform)
        data_loader = DataLoader(
            dataset=data_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )

        data_loaders[name] = data_loader
        data_sizes[name] = len(data_set)

    return data_loaders, data_sizes


def train_model(
    data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None
):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for input_img, target_cls, rois, target_offsets in data_loaders[phase]:
                positive_num_per_batch = (
                    torch.sum(target_cls.reshape(-1, 2)[:, 0] == 1) / batch_size
                ).item()
                positive_num_per_batch = int(positive_num_per_batch)
                target_offsets = target_offsets[:, :positive_num_per_batch, :]
                input_img = input_img.to(device)
                target_cls = target_cls.to(device)
                rois = rois.to(device).float()
                target_offsets = target_offsets.to(device).float()

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs_cls, outputs_offsets = model(input_img, rois)

                    loss = criterion(
                        outputs_cls,
                        outputs_offsets.reshape(batch_size, -1, 4)[
                            :, :positive_num_per_batch, :
                        ],
                        target_cls.reshape(-1, 2),
                        target_offsets,
                    )
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * target_cls.size(0)
                outputs_cls = torch.argmax(outputs_cls, dim=1)
                running_corrects += torch.sum(
                    outputs_cls == torch.argmax(target_cls.reshape(-1, 2), dim=1)
                )
            if phase == "train":
                lr_scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects / data_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # バッチサイズ（画像枚数）
    batch_size = 2
    # エポック数
    num_epochs = 30
    # バウンディングボックスの損失の重み
    lam = 1

    data_loaders, data_sizes = load_data("./data/finetune_car", batch_size)

    model = VGG16_RoI(num_classes=1)

    model = model.to(device)
    print(model)

    criterion = MultiTaskLoss(lam=lam)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model = train_model(
        data_loaders,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        device=device,
        num_epochs=num_epochs,
    )

    check_dir("./models")
    torch.save(best_model.state_dict(), "./models/fast_rcnn.pth")
