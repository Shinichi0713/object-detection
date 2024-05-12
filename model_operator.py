import torch.nn as nn
import torch
import torchvision.models as models
from torchvision.ops import roi_pool
import os


class MultiTaskLoss(nn.Module):
    def __init__(self, lam=1):
        super(MultiTaskLoss, self).__init__()
        self.lam = lam
        self.cls = nn.CrossEntropyLoss(reduction="mean")
        self.loc = nn.SmoothL1Loss(reduction="mean")

    def forward(self, scores, preds, targets, targets_loc):
        """
        :param scores: softmax関数を通過した後の分類結果 [batch size * roi num, class num]
        :param preds: 予測されたbboxのオフセット [batch size * roi num, 4]
        :param targets: 正解クラスのラベル [batch size * roi num, class num]
        :param targets_loc: 正解bboxのオフセット [batch size * roi num, 4]
        """
        cls_loss = self.cls(scores, targets)
        loc_loss = self.loc(preds, targets_loc)
        return cls_loss + self.lam * loc_loss


class VGG16_RoI(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        device=torch.device("cuda:0"),
        pretrained_model_path=None,
    ):
        """
        :param num_classes: 类别数，不包括背景类别
        :param init_weights:
        """
        super(VGG16_RoI, self).__init__()
        # load pretrained vgg16
        model = models.vgg16(pretrained=True)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, 2)
        # load pretrained weight
        if pretrained_model_path is not None:
            model.load_state_dict(
                torch.load(pretrained_model_path, map_location=device)
            )
        if os.path.exists("/home/zaima/zero2one/models/vgg16_car_finetuned.pth"):
            model.load_state_dict(
                torch.load(
                    "/home/zaima/zero2one/models/vgg16_car_finetuned.pth",
                    map_location=device,
                )
            )
        # 512 * 28 * 28の特徴マップを取り出す
        self.features = nn.Sequential(*list(model.features.children())[:23])

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.softmax = nn.Sequential(
            nn.Linear(4096, num_classes + 1),
            nn.Softmax(dim=1),
        )

        self.bbox = nn.Sequential(
            nn.Linear(4096, num_classes * 4),
            nn.ReLU(True),
        )

    def forward(self, x, rois):
        x = self.features(x)
        rois = list(rois)
        x = roi_pool(x, rois, (7, 7), spatial_scale=x.shape[2])
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        class_score = self.softmax(x)
        bbox_regression = self.bbox(x)
        return class_score, bbox_regression