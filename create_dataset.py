import random
import cv2
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def iou(pred_box, target_box):
    """
    候補となる提案とラベル付きバウンディングボックスのIoUを計算する
    :param pred_box: size [4].
    :param target_box: size [N, 4] :return: [N].
    :return: [N］
    """
    if len(target_box.shape) == 1:
        target_box = target_box[np.newaxis, :]

    xA = np.maximum(pred_box[0], target_box[:, 0])
    yA = np.maximum(pred_box[1], target_box[:, 1])
    xB = np.minimum(pred_box[2], target_box[:, 2])
    yB = np.minimum(pred_box[3], target_box[:, 3])

    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)

    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (target_box[:, 2] - target_box[:, 0]) * (
        target_box[:, 3] - target_box[:, 1]
    )

    scores = intersection / (boxAArea + boxBArea - intersection)
    return scores


def parse_car_csv(csv_dir):
    csv_path = os.path.join(csv_dir, "car.csv")
    samples = np.loadtxt(csv_path, dtype="unicode")
    return samples


class CustomFinetuneDataset(Dataset):
    def __init__(self, root_dir, transform):
        """
        positive_anotations: [x_min, y_min, x_max, y_max, t_x, t_y, t_w, t_h]
        最初の4つは正規化された提案領域の座標。次の正解とのオフセット
        negative_annotations: [x_min, y_min, x_max, y_max]
        """
        self.transform = transform

        samples = parse_car_csv(root_dir)

        jpeg_images = list()
        annotation_dict = dict()

        for idx in range(len(samples)):
            sample_name = samples[idx]
            img = cv2.imread(os.path.join(root_dir, "JPEGImages", sample_name + ".jpg"))
            h, w = img.shape[:2]
            jpeg_images.append(img)

            bndbox_path = os.path.join(root_dir, "bndboxs", sample_name + ".csv")
            bndboxes = np.loadtxt(bndbox_path, dtype="float32", delimiter=" ")

            positive_annotation_path = os.path.join(
                root_dir, "Annotations", sample_name + "_1.csv"
            )
            positive_annotations = np.loadtxt(
                positive_annotation_path, dtype="float32", delimiter=" "
            )

            offsets = list()
            if len(positive_annotations.shape) == 1:
                positive_annotations = positive_annotations[np.newaxis, :]
                gt_bbox = self.get_bndbox(bndboxes, positive_annotations[0])
                # オフセットを計算する
                x_min, y_min, x_max, y_max = positive_annotations[0][:4]
                p_w = x_max - x_min
                p_h = y_max - y_min
                p_x = x_min + p_w / 2
                p_y = y_min + p_h / 2

                x_min, y_min, x_max, y_max = gt_bbox
                g_w = x_max - x_min
                g_h = y_max - y_min
                g_x = x_min + g_w / 2
                g_y = y_min + g_h / 2

                t_x = (g_x - p_x) / p_w
                t_y = (g_y - p_y) / p_h
                t_w = np.log(g_w / p_w)
                t_h = np.log(g_h / p_h)

                positive_annotations[0][0] /= w
                positive_annotations[0][1] /= h
                positive_annotations[0][2] /= w
                positive_annotations[0][3] /= h

                offsets.append(np.array([t_x, t_y, t_w, t_h]))

            else:
                for i in range(len(positive_annotations)):
                    gt_bbox = self.get_bndbox(bndboxes, positive_annotations[i])
                    # オフセットを計算する
                    x_min, y_min, x_max, y_max = positive_annotations[i][:4]
                    p_w = x_max - x_min
                    p_h = y_max - y_min
                    p_x = x_min + p_w / 2
                    p_y = y_min + p_h / 2

                    x_min, y_min, x_max, y_max = gt_bbox
                    g_w = x_max - x_min
                    g_h = y_max - y_min
                    g_x = x_min + g_w / 2
                    g_y = y_min + g_h / 2

                    t_x = (g_x - p_x) / p_w
                    t_y = (g_y - p_y) / p_h
                    t_w = np.log(g_w / p_w)
                    t_h = np.log(g_h / p_h)

                    positive_annotations[i][0] /= w
                    positive_annotations[i][1] /= h
                    positive_annotations[i][2] /= w
                    positive_annotations[i][3] /= h

                    offsets.append(np.array([t_x, t_y, t_w, t_h]))

            negative_annotation_path = os.path.join(
                root_dir, "Annotations", sample_name + "_0.csv"
            )
            negative_annotations = np.loadtxt(
                negative_annotation_path, dtype="float32", delimiter=" "
            )
            negative_annotations[:, 0] /= w
            negative_annotations[:, 1] /= h
            negative_annotations[:, 2] /= w
            negative_annotations[:, 3] /= h
            # positive_annotationsとoffsetsを結合
            offsets = np.array(offsets).reshape(-1, 4)
            positive_annotations = np.concatenate(
                (positive_annotations, offsets), axis=1
            )

            annotation_dict[str(idx)] = {
                "positive": positive_annotations,
                "negative": negative_annotations,
            }

        self.jpeg_images = jpeg_images
        self.annotation_dict = annotation_dict

    def __getitem__(self, index: int):
        """
        positiveな領域に関してはboxの座標とオフセットを返す
        negativeな領域に関してはboxの座標のみを返す
        :param index:
        :return:
        """
        assert index < len(self.jpeg_images), "現在のデータセットの合計数: %d、入力インデックス: %d" % (
            len(self.jpeg_images),
            index,
        )

        image = self.jpeg_images[index]
        annotation_dict = self.annotation_dict[str(index)]
        positive_annotations = annotation_dict["positive"]
        negative_annotations = annotation_dict["negative"]

        positive_num = 32
        negative_num = 32

        if len(positive_annotations) < positive_num:
            positive_num = len(positive_annotations)
            negative_num = 64 - positive_num

            positive_array = positive_annotations
        else:
            positive_array = positive_annotations[
                random.sample(range(positive_annotations.shape[0]), positive_num)
            ]
        positive_bbox = positive_array[:, :4]
        # zero array [negative_num, 4]. this is dummy data
        negative_offset = np.zeros((negative_num, 4))
        offset = positive_array[:, 4:]
        # concat negative offset and offset
        offset = np.concatenate((offset, negative_offset), axis=0)

        negative_array = negative_annotations[
            random.sample(range(negative_annotations.shape[0]), negative_num)
        ]

        # rect_array = np.vstack((positive_array, negative_array))
        rect_array = np.vstack((positive_bbox, negative_array))
        # targets = np.hstack((np.ones(positive_num), np.zeros(negative_num)))
        # make one-hot vector
        targets = np.zeros((positive_num + negative_num, 2))
        targets[:positive_num, 0] = 1
        targets[positive_num:, 1] = 1

        if self.transform:
            image = self.transform(image)

        return image, targets, rect_array, offset

    def __len__(self) -> int:
        return len(self.jpeg_images)

    def get_bndbox(self, bndboxes, positive):
        """
        入力されたpositiveなboxに対して、最もIoUが高いgt boxを返す
        :param bndboxes: [n, 4]
        :param positive: [4]
        :return: [4]
        """

        if len(bndboxes.shape) == 1:
            return bndboxes
        else:
            scores = iou(positive, bndboxes)
            return bndboxes[np.argmax(scores)]