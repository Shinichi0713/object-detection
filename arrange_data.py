import time
import shutil
import numpy as np
import cv2
import os
import sys
import xmltodict
import torch
import matplotlib.pyplot as plt


# train
# positive num: 66517
# negatie num: 464340
# val
# positive num: 64712
# negative num: 415134


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def parse_car_csv(csv_dir):
    csv_path = os.path.join(csv_dir, "car.csv")
    samples = np.loadtxt(csv_path, dtype="unicode")
    return samples


def parse_xml(xml_path):
    """
    アノテーションのバウンディングボックスの座標を返すためにxmlファイルをパースする
    """
    with open(xml_path, "rb") as f:
        xml_dict = xmltodict.parse(f)

        bndboxs = list()
        objects = xml_dict["annotation"]["object"]
        if isinstance(objects, list):
            for obj in objects:
                obj_name = obj["name"]
                difficult = int(obj["difficult"])
                if "car".__eq__(obj_name) and difficult != 1:
                    bndbox = obj["bndbox"]
                    bndboxs.append(
                        (
                            int(bndbox["xmin"]),
                            int(bndbox["ymin"]),
                            int(bndbox["xmax"]),
                            int(bndbox["ymax"]),
                        )
                    )
        elif isinstance(objects, dict):
            obj_name = objects["name"]
            difficult = int(objects["difficult"])
            if "car".__eq__(obj_name) and difficult != 1:
                bndbox = objects["bndbox"]
                bndboxs.append(
                    (
                        int(bndbox["xmin"]),
                        int(bndbox["ymin"]),
                        int(bndbox["xmax"]),
                        int(bndbox["ymax"]),
                    )
                )
        else:
            pass

        return np.array(bndboxs)


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


def compute_ious(rects, bndboxs):
    iou_list = list()
    for rect in rects:
        scores = iou(rect, bndboxs)
        iou_list.append(max(scores))
    return iou_list


def save_model(model, model_save_path):
    check_dir("./models")
    torch.save(model.state_dict(), model_save_path)


def plot_loss(loss_list):
    x = list(range(len(loss_list)))
    fg = plt.figure()

    plt.plot(x, loss_list)
    plt.title("loss")
    plt.savefig("./loss.png")


def get_selective_search():
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    return gs


def config(gs, img, strategy="q"):
    gs.setBaseImage(img)

    if strategy == "s":
        gs.switchToSingleStrategy()
    elif strategy == "f":
        gs.switchToSelectiveSearchFast()
    elif strategy == "q":
        gs.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)


def get_rects(gs):
    rects = gs.process()
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]

    return rects


def parse_annotation_jpeg(annotation_path, jpeg_path, gs):
    """
    正と負のサンプルを取得する（注：属性difficultがTrueのラベル付きバウンディングボックスは無視する）
    正のサンプル：候補とラベル付きバウンディングボックスの間のIoUが0.5以上のもの
    負のサンプル：IoUが0以上0.5未満。さらに負のサンプルの数を制限するために、サイズはラベルボックスの1/5より大きくなければなりません。
    """
    img = cv2.imread(jpeg_path)

    config(gs, img, strategy="q")
    # 候補領域の算出
    rects = get_rects(gs)
    # ラベルのバウンディングボックスを取得する
    bndboxs = parse_xml(annotation_path)

    # ラベルのバウンディングボックスの最大サイズを取得する
    maximum_bndbox_size = 0
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        bndbox_size = (ymax - ymin) * (xmax - xmin)
        if bndbox_size > maximum_bndbox_size:
            maximum_bndbox_size = bndbox_size

    # 候補の提案とラベル付きバウンディングボックスのIoUを取得する
    iou_list = compute_ious(rects, bndboxs)

    positive_list = list()
    negative_list = list()
    for i in range(len(iou_list)):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)

        iou_score = iou_list[i]
        if iou_list[i] >= 0.5:

            positive_list.append(rects[i])
        if 0 < iou_list[i] < 0.5 and rect_size > maximum_bndbox_size / 5.0:

            negative_list.append(rects[i])
        else:
            pass

    path_save = annotation_path.replace("voc_car", "finetune_car").replace("Annotations", "bndboxs").replace("xml", ".csv")
    print(path_save)
    np.savetxt(
        path_save,
        bndboxs,
        fmt="%d",
        delimiter=" ",
    )
    return positive_list, negative_list


if __name__ == "__main__":
    car_root_dir = "./data/voc_car/"
    finetune_root_dir = "./data/finetune_car/"
    check_dir(finetune_root_dir)

    gs = get_selective_search()
    for name in ["train", "val"]:
        src_root_dir = os.path.join(car_root_dir, name)
        src_annotation_dir = os.path.join(src_root_dir, "Annotations")
        src_jpeg_dir = os.path.join(src_root_dir, "JPEGImages")

        dst_root_dir = os.path.join(finetune_root_dir, name)
        dst_annotation_dir = os.path.join(dst_root_dir, "Annotations")
        dst_jpeg_dir = os.path.join(dst_root_dir, "JPEGImages")
        check_dir(dst_root_dir)
        check_dir(dst_annotation_dir)
        check_dir(dst_jpeg_dir)

        total_num_positive = 0
        total_num_negative = 0

        samples = parse_car_csv(src_root_dir)

        src_csv_path = os.path.join(src_root_dir, "car.csv")
        dst_csv_path = os.path.join(dst_root_dir, "car.csv")
        shutil.copyfile(src_csv_path, dst_csv_path)
        for sample_name in samples:
            since = time.time()

            src_annotation_path = os.path.join(src_annotation_dir, sample_name + ".xml")
            src_jpeg_path = os.path.join(src_jpeg_dir, sample_name + ".jpg")

            positive_list, negative_list = parse_annotation_jpeg(
                src_annotation_path, src_jpeg_path, gs
            )
            total_num_positive += len(positive_list)
            total_num_negative += len(negative_list)

            dst_annotation_positive_path = os.path.join(
                dst_annotation_dir, sample_name + "_1" + ".csv"
            )
            dst_annotation_negative_path = os.path.join(
                dst_annotation_dir, sample_name + "_0" + ".csv"
            )
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample_name + ".jpg")

            shutil.copyfile(src_jpeg_path, dst_jpeg_path)

            np.savetxt(
                dst_annotation_positive_path,
                np.array(positive_list),
                fmt="%d",
                delimiter=" ",
            )
            np.savetxt(
                dst_annotation_negative_path,
                np.array(negative_list),
                fmt="%d",
                delimiter=" ",
            )

            time_elapsed = time.time() - since
            print(
                "parse {}.png in {:.0f}m {:.0f}s".format(
                    sample_name, time_elapsed // 60, time_elapsed % 60
                )
            )
        print("%s positive num: %d" % (name, total_num_positive))
        print("%s negative num: %d" % (name, total_num_negative))
    print("done")