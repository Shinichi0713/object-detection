import os
import mojimoji
import glob

def change_filename(filename_target):
    filename_renamed = mojimoji.zen_to_han(filename_target).replace("..csv",".csv")
    os.rename(filename_target, filename_renamed)


if __name__ == "__main__":
    folder_targets = ["./data/finetune_car/train/bndboxs/", "./data/finetune_car/val/bndboxs/"]
    for folder_target in folder_targets:
        files_target = glob.glob(folder_target + "*.csv")
        for file_target in files_target:
            change_filename(file_target)