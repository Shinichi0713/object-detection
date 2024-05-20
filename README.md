# 目的
CNNモデルの構築を行い、代表的なタスクである物体検出を行う。
※本モデルは作成者がAI勉強したての頃に記載したもののため、ファイル構成など、古いやり方の部分が多々ある点をご留意ください。

# モデルの構成
Faster-RNNを構築する。<br>
具体的には、モデルは特徴検出器と、ダウンストリーム2ブロックで構成<br>
特徴検出器：画像から特徴マップを抽出する。今回はVGG16を用いる<br>
ダウンストリーム2ブロック：
・分類器：特徴マップより画像の具体的な分類を行う
・位置検出器：特徴マップよりバウンディングボックスの推定を行う

以下のようなモデルとする
```
VGG16_RoI
|
├─ features: Sequential
|    |
|    ├─ (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
|    ├─ (1): ReLU(inplace=True)
|    ├─ (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
|    ├─ (3): ReLU(inplace=True)
|    ├─ (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
|    ├─ (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
|    ├─ (6): ReLU(inplace=True)
|    ├─ (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
|    ├─ (8): ReLU(inplace=True)
|    ├─ (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
|    ├─ (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
|    ├─ (11): ReLU(inplace=True)
|    ├─ (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
|    ├─ (13): ReLU(inplace=True)
|    ├─ (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
|    ├─ (15): ReLU(inplace=True)
|    ├─ (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
|    ├─ (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
|    ├─ (18): ReLU(inplace=True)
|    ├─ (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
|    ├─ (20): ReLU(inplace=True)
|    ├─ (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
|    └─ (22): ReLU(inplace=True)
|
├─ classifier: Sequential
|    |
|    ├─ (0): Linear(in_features=25088, out_features=4096, bias=True)
|    ├─ (1): ReLU(inplace=True)
|    ├─ (2): Dropout(p=0.5, inplace=False)
|    ├─ (3): Linear(in_features=4096, out_features=4096, bias=True)
|    └─ (4): ReLU(inplace=True)
|
├─ softmax: Sequential
|    |
|    ├─ (0): Linear(in_features=4096, out_features=2, bias=True)
|    └─ (1): Softmax(dim=1)
|
└─ bbox: Sequential
     |
     ├─ (0): Linear(in_features=4096, out_features=4, bias=True)
     └─ (1): ReLU(inplace=True)

```
