# 目的
CNNモデルの構築を行い、代表的なタスクである物体検出を行う。

# モデルの構成
Faster-RNNを実施する。<br>
具体的には、モデルは特徴検出器と、ダウンストリーム2ブロックで構成<br>
特徴検出器：画像から特徴マップを抽出する。今回はVGG16を用いる<br>
ダウンストリーム2ブロック：
・分類器：特徴マップより画像の具体的な分類を行う
・位置検出器：特徴マップよりバウンディングボックスの推定を行う

