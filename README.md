<!----------------------------------------------------------------------------------------------------------------------
#
#   Title
#
# --------------------------------------------------------------------------------------------------------------------->
# em_simsiam
<!----------------------------------------------------------------------------------------------------------------------
#
#   Description
#
# --------------------------------------------------------------------------------------------------------------------->
Customized repository for [SimSiam](https://github.com/facebookresearch/simsiam) for Emergent System Lab.  


*   Maintainer: Shoichi Hasegawa ([hasegawa.shoichi@em.ci.ritsumei.ac.jp](mailto:hasegawa.shoichi@em.ci.ritsumei.ac.jp)).
*   Author: Taichi Sakaguchi ([sakaguchi.taichi@em.ci.ritsumei.ac.jp](mailto:sakaguchi.taichi@em.ci.ritsumei.ac.jp)).

<!----------------------------------------------------------------------------------------------------------------------
#
#   Table of Contents
#
# --------------------------------------------------------------------------------------------------------------------->
**Table of Contents:**
*   [Requirements](#requirements)
*   [Getting Started](#getting-started)
*   [How to use](#how-to-use)
*   [Programs](#programs)
*   [References](#references)

<!----------------------------------------------------------------------------------------------------------------------
#
#   Requirement
#
# --------------------------------------------------------------------------------------------------------------------->
## Requirements
* Required
    * Ubuntu: 20.04  
    * ROS: Noetic  
    * Python: 3.8
    * pytorch>=1.8

* Confirmed Condition
```
Ubuntu: 20.04LTS
CUDA: 11.2
NVIDIA driver: 570.133.20
ROS: Noetic  
Python: 3.8.10  
torch: 1.9.1+cu111
torchvision: 0.10.1+cu111
```

<!----------------------------------------------------------------------------------------------------------------------
#
#   Getting Started
#
# --------------------------------------------------------------------------------------------------------------------->
## Getting Started
```shell
git clone https://gitlab.com/general-purpose-tools-in-emlab/self-supervised-learning/em_simsiam.git

```

```shell
cd em_simsiam
bash setup_env.sh
```

<!----------------------------------------------------------------------------------------------------------------------
#
#   How to use
#
# --------------------------------------------------------------------------------------------------------------------->
## How to use
### Step1: SimSiam または PhiNet を選択して自己教師あり学習
コマンド実行直後に実行するモデル名の入力を求められます。
SimSiam : s（小文字） 
PhiNet : p（小文字）

※ xphinet は実装途中のため 実行できません（コード上でも無効化しています）。

※データセットとepoch数はコマンドラインから変更できないため、`train_simsiam.py` の `epochs = 10` などを直接編集する。  

データセット（例：CIFAR-10）でSimSiamを自己教師あり学習し、各エポックでk-NNの検証とTensorBoardログ、学習曲線の画像出力、チェックポイントの保存まで行う.  

```shell
python train_simsiam.py
```

```shell
# 学習の様子を確認
tensorboard --logdir log
# ブラウザで http://localhost:6006 を開く
```

学習が終わると、result_figureフォルダに以下が生成される. （※データセットに関係なく，ファイル名がCIFAR10になっています）
- `CIFAR10_resnet18_loss.png`: 学習過程におけるコサイン類似度の損失の推移 
- `CIFAR10_resnet18_kNN.png`: 学習過程におけるk-NN精度の推移 

dataフォルダにはCIFAR-10のデータセット、logフォルダにはTensorBoardのログデータが入る。  


### Step2: SimSiamとResNetを経由して得たCIFAR-10の特徴量をK-means法でクラスタリングし、似た画像がどのように集まっているか可視化

```shell
python k_means.py
```

処理が終わると、matplotlibで以下のグラフや画像表示される。  
- 主成分分析 (principal component analysis (PCA)) における累積寄与率 (cumulative contribution rate): 各主成分の寄与率を大きい順に足し上げたもので、そこまでの主成分でデータの持っていた情報量がどのくらい説明されているかを示す。
- PCAにおける各主成分の寄与率 (contribution rate of each principal component): ある主成分の固有値を表す情報が、データのすべての情報の中でどのくらいの割合を占めるかを表す。
- PCAの結果を2次元散布図として可視化: SimSiamで得られた特徴量をPCAで次元圧縮し、各クラスごとに色分けして配置


またresult_figureフォルダには、
- tsne_CIFAR10_100_ResNet18.png: t-SNEの結果を2次元散布図として可視化したもの。
- top_of_10.png: k-means法でクラスタリングを行った結果の代表画像 (各クラスタ中心に最も近いサンプル) をまとめたタイル画像. 各行が1つのクラスタを表し、その行に並ぶ10枚が「そのクラスタ中心に最も近い上位10枚の画像」.

Terminalには、Adjusted Rand Index (ARI) や主成分分析の結果の値が示されている。  
- ARI: クラスタリングの性能評価に使用されるもの. 1は完全一致、0は偶然レベルの一致、0より下は偶然以下。

### Step3: CIFAR-10で学習したSimSiamモデルの中間特徴を可視化し、画像の注目領域を可視化
```shell
python simsiam_attention_map.py
```

実行するとcifar_attention_mapフォルダに、CIFAR-10のtest画像について、(1) 元画像、(2) チャネル平均の活性マップ、(3) 活性で画素を重み付けした合成画像が保存される.  

### おまけ
```shell
python binary2image.py
```

CIFAR-10データセットのバイナリ形式から実際のPNG画像をファイル出力する変換するコード。


<!----------------------------------------------------------------------------------------------------------------------
#
#   Programs
#
# --------------------------------------------------------------------------------------------------------------------->

## Programs

- train_simsiam.py
- k-means.py
- simsiam_attention_map.py
- binary2image.py
- main_simsiam.py: Facebookが作ったSimSiamのPytorch実装したものでオリジナル。ImageNetなど大規模データでの学習を想定。今回は使用しない。単GPU非対応。


<!----------------------------------------------------------------------------------------------------------------------
#
#   References
#
# --------------------------------------------------------------------------------------------------------------------->
## References
[SimSiam](https://github.com/facebookresearch/simsiam)  
