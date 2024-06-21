"""
言語処理100本ノック 第8章課題

79 多層ニューラルネットワーク
問題78のコードを改変し，バイアス項の導入や多層化など，ニューラルネットワークの形状を変更しながら，高性能なカテゴリ分類器を構築せよ

"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 77の追加部
import time

# -----------------------------------------
#   クラス，関数定義
# -----------------------------------------

# データセットクラス
class NewsDataset(Dataset):
    def __init__(self, x, y, phase="train"):
        self.x = x
        self.y = y
        self.phase = phase

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ニューラルネットワーク
class SLNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SLNet, self).__init__()
        self.fc = nn.Linear(input_size, output_size)  # 全結合層を定義

    def forward(self, x):
        return self.fc(x)  # ロジットを出力
    
# ニューラルネットワーク
class MLNet(nn.Module):
    def __init__(self, input_size, mid1_size, mid2_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, mid1_size),
            nn.BatchNorm1d(mid1_size),
            nn.ReLU(),
            nn.Linear(mid1_size, mid2_size),
            nn.BatchNorm1d(mid2_size),
            nn.Tanh(),
            nn.Linear(mid2_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)



def train_model(model, dataloaders_dict, criterion, optimizer, epoch_num):
    """
        学習用関数
        
        ＊引数＊
        model: ニューラルネットワークモデル
        dataloaders_dict: データローダー
        criterion: 損失関数(クロスエントロピー損失など)
        optimizer: 確率的勾配降下法(重みとバイアスを変更するアルゴリズム)
        epoch_num: エポック数
    """
    # 78 追加部
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス:", device)
    
    model.to(device)
    
    torch.backends.cudnn.benchmark = True
    
    # 75 追加部
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    
    prog_times = []
    
    for epoch in range(epoch_num): # エポック数を持ってくる
        # 77 追加部
        start_time = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("--------------------------------------------")

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()  # 訓練モード
            else:
                model.eval()  # 検証モード

            epoch_loss = 0.0    # epochの損失和
            epoch_corrects = 0  # epochの正解数

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                optimizer.zero_grad()  # 勾配の初期化

                with torch.set_grad_enabled(phase == "train"):  # trainのときのみ，勾配計算を有効にする
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)  # 損失の計算
                    _, preds = torch.max(outputs, 1)  # ラベルの出現率の予測

                    if phase == "train":
                        loss.backward()  # 勾配の計算
                        optimizer.step()  # 重みの更新

                    epoch_loss += loss.item() * inputs.size(0)  # 平均損失に対してバッチサイズをかけて，損失を計算
                    epoch_corrects += torch.sum(preds == labels.data)   #　予測とラベルが等しいものを集計

            epoch_loss /= len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            
            # 75 追加部
            if phase == "train":
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)

            print(f"{phase} 損失: {epoch_loss:.4f}, 正解率: {epoch_acc:.4f}")
            
        end_time = time.time()
        prog_times.append(end_time - start_time)
        
        print(f"バッチサイズ: {batch_size}，経過時間: {prog_times[-1]:.4f}")
        
    return train_loss, train_acc, valid_loss, valid_acc, sum(prog_times) / len(prog_times)


batch_sizes = [64]
prog_times = []
for batch_size in batch_sizes:
    print("現在のバッチサイズは，{}".format(batch_size))
    
    # -----------------------------------------
    #   データセットのロード
    # -----------------------------------------

    # データのロードと型変換
    x_train = torch.load("./assignments_folder/Chapter8/x_train.pt").float()
    x_valid = torch.load("./assignments_folder/Chapter8/x_valid.pt").float()
    x_test = torch.load("./assignments_folder/Chapter8/x_test.pt").float()

    y_train = torch.load("./assignments_folder/Chapter8/y_train.pt").long()
    y_valid = torch.load("./assignments_folder/Chapter8/y_valid.pt").long()
    y_test = torch.load("./assignments_folder/Chapter8/y_test.pt").long()

    # データセットとデータローダーの準備
    train_dataset = NewsDataset(x_train, y_train, phase="train")
    valid_dataset = NewsDataset(x_valid, y_valid, phase="val")
    test_dataset = NewsDataset(x_test, y_test, phase="test")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dataloaders_dict = {"train": train_dataloader, "valid": valid_dataloader, "test": test_dataloader}



    # -----------------------------------------
    #   モデル，損失関数，最適化手法の定義
    #
    #   確率的勾配降下法を用いる。model.parametersでモデルの学習可能なすべてのパラメータ(重みとバイアス)をoptimizerに渡す。
    #   lr=0.01で学習率を設定する。momentum=0.9で前回の更新の影響をどれだけ残すかの調整を行う。
    #   参考にしたサイト　https://qiita.com/mathlive/items/2c67efa2d451ea1da1b1
    # -----------------------------------------

    model = MLNet(300, 250, 125, 4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  


    # -----------------------------------------
    #   モデルの学習
    # -----------------------------------------

    # 学習の実行
    num_epochs = 10
    train_loss, train_acc, valid_loss, valid_acc, prog_time = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs)
    prog_times.append(prog_time)

    # 学習したモデルの保存
    model_scripted = torch.jit.script(model)
    model_scripted.save("./assignments_folder/Chapter8/model_scripted2.pth")


# -----------------------------------------
#   図の作成
# -----------------------------------------

epochs = np.arange(num_epochs)

# 図を作成
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# 左のサブプロット（Loss）
axs[0].plot(epochs, train_loss, label='train')
axs[0].plot(epochs, valid_loss, label='valid')
axs[0].set_title('Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend(loc='best')

# 右のサブプロット（Accuracy）
axs[1].plot(epochs, train_acc, label='train')
axs[1].plot(epochs, valid_acc, label='valid')
axs[1].set_title('Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].legend(loc='best')

# レイアウトの調整と保存
plt.tight_layout()
plt.savefig('./assignments_folder/Chapter8/fig79.png')

# グラフの表示
plt.show()


# -----------------------------------------
#   補足情報
# -----------------------------------------

""" 
...
出力結果
Epoch 9/10
--------------------------------------------
100%|██████████████████| 167/167 [00:00<00:00, 531.84it/s] 
train Loss: 0.4989, Acc: 0.8290
100%|████████████████████| 21/21 [00:00<00:00, 656.31it/s] 
val Loss: 0.4914, Acc: 0.8276
Epoch 10/10
--------------------------------------------
100%|██████████████████| 167/167 [00:00<00:00, 566.10it/s] 
train Loss: 0.4847, Acc: 0.8359
100%|████████████████████| 21/21 [00:00<00:00, 875.17it/s] 
val Loss: 0.4785, Acc: 0.8298
"""


"""
リーダブルコードの実践

明確な名前付け
クラス名: NewsDataset, SLNet は、それぞれデータセット管理とシンプルなニューラルネットワークの役割を明確に反映。
関数名: train_model はモデルを訓練する関数であることが直感的に分かる。
変数名: x_train, y_train, criterion, optimizer など、役割を明確に示す名前が使用されている。
一貫性のあるスタイル
インデント: 4スペースで統一されている。
命名規則: スネークケース（小文字とアンダースコア）で統一。
適切なコメント
クラスと関数の説明: 各クラスと関数の上部にコメントで説明を追加し、役割を明確にしている。
重要な処理: 各処理ステップ（データセットの作成、データローダーの準備、学習プロセスなど）について、適切にコメントを追加している。
関数の分割
単一責任の原則:
NewsDataset クラスはデータセットの読み込みと管理。
SLNet クラスはニューラルネットワークの定義。
train_model 関数はモデルの訓練プロセスを実行。
DRY原則（Don"t Repeat Yourself）
データローダーの生成: dataloaders_dict を用いて、データローダーの再利用を容易にしている。
データセットのロードと型変換: 同じ処理をまとめて簡潔に記述。
エラーハンドリング
型変換: float() や long() を用いて、PyTorchモデルに適した型へ明示的に変換。
損失計算: loss.backward() の前に optimizer.zero_grad() で勾配を初期化。
コードの再利用性と保守性
モジュール化: データセットのクラス化、モデルのクラス化、訓練プロセスの関数化により、再利用性と保守性を向上。
パラメータ化: input_size, output_size をコンストラクタで受け取り、汎用的なモデル定義を可能にしている。
学習プロセスの明確化
学習と評価の切り替え: model.train() と model.eval() を使い分けて、学習と評価のモードを明確にしている。
訓練プロセスの進行表示: tqdm を使用して進行状況を表示し、視覚的なフィードバックを提供。
バッチ処理: データローダーを使用してバッチ処理を行い、効率的な学習を実現。
学習結果の出力
損失と精度の表示: 各エポック終了後に損失と精度を出力し、学習の進行状況を可視化。

"""