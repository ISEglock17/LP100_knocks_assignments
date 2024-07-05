"""
言語処理100本ノック 第8章課題

88. パラメータチューニング

問題85や問題87のコードを改変し，ニューラルネットワークの形状やハイパーパラメータを調整しながら，高性能なカテゴリ分類器を構築せよ．


"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from collections import Counter
from tqdm import tqdm
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import optuna

# -----------------------------------------
#   データ準備
# -----------------------------------------
# 単語ベクトルの導入
file = "./assignments_folder/Chapter7/GoogleNews-vectors-negative300.bin.gz"
model = KeyedVectors.load_word2vec_format(file, binary=True)

# newsCorporaから記事をDataFrame形式で読み取る
file = "./assignments_folder/Chapter6/news+aggregator/newsCorpora.csv"
data = pd.read_csv(file, encoding="utf-8", header=None, sep="\t", names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])

# 特定のpublisherのみを抽出する
publishers = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]
data = data.loc[data["PUBLISHER"].isin(publishers), ["TITLE", "CATEGORY"]].reset_index(drop=True)

# dataの前処理を行う
for i in range(len(data["TITLE"])):
    text = data["TITLE"][i]
    text_clean = re.sub(r"[\"\".,:;\(\)#\|\*\+\!\?#$%&/\]\[\{\}]", "", text)
    text_clean = re.sub("[0-9]+", "0", text_clean)
    text_clean = re.sub("\s-\s", " ", text_clean)
    data.at[i, "TITLE"] = text_clean
# 確認用デバッグ
# print(data)

# 学習用(Train)，検証用(Valid)，評価用(Test)に分割する
train, valid_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=15, stratify=data["CATEGORY"])     # train_test_splitを用いて，分割する。stratifyでカテゴリの分布が元のデータセットと同じになるように指定。
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=15, stratify=valid_test["CATEGORY"])

train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)
test = test.reset_index(drop=True)

"""
# 分割データの確認
print("学習データ")
print(train["CATEGORY"].value_counts())
print("検証データ")
print(valid["CATEGORY"].value_counts())
print("評価データ")
print(test["CATEGORY"].value_counts())
"""

# -----------------------------------------
#   単語辞書生成
# -----------------------------------------
# 単語(words)の抽出
words = []
for text in train["TITLE"]:
    for word in text.rstrip().split():
        words.append(word)

c = Counter(words)

word_id = {}
for i, count in enumerate(c.most_common()): # countの例: ("Eurosail-UK", 1)
    if count[1] >= 2:
        word_id[count[0]] = i + 1   # word_idに単語の出現頻度を記録


def text2id(text: str) -> list:
    """
        テキストからIDリストを取得する関数
    """
    words = text.rstrip().split()
    return [word_id.get(word, 0) for word in words]

"""
# 出力テスト
for i in range(10):
    text = train.at[i, "TITLE"]
    print(text)
    print(text2id(text))
    print()
"""


# -----------------------------------------
#   CNN
# ----------------------------------------- 
class CNN(nn.Module):
    """ 
        vocab_size: ボキャブラリー（語彙）のサイズ。
        emb_size: 埋め込みベクトルのサイズ。
        padding_idx: パディングのインデックス。
        output_size: 出力層のサイズ（分類するクラス数など）。
        out_channels: 畳み込み層の出力チャンネル数。
        kernel_heights: 畳み込みカーネルの高さ（フィルターサイズ）。
        stride: 畳み込みのストライド（移動幅）。
        padding: 畳み込みのパディング。
        emb_weights: 事前学習済みの埋め込み重み（省略可能）。
    """
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, out_channels, kernel_heights, stride, padding, dropout_rate, emb_weights=None):
        super(CNN, self).__init__()
        if emb_weights is not None:  # 指定があれば，重みを指定通りに定義する
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.conv = nn.Conv2d(1, out_channels, (kernel_heights, emb_size), stride, (padding, 0))
        self.drop = nn.Dropout(dropout_rate)  # ドロップアウト率を引数として受け取る
        self.fc = nn.Linear(out_channels, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        emb = self.emb(x).unsqueeze(1)  # 埋め込み層に通し，埋め込みベクトルを取得する。unsqueeze(1)で次元を増やしておく
        conv = self.conv(emb)   # 畳み込み層に通す
        act = F.relu(conv.squeeze(3))   # ReLU活性化関数を適用し，squeeze(3)で不要な次元を削除する
        max_pool = F.max_pool1d(act, act.size()[2]) # 畳み込みの出力に対して，最大プーリングを行う
        logits = self.fc(self.drop(max_pool.squeeze(2)))    # プーリングされた特徴に対して，ドロップアウトを適用し，全結合層に入力する
        logits = self.softmax(logits)   # ソフトマックスを計算
                
        return logits


# -----------------------------------------
#   データセット作成
# -----------------------------------------
class NewsDataset(Dataset):
    def __init__(self, x, y, phase="train"):
        self.x = x["TITLE"]
        self.y = y
        self.phase = phase

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        inputs = torch.tensor(text2id(self.x[index]))
        return inputs, self.y[index]


# カテゴリーのテンソル変換
category_dict = {"b": 0, "t": 1, "e":2, "m":3}
y_train = torch.from_numpy(train["CATEGORY"].map(category_dict).values)
y_valid = torch.from_numpy(valid["CATEGORY"].map(category_dict).values)
y_test = torch.from_numpy(test["CATEGORY"].map(category_dict).values)


train_dataset = NewsDataset(train, y_train, phase="train")
valid_dataset = NewsDataset(valid, y_valid, phase="val")
test_dataset = NewsDataset(test, y_test, phase="val")


# データローダー用のカスタムcollate_fn
def my_collate_fn(batch):
    inputs, labels = zip(*batch)
    lengths = [len(x) for x in inputs]
    max_length = max(lengths)
    padded_inputs = torch.zeros(len(inputs), max_length, dtype=torch.long)
    for i, x_len in enumerate(lengths):
        padded_inputs[i, :x_len] = inputs[i]
    labels = torch.tensor(labels)
    return padded_inputs, labels

# データローダーの作成
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=my_collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False, collate_fn=my_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=my_collate_fn)

dataloaders_dict = {"train": train_dataloader, "val": valid_dataloader, "test": test_dataloader}

def get_dataloader(batch_size=64):
    """
    トレーニング、バリデーション、テストのデータローダーを作成する関数。
    """
    train_dataset = NewsDataset(train, y_train, phase="train")
    valid_dataset = NewsDataset(valid, y_valid, phase="val")
    test_dataset = NewsDataset(test, y_test, phase="val")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

    dataloaders_dict = {"train": train_dataloader, "val": valid_dataloader, "test": test_dataloader}
    return dataloaders_dict

# -----------------------------------------
#   学習用関数の定義
# -----------------------------------------
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    """
        モデルの学習を行う関数
    """
    
    # 損失とaccの定義
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    
    # epochごとのループ
    for epoch in range(num_epochs):
        print("＊現在のエポック {} / {}".format(epoch + 1, num_epochs))
        print("* -------------------------------------------- *")
        
        """
        # GPUの使用確認
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.get_device_name())
        print("使用しているデバイスは", device)
        
        net.to(device)
        """
        
        # フェイズ判定
        for phase in ["train", "val"]:
            if phase == "train":
                net.train() # 訓練モード
            else:
                net.eval() # 検証モード
            
            epoch_loss = 0.0 
            epoch_corrects = 0
        
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                optimizer.zero_grad() # optimizerを初期化
                """
                inputs = inputs.to(device)
                labels = labels.to(device)
                """
                
                # 順伝播計算
                with torch.set_grad_enabled(phase == "train"):  # フェイズがtrainのとき，勾配計算をONにする
                    outputs = net(inputs)
                    loss = criterion(outputs, labels) # 損失の計算
                    _, predicts = torch.max(outputs, 1) # ラベルを予想
                    
                    # 訓練時は逆伝播にする
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(predicts == labels.data)
            
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            
            if phase == "train":
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            
            print("{} 損失(loss): {:.4f}, 精度(accuracy): {:.4f}".format(phase, epoch_loss, epoch_acc))
    return train_loss, train_acc, valid_loss, valid_acc


# -----------------------------------------
#   Optunaを用いたハイパーパラメータチューニング
# -----------------------------------------
# データセットの準備
VOCAB_SIZE = 10000  # 語彙のサイズ
EMB_SIZE = 300      # 埋め込みの次元数
PADDING_IDX = 0     # パディングのインデックス
OUTPUT_SIZE = 2     # 出力クラス数（例：2クラス分類）
STRIDE = 1          # ストライド
PADDING = 0         # パディング
weights = torch.randn(VOCAB_SIZE, EMB_SIZE)  # 仮の埋め込み行列

def calc_acc(net, dataloader):
    """
        Calculate_Accuracy
        正解率(accuracy)を計算するメソッド
        accuracy の説明 https://atmarkit.itmedia.co.jp/ait/articles/2209/15/news040.html
    """
    net.eval()  # モデルを評価モードにする
    correct_num = 0
    with torch.no_grad():   # 勾配計算をオフにする
        for feature_vals, labels in dataloader:       # データローダーで，特徴量Xと正解のラベルyを取り出す
            outputs = net(feature_vals)     # 特徴量を学習済ニューラルネットワークモデルに掛けて各カテゴリーの出現率を出力
            _, preds = torch.max(outputs, 1) # 各カテゴリーの出現率のテンソルから，最大値を取るインデックスをカテゴリ名として出力 torch.max(outputs, 1)の第2引数は次元を示す
            correct_num += torch.sum(preds == labels.data)  # 正解数の集計
    return correct_num / len(dataloader.dataset)    # 正解率の計算


# OptunaのObjective関数の定義
def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    kernel_height = trial.suggest_int('kernel_height', 2, 5)
    out_channels = trial.suggest_int('out_channels', 50, 200)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.5, 0.9)  # momentum のハイパーパラメータを追加

    # `get_dataloader` を呼び出してデータローダーを取得
    dataloaders_dict = get_dataloader(batch_size=batch_size)

    # モデルの設定
    model = CNN(
        vocab_size=len(word_id) + 1,  # ボキャブラリーのサイズは辞書のサイズ + 1
        emb_size=300,  # 埋め込みベクトルのサイズ
        padding_idx=0,  # パディングのインデックス
        output_size=4,  # 出力層のサイズ（クラス数）
        out_channels=out_channels,  # 畳み込み層の出力チャンネル数
        kernel_heights=kernel_height,  # カーネルの高さ
        stride=1,  # ストライド
        padding=1,  # パディング
        dropout_rate=dropout_rate,  # ドロップアウトの割合
        emb_weights=weights  # 事前学習済みの埋め込み重み
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_loss, train_acc, valid_loss, valid_acc = train_model(
        model, dataloaders_dict, criterion, optimizer, num_epochs=10
    )

    valid_loss_final = valid_loss[-1] if valid_loss else float('inf')
    return valid_loss_final

# Optunaのスタディの設定
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

# 最良のハイパーパラメータを取得
best_params = study.best_params
best_value = study.best_value

# 最良のハイパーパラメータを出力
print(f'Best hyperparameters: {study.best_params}')
print(f'Best validation loss: {study.best_value:.4f}')

# -----------------------------------------
#   最適なハイパーパラメータでモデルを再学習
# -----------------------------------------
OUT_CHANNELS = best_params['out_channels']
KERNEL_HEIGHTS = best_params['kernel_heights']
DROPOUT_RATE = best_params['dropout_rate']
LR = best_params['lr']
MOMENTUM = best_params['momentum']

net = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, DROPOUT_RATE, emb_weights=weights)
net.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)

num_epochs = 10
train_loss, train_acc, valid_loss, valid_acc = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)



# -----------------------------------------
#   学習結果 表示
# -----------------------------------------
fig, axes = plt.subplots(1,2, figsize=(12, 6))
epochs = np.arange(num_epochs)
axes[0].plot(epochs, train_loss, label="train")
axes[0].plot(epochs, valid_loss, label="valid")
axes[0].set_title("loss")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("loss")

axes[1].plot(epochs, train_acc, label="train")
axes[1].plot(epochs, valid_acc, label="valid")
axes[1].set_title("acc")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("acc")

axes[0].legend(loc="best")
axes[1].legend(loc="best")

plt.tight_layout()
plt.savefig("./assignments_folder/Chapter9/fig88.png")
plt.show()

acc_train = calc_acc(net, train_dataloader)
acc_valid = calc_acc(net, valid_dataloader)
acc_test = calc_acc(net, test_dataloader)
print("学習データのaccuracy: {:.4f}".format(acc_train))
print("検証データのaccuracy: {:.4f}".format(acc_valid))
print("テストデータのaccuracy: {:.4f}".format(acc_test))


""" 
出力確認
誤って途中(40/50あたり)で切ってしまったので，途中の結果を表示
＊現在のエポック 9 / 10
* -------------------------------------------- *
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 334/334 [00:02<00:00, 123.77it/s] 
train 損失(loss): 0.9461, 精度(accuracy): 0.7966
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 42/42 [00:00<00:00, 208.96it/s] 
val 損失(loss): 0.9817, 精度(accuracy): 0.7616
＊現在のエポック 10 / 10
* -------------------------------------------- *
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 334/334 [00:02<00:00, 121.01it/s] 
train 損失(loss): 0.9467, 精度(accuracy): 0.7955
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 42/42 [00:00<00:00, 229.51it/s] 
val 損失(loss): 0.9820, 精度(accuracy): 0.7586
Best hyperparameters: {'batch_size': 16, 'kernel_height': 3, 'out_channels': 119, 'dropout_rate': 0.2677735840508308, 'learning_rate': 0.020915951078648447, 'momentum': 0.823206230091841}
Best validation loss: 0.9593
"""

"""
リーダブルコードの実践

"""