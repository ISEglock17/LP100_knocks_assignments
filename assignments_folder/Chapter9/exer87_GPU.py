"""
言語処理100本ノック 第8章課題

87. 確率的勾配降下法によるCNNの学習

確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，問題86で構築したモデルを学習せよ．
訓練データ上の損失と正解率，評価データ上の損失と正解率を表示しながらモデルを学習し，適当な基準（例えば10エポックなど）で終了させよ．

【参考にしたサイト】
https://zero2one.jp/learningblog/cnn-for-beginners/

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
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, out_channels, kernel_heights, stride, padding, emb_weights=None):
        super().__init__()
        if emb_weights != None: # 指定があれば，重みを指定通りに定義する
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.conv = nn.Conv2d(1, out_channels, (kernel_heights, emb_size), stride, (padding, 0))
        self.drop = nn.Dropout(0.4)
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
#   モデルの定義
# -----------------------------------------
# パラメータ設定
VOCAB_SIZE = len(set(word_id.values())) + 2  # 辞書のIDの数 + 予備 + パディングID
EMB_SIZE = 300
PADDING_IDX = len(set(word_id.values())) + 1
OUTPUT_SIZE = 4
OUT_CHANNELS = 100
KERNEL_HEIGHTS = 3
STRIDE = 1
PADDING = 1


# ランダムな埋め込み重みを生成
weights = torch.rand(VOCAB_SIZE, EMB_SIZE)

# モデル定義
net = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, emb_weights=weights)
net.train()

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
        
        # GPUの使用確認
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.get_device_name())
        print("使用しているデバイスは", device)
        
        net.to(device)
        
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
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                
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
#   学習
# -----------------------------------------
# trainモードに設定
net.train()

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# 最適化手法の定義
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 確率的勾配降下法を用いる。model.parametersでモデルの学習可能なすべてのパラメータ(重みとバイアス)をoptimizerに渡す。
# lr=0.01で学習率を設定する。momentum=0.9で前回の更新の影響をどれだけ残すかの調整を行う。
# 参考にしたサイト　https://qiita.com/mathlive/items/2c67efa2d451ea1da1b1

# エポック数
num_epochs = 10
train_loss, train_acc, valid_loss, valid_acc = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

"""
リーダブルコードの実践

"""