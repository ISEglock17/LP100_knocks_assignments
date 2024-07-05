"""
言語処理100本ノック 第8章課題

86. 畳み込みニューラルネットワーク (CNN)

ID番号で表現された単語列x=(x1,x2,…,xT)
がある．ただし，T
は単語列の長さ，xt∈RV
は単語のID番号のone-hot表記である（V
は単語の総数である）．畳み込みニューラルネットワーク（CNN: Convolutional Neural Network）を用い，単語列x
からカテゴリy
を予測するモデルを実装せよ．

ただし，畳み込みニューラルネットワークの構成は以下の通りとする．

単語埋め込みの次元数: dw
畳み込みのフィルターのサイズ: 3 トークン
畳み込みのストライド: 1 トークン
畳み込みのパディング: あり
畳み込み演算後の各時刻のベクトルの次元数: dh
畳み込み演算後に最大値プーリング（max pooling）を適用し，入力文をdh
次元の隠れベクトルで表現
すなわち，時刻t
の特徴ベクトルpt∈Rdh
は次式で表される．

pt=g(W(px)[emb(xt−1);emb(xt);emb(xt+1)]+b(p))
ただし，W(px)∈Rdh×3dw,b(p)∈Rdh
はCNNのパラメータ，g
は活性化関数（例えばtanh
やReLUなど），[a;b;c]
はベクトルa,b,c
の連結である．なお，行列W(px)
の列数が3dw
になるのは，3個のトークンの単語埋め込みを連結したものに対して，線形変換を行うためである．

最大値プーリングでは，特徴ベクトルの次元毎に全時刻における最大値を取り，入力文書の特徴ベクトルc∈Rdh
を求める．c[i]
でベクトルc
のi
番目の次元の値を表すことにすると，最大値プーリングは次式で表される．

c[i]=max1≤t≤Tpt[i]
最後に，入力文書の特徴ベクトルc
に行列W(yc)∈RL×dh
とバイアス項b(y)∈RL
による線形変換とソフトマックス関数を適用し，カテゴリy
を予測する．

y=softmax(W(yc)c+b(y))
なお，この問題ではモデルの学習を行わず，ランダムに初期化された重み行列でy
を計算するだけでよい．

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

# サンプル用のトークナイザーを定義
def tokenizer(text):
    return text2id(text)

# サンプルのテキストを用意
sample = "This is a sample sentence for testing"

# モデル定義
net = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, emb_weights=weights)

# トークナイズしてテンソルに変換
x = torch.tensor([tokenizer(sample)], dtype=torch.int64)

print(x)
print(x.size())
print(net(x))

""" 
出力確認
tensor([[ 104,   34,   17,    0,    0,    8, 8819]])
torch.Size([1, 7])
tensor([[0.3945, 0.1738, 0.1499, 0.2817]], grad_fn=<SoftmaxBackward0>)
"""

"""
リーダブルコードの実践

"""