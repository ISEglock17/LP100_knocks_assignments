"""
言語処理100本ノック 第8章課題

81. RNNによる予測

ID番号で表現された単語列x=(x1,x2,…,xT)
がある．ただし，T
は単語列の長さ，xt∈RV
は単語のID番号のone-hot表記である（V
は単語の総数である）．再帰型ニューラルネットワーク（RNN: Recurrent Neural Network）を用い，単語列x
からカテゴリy
を予測するモデルとして，次式を実装せよ．

h→0=0,h→t=RNN−→−−(emb(xt),h→t−1),y=softmax(W(yh)h→T+b(y))
ただし，emb(x)∈Rdw
は単語埋め込み（単語のone-hot表記から単語ベクトルに変換する関数），h→t∈Rdh
は時刻t
の隠れ状態ベクトル，RNN−→−−(x,h)
は入力x
と前時刻の隠れ状態h
から次状態を計算するRNNユニット，W(yh)∈RL×dh
は隠れ状態ベクトルからカテゴリを予測するための行列，b(y)∈RL
はバイアス項である（dw,dh,L
はそれぞれ，単語埋め込みの次元数，隠れ状態ベクトルの次元数，ラベル数である）．RNNユニットRNN−→−−(x,h)
には様々な構成が考えられるが，典型例として次式が挙げられる．

RNN−→−−(x,h)=g(W(hx)x+W(hh)h+b(h))
ただし，W(hx)∈Rdh×dw，W(hh)∈Rdh×dh,b(h)∈Rdh
はRNNユニットのパラメータ，g
は活性化関数（例えばtanh
やReLUなど）である．

なお，この問題ではパラメータの学習を行わず，ランダムに初期化されたパラメータでy
を計算するだけでよい．次元数などのハイパーパラメータは，dw=300,dh=50
など，適当な値に設定せよ（以降の問題でも同様である）．


【参考にしたサイト】
nn.Embeddingの解説サイト: https://gotutiyan.hatenablog.com/entry/2020/09/02/200144
RNNの参考サイト: https://qiita.com/Mikeinu/items/7bb3f223b96fce65c110
https://ex-ture.com/blog/2021/01/12/pytorch-rnn/
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from collections import Counter

# -----------------------------------------
#   データ準備
# -----------------------------------------
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
for i, count in enumerate(c.most_common()): # countの例: ('Eurosail-UK', 1)
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

# 81部分 
import torch
from torch import nn
import random
import torch.utils.data as data

# -----------------------------------------
#   RNN
# -----------------------------------------
class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)  # nn.Embeddingは，語彙サイズ(単語IDの最大値+1), 埋め込み次元，無視するindexを指定
        self.rnn = nn.RNN(emb_size, hidden_size, batch_first=True)  # 入力層と隠れ層のサイズ指定，batch_firstで引数の順番を変更
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, h0=None):
        x = self.emb(x)
        x, h = self.rnn(x, h0)
        x = x[:, -1, :] # 最後の出力に絞る
        logits = self.fc(x)
        logits = self.softmax(x)
        return logits

# -----------------------------------------
#   モデルの定義
# -----------------------------------------
# パラメータの設定
VOCAB_SIZE = len(set(word_id.values())) + 2  # 辞書のIDの数 + 予備 + パディングID
EMB_SIZE = 300
PADDING_IDX = len(set(word_id.values())) + 1
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50

# モデル定義
model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, HIDDEN_SIZE, OUTPUT_SIZE)
print(model)

"""
＊出力確認
RNN(
  (emb): Embedding(9826, 300, padding_idx=9825)
  (rnn): RNN(300, 50, batch_first=True)
  (fc): Linear(in_features=50, out_features=4, bias=True)
)
"""

"""
リーダブルコードの実践

"""