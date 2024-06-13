"""
言語処理100本ノック 第8章課題

70

FORMAT: ID ︙TITLE ︙URL ︙PUBLISHER ︙CATEGORY ︙STORY ︙HOSTNAME ︙TIMESTAMP

＊詳細内容＊
ID 数値ID
TITLE ニュースのタイトル 
URL URL
PUBLISHER 出版社名
CATEGORY ニュースのカテゴリー（b = ビジネス、t = 科学技術、e = エンターテインメント、m = 健康）
STORY 同じストーリーに関するニュースを含むクラスターの英数字ID
HOSTNAME URLホスト名
TIMESTAMP 1970年1月1日00:00:00 GMTからのミリ秒数で表した、ニュースが発表されたおおよその時間。

"""
import pandas as pd
import re
import numpy as np
import torch

from gensim.models import KeyedVectors

# 単語ベクトルの学習
file = "./assignments_folder/Chapter7/GoogleNews-vectors-negative300.bin.gz"
model = KeyedVectors.load_word2vec_format(file, binary=True)

# newsCorporaから記事をDataFrame形式で読み取る
file = "./assignments_folder/Chapter6/news+aggregator/newsCorpora.csv"
data = pd.read_csv(file, encoding="utf-8", header=None, sep="\t", names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])

# 特定のpublisherのみを抽出する
publishers = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]
data = data.loc[data["PUBLISHER"].isin(publishers), ["TITLE", "CATEGORY"]].reset_index(drop=True)



# 学習用(Train)，検証用(Valid)，評価用(Test)に分割する
from sklearn.model_selection import train_test_split

train, valid_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=15, stratify=data["CATEGORY"])     # train_test_splitを用いて，分割する。stratifyでカテゴリの分布が元のデータセットと同じになるように指定。
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=15, stratify=valid_test["CATEGORY"])

train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)
test = test.reset_index(drop=True)


# 分割データの確認
print("学習データ")
print(train["CATEGORY"].value_counts())
print("検証データ")
print(valid["CATEGORY"].value_counts())
print("評価データ")
print(test["CATEGORY"].value_counts())


# データの結合
df = pd.concat([train, valid, test], axis=0).reset_index(drop=True)

def word2vec(text: str):
    """
    記事のタイトルから平均単語ベクトルを計算する関数
    """
    words = text.rstrip().split()
    vec = [model.get_vector(word) for word in words if word in model.key_to_index]
    if vec:
        return np.mean(vec, axis=0)
    else:
        # もし単語が一つもモデルに存在しない場合、ゼロベクトルを返す
        return np.zeros(model.vector_size)


vecs = np.array([])
for text in df["TITLE"]:    # 記事見出しから単語列を抽出
    if len(vecs) == 0:
        vecs = word2vec(text)   # 平均単語ベクトル算出後連結
    else:
        vecs = np.vstack([vecs, word2vec(text)])

# シード値の設定
np.random.seed(1024)
torch.manual_seed(1024)

x_train = torch.from_numpy(vecs[:len(train), :])
x_valid = torch.from_numpy(vecs[len(train):len(train)+ len(valid), :])
x_test = torch.from_numpy(vecs[len(train)+ len(valid):, :])

# カテゴリーのテンソル変換
category_dict = {"b": 0, "t": 1, "e":2, "m":3}
y_train = torch.from_numpy(train["CATEGORY"].map(category_dict).values)
y_valid = torch.from_numpy(valid["CATEGORY"].map(category_dict).values)
y_test = torch.from_numpy(test["CATEGORY"].map(category_dict).values)

torch.save(x_train, "./assignments_folder/Chapter8/x_train.pt")
torch.save(x_valid, "./assignments_folder/Chapter8/x_valid.pt")
torch.save(x_test, "./assignments_folder/Chapter8/x_test.pt")
torch.save(y_train, "./assignments_folder/Chapter8/y_train.pt")
torch.save(y_valid, "./assignments_folder/Chapter8/y_valid.pt")
torch.save(y_test, "./assignments_folder/Chapter8/y_test.pt")

"""
# デバッグ用出力結果

学習データ
CATEGORY
b    4502
e    4223
t    1219
m     728
Name: count, dtype: int64
検証データ
CATEGORY
b    562
e    528
t    153
m     91
Name: count, dtype: int64
評価データ
CATEGORY
b    563
e    528
t    152
m     91
Name: count, dtype: int64
"""

"""
リーダブルコードの実践



"""