"""
言語処理100本ノック 第8章課題

80

問題51で構築した学習データ中の単語にユニークなID番号を付与したい．
学習データ中で最も頻出する単語に1，2番目に頻出する単語に2，……といった方法で，
学習データ中で2回以上出現する単語にID番号を付与せよ．そして，与えられた単語列に対して，
ID番号の列を返す関数を実装せよ．ただし，出現頻度が2回未満の単語のID番号はすべて0とせよ．

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


# 分割データの確認
print("学習データ")
print(train["CATEGORY"].value_counts())
print("検証データ")
print(valid["CATEGORY"].value_counts())
print("評価データ")
print(test["CATEGORY"].value_counts())


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


# 出力テスト
for i in range(10):
    text = train.at[i, "TITLE"]
    print(text)
    print(text2id(text))
    print()

""" 
＊出力結果＊

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
Fitch Affirms Coventry Building Society's Mortgage Covered Bonds at 'AAA  
[235, 256, 0, 5017, 0, 1020, 5018, 288, 16, 0]

UPDATE 0-Lenovo expects IBM Mobility deals to be completed by year end
[6, 0, 1717, 2157, 5019, 1718, 1, 71, 0, 28, 317, 429]

Sara Gilbert and Linda Perry's wedding included the entire Talk crew
[4028, 4029, 13, 4030, 0, 223, 6591, 12, 2846, 1719, 4031]

Showtime Developing Spike Lee's 'She's Gotta Have It' As Comedy Series
[0, 5020, 2478, 0, 5021, 6592, 181, 0, 64, 1922, 1293]

Germany prefers Siemens tie-up with Alstom to GE deal source
[1720, 6593, 1185, 3349, 22, 407, 1, 620, 85, 1186]

SPOILER ALERT Justice served on the privy council Tyrion Lannister's fate is
[4032, 4033, 3350, 0, 4, 12, 0, 0, 6594, 0, 6595, 34]

Chris Evans Will Walk Away From Acting After Marvel Contract Expires
[52, 1557, 55, 3351, 1923, 35, 2158, 29, 1410, 2847, 6596]

Gisele Bundchen Gisele Bundchen and Tom Brady put LA mansion up for sale
[3352, 3353, 3352, 3353, 13, 889, 2159, 1558, 653, 4034, 30, 8, 734]

Amy Adams Gives Her First Class Plane Seat To US Soldier
[1411, 2160, 788, 83, 49, 4035, 1294, 2848, 11, 9, 2849]

Russia Cuts Gas to Ukraine While Maintaining Flow to EU
[289, 487, 654, 1, 65, 1187, 0, 5022, 1, 408]
"""


"""
リーダブルコードの実践

"""