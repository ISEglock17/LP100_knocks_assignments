"""
言語処理100本ノック 第3章課題

22. カテゴリ名の抽出
記事のカテゴリ名を（行単位ではなく名前で）抽出せよ．

"""
import json

# 初期化
folderpath = "./assignments_folder/Chapter3/"
filename = "jawiki-country.json"

# JSONファイル読み込み
with open(folderpath + filename, "r", encoding="utf-8") as f:
    data_list = f.readlines()
    article_list = [json.loads(data) for data in data_list]

# イギリスに関する記事JSONファイルの抽出
    UK_article = str(list(filter(lambda x: x["title"] == "イギリス", article_list))[0])


# 課題22
import re
pattern = "\[\[Category:(.*?)(?:\|.*?|)\]\]"   # [[Category: (.*? 抽出する文字列)(?: 抽出しない文字列)]]を正規表現で示す。
category_list = re.findall(pattern, UK_article) # 正規表現パターンにしたがって，カテゴリーを抽出してリスト化

for category in category_list:  # リストの中身を順次出力
    print(category)


"""
＊　出力結果　＊
イギリス
イギリス連邦加盟国
英連邦王国
G8加盟国
欧州連合加盟国
海洋国家
現存する君主国
島国
1801年に成立した国家・領域
"""


"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
Pythonの文字列の特性を活かしてスライス[0:1]でスマートにまとめた。
・p.10の「2.1 明確な単語を選ぶ」で，
str_reversedと逆順にしたことを示した。

"""