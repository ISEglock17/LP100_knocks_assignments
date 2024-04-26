"""
言語処理100本ノック 第3章課題

21. カテゴリ名を含む行を抽出
記事中でカテゴリ名を宣言している行を抽出せよ．

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



# 課題21
import re
pattern = "\[\[Category:.*?\]\]"    # [[Category: (ここは任意のなるべく少ない文字)]]を正規表現で示す。
category_list = re.findall(pattern, UK_article) # 正規表現パターンにしたがって，カテゴリーを抽出してリスト化

for category in category_list:  # リストの中身を順次出力
    print(category)


"""
＊　出力結果　＊
[[Category:イギリス|*]]
[[Category:イギリス連邦加盟国]]
[[Category:英連邦王国|*]]
[[Category:G8加盟国]]
[[Category:欧州連合加盟国|元]]
[[Category:海洋国家]]
[[Category:現存する君主国]]
[[Category:島国]]
[[Category:1801年に成立した国家・領域]]
"""

"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
リスト内包表記や高階関数を使用して、コンパクトかつ効率的なコードを実現しています。
正規表現を使って一度にカテゴリ名を抽出し、それをリスト化しています。

・p.10の「2.1 明確な単語を選ぶ」で，
変数名やコメントに明確な単語が使用されています。例えば、"folderpath"や"filename"はそのまま読んで意味が理解できます。
正規表現パターンの変数名"pattern"も、何を表しているかが明確です。

"""