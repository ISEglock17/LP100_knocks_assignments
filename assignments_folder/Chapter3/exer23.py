"""
言語処理100本ノック 第3章課題

23. セクション構造
記事中に含まれるセクション名とそのレベル（例えば”== セクション名 ==”なら1）を表示せよ．

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


# 課題23
import re
pattern = "={2,5}.*?={2,5}" # ={2,5}で"="が2から5つ含まれている場合を示す
section_list = re.findall(pattern, UK_article) # 正規表現パターンにしたがって，セクションを抽出してリスト化

for section in section_list:
    print(section.replace('=', '').strip() + ": LV{}".format(section.count('=') // 2 - 1))  # セクションの書式設定('='とスペースを削除)してレベルを表示する


"""
＊　出力結果　＊
国名: LV1
歴史: LV1
地理: LV1
主要都市: LV2
気候: LV2
政治: LV1
元首: LV2
法: LV2
内政: LV2
地方行政区分: LV2
外交・軍事: LV2
経済: LV1
鉱業: LV2
農業: LV2
貿易: LV2
不動産: LV2
エネルギー政策: LV2
通貨: LV2
企業: LV2
通信: LV3
交通: LV1
道路: LV2
鉄道: LV2
海運: LV2
航空: LV2
科学技術: LV1
国民: LV1
言語: LV2
宗教: LV2
婚姻: LV2
移住: LV2
教育: LV2
医療: LV2
文化: LV1
食文化: LV2
文学: LV2
哲学: LV2
音楽: LV2
ポピュラー音楽: LV3
映画: LV2
コメディ: LV2
国花: LV2
世界遺産: LV2
祝祭日: LV2
スポーツ: LV2
サッカー: LV3
クリケット: LV3
競馬: LV3
モータースポーツ: LV3
野球: LV3
カーリング: LV3
自転車競技: LV3
脚注: LV1
関連項目: LV1
外部リンク: LV1
"""


"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
セクションのレベルを表示する際に、replaceを用いて文字列の先頭の = を除去し、strip() メソッドを使用して余分なスペースを削除しています。これにより、コードが簡潔になり、理解しやすくなります。

・p.10の「2.1 明確な単語を選ぶ」で，
変数名はその役割を示すように選ばれています。たとえば、folderpathやfilenameは、ファイルのパスやファイル名を意味します。
re.findall()で使用されている正規表現のパターンpatternは、その用途が明確になるように名前が付けられています。また、section_listはセクション名のリストを示しており、その名前からその用途が推測しやすいです。

適切なコメントの利用: コメントは各行やブロックの目的を説明しており、コードの理解を助けます。正規表現のパターンが何を表しているかを説明するコメントがあることで、その行の目的が明確になっています。-

"""