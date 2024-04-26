"""
言語処理100本ノック 第3章課題

29. 国旗画像のURLを取得する
テンプレートの内容を利用し，国旗画像のURLを取得せよ．（ヒント: MediaWiki APIのimageinfoを呼び出して，ファイル参照をURLに変換すればよい）

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

import re
pattern = "基礎情報(.*?\<references/\>\\\\n)"    # イギリスの記事を見ると，referencesのところが最後となっているため，そこまでを抜き出す
basic_information = re.findall(pattern, UK_article)[0]
# print(basic_information) #デバッグ用出力

pattern = "(?<=\\\\n\|)(.*?) *= *(.*?)(?=\\\\n)"    # 後読みと先読みを活用して前後が\n| \nで囲まれていることを条件とし，"="の前後のテキストを抽出する
basic_information_set = re.findall(pattern, basic_information)
# print(basic_information_set) # デバッグ用出力

info_dic = {key: value for key, value in basic_information_set} # 辞書の生成

# 強調マークアップの削除
pattern = "(\\\'){2,5}"
info_dic2 = {key: re.sub(pattern , "", value) for key, value in info_dic.items()}    # re.subを利用して，\'が2～5回繰り返されている箇所を削除する　https://www.mediawiki.org/wiki/Help:Formatting/ja　参照
# print(info_dic2)    # 出力


# 内部リンクマークアップの削除 
# リンクの記法 https://www.mediawiki.org/wiki/Help:What_links_here/ja    

pattarn_list = []
pattarn_list.append("(?<=\}\}\<br \/\>（)\[{2}")
pattarn_list.append("\[{2}.*?\|.*?px\|(?=.*?\]\])")
pattarn_list.append("(?<=(\|))\[{2}")
pattarn_list.append("(?<=\}{2}（)\[{2}")
pattarn_list.append("(?<=\>（)\[{2}.*?\|")
pattarn_list.append("(?<=（.{4}).*?\[{2}.*?\)\|")
pattarn_list.append("\[{2}.*?\|")
pattarn_list.append("(\[{2}|\]{2})")

pattarn_list.append("\{\{.*?\{\{center\|")
pattarn_list.append("\{\{.*?\|.*?\|.{2}\|")
pattarn_list.append("\<ref.*?\>.*?\<\/ref\>")
pattarn_list.append("\<ref.*?\>|\<br \/\>")
pattarn_list.append("\{\{lang\|.*?\|")
pattarn_list.append("\{\{.*?\}\}")
pattarn_list.append("\}\}")

info_dic3 = {}
for key, value in info_dic2.items():
    for pattarn in pattarn_list:
        value = re.sub(pattarn, "", value)
    info_dic3[key] = value

# print(info_dic3)  # デバッグ用出力

# 課題29
import urllib.request

# URLをエンコードする
encoded_title = urllib.parse.quote(info_dic3['国旗画像'])
url = f'https://www.mediawiki.org/w/api.php?action=query&titles=File:{encoded_title}&format=json&prop=imageinfo&iiprop=url'

# URLにリクエストを送信してレスポンスを取得する
request = urllib.request.Request(url)
with urllib.request.urlopen(request) as connection:
    # レスポンスを読み取り、JSON形式に変換する
    json_response = connection.read().decode()
    parsed_response = json.loads(json_response)

# print(parsed_response) # デバッグ用出力

# レスポンスからURLを取得する
image_url = parsed_response['query']['pages']['-1']['imageinfo'][0]['url']

# 取得したURLを出力する
print(image_url)

"""
＊　出力結果　＊
https://upload.wikimedia.org/wikipedia/commons/8/83/Flag_of_the_United_Kingdom_%283-5%29.svg
"""

"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
Pythonの文字列の特性を活かしてスライス[0:1]でスマートにまとめた。
・p.10の「2.1 明確な単語を選ぶ」で，
str_reversedと逆順にしたことを示した。

"""