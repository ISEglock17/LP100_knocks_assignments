"""
言語処理100本ノック 第3章課題

28. MediaWikiマークアップの除去
27の処理に加えて，テンプレートの値からMediaWikiマークアップを可能な限り除去し，国の基本情報を整形せよ．

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
# リンクの記法 https://www.mediawiki.org/wiki/Help:Links/ja#:~:text=%E5%90%8C%E4%B8%80%E3%82%A6%E3%82%A3%E3%82%AD%E4%B8%8A%E3%81%AE%E3%83%9A%E3%83%BC%E3%82%B8,%E3%83%AA%E3%83%B3%E3%82%AF%E3%81%8C%E8%A1%A8%E7%A4%BA%E3%81%95%E3%82%8C%E3%81%BE%E3%81%99%E3%80%82

pattarn_list = []
pattarn_list.append("(?<=\}\}\<br \/\>（)\[{2}")
pattarn_list.append("\[{2}.*?\|.*?px\|(?=.*?\]\])")
pattarn_list.append("(?<=(\|))\[{2}")
pattarn_list.append("(?<=\}{2}（)\[{2}")
pattarn_list.append("(?<=\>（)\[{2}.*?\|")
pattarn_list.append("(?<=（.{4}).*?\[{2}.*?\)\|")
pattarn_list.append("\[{2}.*?\|")
pattarn_list.append("(\[{2}|\]{2})")
# 課題28
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

print(info_dic3)  

"""
＊　出力結果　＊
'略名': 'イギリス', '日本語国名': 'グレートブリテン及び北アイルランド連合王国', '
公式国名': 'United Kingdom of Great Britain and Northern Ireland英語以外での正式国 
名:', '国旗画像': 'Flag of the United Kingdom.svg', '国章画像': 'イギリスの国章', '国章リンク': '（国章）', '標語': 'Dieu et mon droit（フランス語:神と我が権利）', ' 
国歌': 'ファイル:United States Navy Band - God Save the Queen.ogg', '地図画像': 'Europe-UK.svg', '位置画像': 'United Kingdom (+overseas territories) in the World (+Antarctica claims).svg', '公用語': '英語', '首都': 'ロンドン（事実上）', '最大都市': 
'ロンドン', '元首等肩書': '女王', '元首等氏名': 'エリザベス2世', '首相等肩書': '首 
相', '首相等氏名': 'ボリス・ジョンソン', '他元首等肩書1': '貴族院議長', '他元首等氏
名1': 'ノーマン・ファウラー', '他元首等肩書2': '庶民院議長', '他元首等氏名2': 'Lindsay Hoyle', '他元首等肩書3': '最高裁判所長官', '他元首等氏名3': 'ブレンダ・ヘイル', '面積順位': '76', '面積大きさ': '1 E11', '面積値': '244,820', '水面積率': '1.3%', 
'人口統計年': '2018', '人口順位': '22', '人口大きさ': '1 E7', '人口値': '6643万5600', '人口密度値': '271', 'GDP統計年元': '2012', 'GDP値元': '1兆5478億', 'GDP統計年MER': '2012', 'GDP順位MER': '6', 'GDP値MER': '2兆4337億', 'GDP統計年': '2012', 'GDP順
位': '6', 'GDP値': '2兆3162億', 'GDP/人': '36,727', '建国形態': '建国', '確立形態1': 'イングランド王国／スコットランド王国（両国とも1707年合同法まで）', '確立年月日1': '927年／843年', '確立形態2': 'グレートブリテン王国成立（1707年合同法）', '確立年 
月日2': '1707年5月1日', '確立形態3': 'グレートブリテン及びアイルランド連合王国成立 
（1800年合同法）', '確立年月日3': '1801年1月1日', '確立形態4': "現在の国号「\\'\\'\\'グレートブリテン及び北アイルランド連合王国\\'\\'\\'」に変更", '確立年月日4': '1927年4月12日', '通貨': 'UKポンド (£)', '通貨コード': 'GBP', '時間帯': '±0', '夏時間': '+1', 'ISO 3166-1': 'GB / GBR', 'ccTLD': '.uk / .gb', '国際電話番号': '44', '注記': ''}
"""


"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
Pythonの文字列の特性を活かしてスライス[0:1]でスマートにまとめた。
・p.10の「2.1 明確な単語を選ぶ」で，
str_reversedと逆順にしたことを示した。

"""