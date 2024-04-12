"""
言語処理100本ノック 第1章課題

03. 円周率Permalink
“Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.”
という文を単語に分解し，各単語の（アルファベットの）文字数を先頭から出現順に並べたリストを作成せよ．

"""
import re

# 初期化
str1 = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

#文字列操作: 単語に分解，文字数カウント
splited_word_list = str1.split()
character_count_list = [len(re.sub(r"[^a-zA-Z]", "", s)) for s in splited_word_list]

#出力
print(character_count_list)

"""
―リーダブルコードの内容で実践したこと―
・p.10の「2.1 明確な単語を選ぶ」で，
character_count_listと文字数のリストであることを示した。

"""