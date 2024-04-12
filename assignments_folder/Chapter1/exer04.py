"""
言語処理100本ノック 第1章課題

04. 元素記号Permalink
“Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.”
という文を単語に分解し，1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字，それ以外の単語は先頭の2文字を取り出し，
取り出した文字列から単語の位置（先頭から何番目の単語か）への連想配列（辞書型もしくはマップ型）を作成せよ．

"""
# 初期化
str1 = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
index2extract = [i - 1 for i in [1, 5, 6, 7, 8, 9, 15, 16, 19]]

#文字列操作: 単語に分解，
splited_word_list = str1[0:-1].split()
extracted_word_list = [word[0] if index in index2extract else word[0:2] for index, word in enumerate(splited_word_list)]

#辞書型の生成
dict = dict(zip(extracted_word_list, range(1, len(extracted_word_list) + 1)))

#出力
print(dict)

# 出力結果
# {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mi': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20}

"""
―リーダブルコードの内容で実践したこと―
・p.10の「2.1 明確な単語を選ぶ」で，
extracted_word_listと抽出した文字列リストであることを示した。
index2extractと抽出する文字のindexであることを示した。

・p.4「1.3 小さなことは絶対にいいこと?」に従い，
辞書型の生成はまとめて内包表記などで1行に収めることもできるが，今後の発展性も考えて，2つのリストを結合する分かりやすい方法を取ることにした。

"""