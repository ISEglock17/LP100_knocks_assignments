"""
言語処理100本ノック 第1章課題

00. 文字列の逆順Permalink
文字列”stressed”の文字を逆に（末尾から先頭に向かって）並べた文字列を得よ．

"""
str = "stressed"
str_reversed = str[::-1]
print(str_reversed)


"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
Pythonの文字列の特性を活かしてスライス[0:1]でスマートにまとめた。
・p.10の「2.1 明確な単語を選ぶ」で，
str_reversedと逆順にしたことを示した。

"""