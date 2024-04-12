"""
言語処理100本ノック 第1章課題

02. 「パトカー」＋「タクシー」＝「パタトクカシーー」Permalink
「パトカー」＋「タクシー」の文字を先頭から交互に連結して文字列「パタトクカシーー」を得よ．

"""
# 初期化
str1 = "パトカー"
str2 = "タクシー"
str_combined_alternately = ""   #交互に結合された文字列

#文字列操作: 交互に結合
for si in range(len(str1)):
    for flag in [True, False]:
        str_combined_alternately += str1[si] if flag else str2[si]

#出力
print(str_combined_alternately)

# ＊出力結果＊
# パタトクカシーー

"""
―リーダブルコードの内容で実践したこと―
・p.10の「2.1 明確な単語を選ぶ」で，
str_combined_alternatelyと交互に結合された文字列であることを示した。

・p.15の「ループイテレータ」で，
イテレータが複数あるため，変数名をsiとしてstrのindexを示していることを示した。

"""