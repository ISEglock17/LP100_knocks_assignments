"""
言語処理100本ノック 第1章課題

01. 「パタトクカシーー」Permalink
「パタトクカシーー」という文字列の1,3,5,7文字目を取り出して連結した文字列を得よ．

"""
# 初期化
str1 = "パタトクカシーー"
str_extracted = ""
index2extract = [0,2,4,6]   #抽出する文字のindex

#抽出，結合工程
for i in index2extract:
    str_extracted += str1[i]
    
#出力
print(str_extracted)

# ＊出力結果＊
# パトカー

"""
―リーダブルコードの内容で実践したこと―
・p.10の「2.1 明確な単語を選ぶ」で，
str_extractedと抽出した文字列であることを示した。
index2extractと抽出する文字のindexであることを示した。

"""