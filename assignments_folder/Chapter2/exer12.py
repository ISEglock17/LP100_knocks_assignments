"""
言語処理100本ノック 第2章課題

12. 1列目をcol1.txtに，2列目をcol2.txtに保存Permalink
各行の1列目だけを抜き出したものをcol1.txtに，2列目だけを抜き出したものをcol2.txtとしてファイルに保存せよ．確認にはcutコマンドを用いよ．

"""
import pandas as pd
import subprocess

# 初期化
filepath = "./assignments_folder/Chapter2/popular-names.txt"    # ファイルパスを指定
df = pd.read_table(filepath, header=None)                     # pandasにおけるDataFrame形式に変更する

col1 = df[0]
col2 = df[1]


with open("./assignments_folder/Chapter2/col1.txt", "w") as f:
    for i in col1:
        f.write(str(i) + "\n")
        
"""
＊ col1.txt 出力結果 ＊
Mary
Anna
Emma
...

"""        

with open("./assignments_folder/Chapter2/col2.txt", "w") as f:
    for i in col2:
        f.write(str(i) + "\n")

"""
＊ col2.txt 出力結果 ＊
F
F
F
...

"""
        
output = subprocess.check_output(["wsl", "cut","-f","1,2", filepath])
print(output.decode('utf-8'))

"""
＊ 出力結果 ＊
...
Margaret        F
Ruth    F
Elizabeth       F
...

"""

"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
withを用いて短くした。

UNIXコマンドでの確認方法
wc -l assignments_folder\Chapter2\popular-names.txt

"""