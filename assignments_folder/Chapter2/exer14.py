"""
言語処理100本ノック 第2章課題

13. col1.txtとcol2.txtをマージPermalink
12で作ったcol1.txtとcol2.txtを結合し，元のファイルの1列目と2列目をタブ区切りで並べたテキストファイルを作成せよ．確認にはpasteコマンドを用いよ．

"""
import pandas as pd
import subprocess

# 初期化
filepath = "./assignments_folder/Chapter2/"    # ファイルパスを指定
df1 = pd.read_table(filepath + "col1.txt", header=None)                     # pandasにおけるDataFrame形式に変更する
df2 = pd.read_table(filepath + "col2.txt", header=None)                     # pandasにおけるDataFrame形式に変更する

df3 = pd.concat([df1, df2], axis=1)

df3.to_csv('./assignments_folder/Chapter2/col1_col2_marged.txt', index=False, header=False, sep='\t')
"""
＊ 出力結果 ＊
Mary	F
Anna	F
Emma	F
...

"""

output=subprocess.check_output(["wsl", "paste", filepath + "col1.txt", filepath + "col2.txt"])
print(output.decode('utf-8'))

"""
＊ 出力結果 ＊
Mary    F
Anna    F
Emma    F
...

"""


"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
withを用いて短くした。

UNIXコマンドでの確認方法
wc -l assignments_folder\Chapter2\popular-names.txt

"""