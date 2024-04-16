"""
言語処理100本ノック 第2章課題

18. 各行を3コラム目の数値の降順にソートPermalink
各行を3コラム目の数値の逆順で整列せよ（注意: 各行の内容は変更せずに並び替えよ）．確認にはsortコマンドを用いよ（この問題はコマンドで実行した時の結果と合わなくてもよい）．

"""
import subprocess
import pandas as pd

# ファイルパス指定
filepath = "./assignments_folder/Chapter2/popular-names.txt"    # ファイルパスを指定
df = pd.read_table(filepath, header=None)                     # pandasにおけるDataFrame形式に変更する

df_sorted = df.sort_values(2, ascending=False)

"""
df_sortedの中身
            0  1      2     3
1340    Linda  F  99689  1947
1360    Linda  F  96211  1948
1350    James  M  94757  1947
1550  Michael  M  92704  1957
1351   Robert  M  91640  1947
...       ... ..    ...   ...
27      Annie  F   1326  1881
28     Bertha  F   1324  1881
8      Bertha  F   1320  1880
29      Alice  F   1308  1881
9       Sarah  F   1288  1880
"""

# 出力
print(df_sorted)

print("UNIXコマンドを用いると，")
output = subprocess.check_output(["wsl", "sort", "-n", "-r", "-k", "3", "-t", "\t", filepath])
print(output.decode())

"""
UNIXコマンドを用いた場合の出力は

Linda   F       99689   1947
Linda   F       96211   1948
James   M       94757   1947
Michael M       92704   1957
...

"""


"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
withを用いて短くした。


UNIXコマンドでの確認方法
wc -l assignments_folder\Chapter2\popular-names.txt

"""