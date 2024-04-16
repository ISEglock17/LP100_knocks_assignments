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

適切なコメントの使用(p.76 6.6コードの意図を書く): 
コメントは適切に配置され、コードの理解を助ける役割を果たしています。特に、コマンドの意図やUNIXコマンドでの確認方法が明確に記述されています。

変数名の意味の明確化(p.10 2.1 明確な単語を選ぶ): 
変数名が意味的に適切であり、コードの可読性を向上させています。例えば、df_sorted はデータフレームがソートされた状態を表しており、filepath はファイルのパスを示しています。

with文の使用(p.171～p.173 13.4 身近なライブラリに親しむ): 
with文を使用することで、ファイルの読み込みや操作後の適切なクリーンアップを自動化し、コードの簡潔さと可読性を向上させています。

"""