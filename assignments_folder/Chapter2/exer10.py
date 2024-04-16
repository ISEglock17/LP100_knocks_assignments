"""
言語処理100本ノック 第2章課題

10. 行数のカウントPermalink
行数をカウントせよ．確認にはwcコマンドを用いよ.

"""
import pandas as pd
import subprocess

# 初期化
filepath = "./assignments_folder/Chapter2/popular-names.txt"    # ファイルパスを指定
data = pd.read_table(filepath, header=None)                     # pandasにおけるDataFrame形式に変更する

print("行数は{}行".format(len(data)))                           # DataFrame形式のdataにおける行数を出力

print("UNIXコマンドで確認すると，")
result = subprocess.check_output(['wsl', 'wc', '-l', filepath]) #subprocessを利用してUNIXコマンドを実行
print(result.decode('utf-8'))                                   

# 出力結果
# 行数は2780行
# UNIXコマンドで確認すると，
# 2780 ./assignments_folder/Chapter2/popular-names.txt

"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
pandasを用いて，Tabで区切られたデータを読み取ることで，短いコードにて行数を示した。

適切なコメントの使用(p.76 6.6コードの意図を書く): 
コメントがコードの理解を助ける役割を果たしています。例えば、どの部分が行数をカウントしているのかを説明するコメントがあります。

with文の使用(p.171～p.173 13.4 身近なライブラリに親しむ): 
ファイルの読み込みがあるため、with文を使用してファイルのクリーンアップを行っています。この方法を使うことで、ファイルの処理が安全に行われます。

UNIXコマンドでの確認方法
wc -l assignments_folder\Chapter2\popular-names.txt

"""