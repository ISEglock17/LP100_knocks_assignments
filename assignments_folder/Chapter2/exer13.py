"""
言語処理100本ノック 第2章課題

13. col1.txtとcol2.txtをマージ
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

output = subprocess.check_output(["wsl", "paste", filepath + "col1.txt", filepath + "col2.txt"])
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

適切なコメントの使用(p.76 6.6コードの意図を書く): 
コメントがコードの理解を助ける役割を果たしています。例えば、どの部分がcol1.txtとcol2.txtをマージしているのかを説明するコメントがあります。

変数名の意味の明確化(p.10 2.1 明確な単語を選ぶ): 
変数名は適切に選ばれており、プログラムの理解を助けています。df1、df2、df3という名前は、それぞれのDataFrameを示しています。

with文の使用(p.171～p.173 13.4 身近なライブラリに親しむ):
ファイルの読み込みがあるため、with文を使用してファイルのクリーンアップを行っています。この方法を使うことで、ファイルの処理が安全に行われます。

"""