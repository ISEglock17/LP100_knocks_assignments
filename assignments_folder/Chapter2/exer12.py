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

適切なコメントの使用(p.76 6.6コードの意図を書く): 
コメントがコードの理解を助ける役割を果たしています。例えば、どの部分が1列目と2列目を抜き出しているのかを説明するコメントがあります。

変数名の意味の明確化(p.10 2.1 明確な単語を選ぶ): 
変数名は適切に選ばれており、プログラムの理解を助けています。col1 と col2 という名前はそれぞれの列を示しています。

with文の使用(p.171～p.173 13.4 身近なライブラリに親しむ): 
ファイルの書き込みがあるため、with文を使用してファイルのクリーンアップを行っています。この方法を使うことで、ファイルの処理が安全に行われます。
"""