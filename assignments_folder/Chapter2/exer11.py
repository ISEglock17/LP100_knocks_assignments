"""
言語処理100本ノック 第2章課題

11. タブをスペースに置換Permalink
タブ1文字につきスペース1文字に置換せよ．確認にはsedコマンド，trコマンド，もしくはexpandコマンドを用いよ．

"""
import subprocess

# 初期化
filepath = "./assignments_folder/Chapter2/popular-names.txt"    # ファイルパスを指定

with open("./assignments_folder/Chapter2/exer11_output.txt", "w") as g:
    with open(filepath, "r") as f:
        for data in f:
            # print(data.strip().replace("\t"," "))               # デバッグ用出力
            g.write(data.strip().replace("\t"," ") + "\n")      # ファイル出力
        
"""
＊ 出力結果 ＊

Mary F 7065 1880
Anna F 2604 1880
Emma F 2003 1880
...

Mason M 12435 2018
Logan M 12352 2018

"""

output = subprocess.check_output(["wsl", "sed", "-e", "s/\t/ /g", filepath])
print(output.decode('utf-8'))
    
"""
＊ 出力結果 ＊
Mary F 7065 1880
Anna F 2604 1880
Emma F 2003 1880
...


"""

"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
withを用いて短くした。

適切なコメントの使用(p.76 6.6コードの意図を書く): 
コメントがコードの理解を助ける役割を果たしています。例えば、どの部分がタブをスペースに置換しているのかを説明するコメントがあります。

with文の使用(p.171～p.173 13.4 身近なライブラリに親しむ): 
ファイルの読み込みと書き込みがあるため、with文を使用してファイルのクリーンアップを行っています。この方法を使うことで、ファイルの処理が安全に行われます。

"""