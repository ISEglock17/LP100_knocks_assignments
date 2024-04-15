"""
言語処理100本ノック 第2章課題

14. 先頭からN行を出力Permalink
自然数Nをコマンドライン引数などの手段で受け取り，入力のうち先頭のN行だけを表示せよ．確認にはheadコマンドを用いよ．

"""
import pandas as pd
import subprocess
import sys

filepath = "./assignments_folder/Chapter2/popular-names.txt"

if __name__ == '__main__':      # 本プログラムがインポートされた際には実行されないようにするための記述
    args = sys.argv
    if 2 <= len(args): 
        with open(filepath, "r") as f:
            data = f.readlines()
            for i in range(int(args[1])):
                print(data[i], end="")
    else:
        print('引数に自然数Nを入れてください。')  

print("UNIXコマンドを実行した場合には，")
output = subprocess.check_output(["wsl", "head","-n", str(3), filepath])
print(output.decode('utf-8'))    

"""
＊ 出力結果 ＊
Mary    F       7065    1880
Anna    F       2604    1880
Emma    F       2003    1880
UNIXコマンドを実行した場合には，
Mary    F       7065    1880
Anna    F       2604    1880
Emma    F       2003    1880
"""

"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
withを用いて短くした。

UNIXコマンドでの確認方法
wc -l assignments_folder\Chapter2\popular-names.txt

"""