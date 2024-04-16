"""
言語処理100本ノック 第2章課題

15. 末尾のN行を出力Permalink
自然数Nをコマンドライン引数などの手段で受け取り，入力のうち末尾のN行だけを表示せよ．確認にはtailコマンドを用いよ．

"""
import subprocess
import sys

filepath = "./assignments_folder/Chapter2/popular-names.txt"

if __name__ == '__main__':      # 本プログラムがインポートされた際には実行されないようにするための記述
    args = sys.argv
    if 2 <= len(args): 
        with open(filepath, "r") as f:
            data = f.readlines()            
            print("".join(data[- int(args[1]):]))       
            
    else:
        print('引数に自然数Nを入れてください。')  
        
"""
＊ 解説 ＊
dataには1行ずつ読み取ったデータがリストで保存されているため，これをコマンドライン引数で得たargs[1]分だけ末尾から読み取って文字列を結合すればよい。
このとき，dataの各行末尾には\nがあるためそのまま結合しても改行される。
"""

print("UNIXコマンドを実行した場合には，")
output = subprocess.check_output(["wsl", "tail","-n", str(3), filepath])
print(output.decode('utf-8'))    

"""
＊ 出力結果 ＊
Lucas   M       12585   2018
Mason   M       12435   2018
Logan   M       12352   2018

UNIXコマンドを実行した場合には，
Lucas   M       12585   2018
Mason   M       12435   2018
Logan   M       12352   2018

"""

"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
withを用いて短くした。

"""