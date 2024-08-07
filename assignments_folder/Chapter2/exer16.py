"""
言語処理100本ノック 第2章課題

16. ファイルをN分割するPermalink
自然数Nをコマンドライン引数などの手段で受け取り，入力のファイルを行単位でN分割せよ．同様の処理をsplitコマンドで実現せよ．

"""
import subprocess
import sys

filepath = "./assignments_folder/Chapter2/"
readfile = "popular-names.txt"
writefile = "exer16_output/exer16_"

if __name__ == '__main__':      # 本プログラムがインポートされた際には実行されないようにするための記述
    args = sys.argv
    if 2 <= len(args):  # 引数が入力されていることを確かめる。
        with open(filepath + readfile, "r") as f:   # 入力ファイルのデータを読み取る。
            data = f.readlines()
            
        for file_num in range(int(args[1])):
            with open(filepath + writefile + str(file_num) + ".txt", "w") as g:     # 出力ファイルを指定する。
                for di in range(len(data) // int(args[1]) * file_num, len(data) // int(args[1]) * (file_num + 1)):   # 分割ファイルにおいて参照するindexで回す　data_indexよりdiとした。
                    if file_num == int(args[1]) - 1 and len(data) % int(args[1]) != 0 and di == len(data) // int(args[1]) * (file_num + 1) - 1:     # 割り切れなかった分は最後の分割ファイルの末尾に追加するようにする。
                        for i in range(len(data) % int(args[1]) + 1):          # 割り切れなかった分のindexを回す。
                            g.write("{}".format(data[di + i]))
                    else:
                        g.write("{}".format(data[di]))
    else:
        print('引数に自然数Nを入れてください。')
    
    print("UNIXコマンドを実行した場合には，")
    output = subprocess.check_output(["wsl", "split", "-n", "3", "./assignments_folder/Chapter2/popular-names.txt", "./assignments_folder/Chapter2/exer16_output/exer16_output"])
    print(output.decode("utf-8"))
    
"""
＊ 出力結果 ＊
7分割すると，
- exer16_output/exer16_0.txt -
Mary	F	7065	1880
Anna	F	2604	1880
Emma	F	2003	1880
...

- exer16_output/exer16_6.txt -
...
Lucas	M	12585	2018
Mason	M	12435	2018
Logan	M	12352	2018

"""

"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
withを用いて短くした。

・変数名の意味の明確化(p.10 2.1 明確な単語を選ぶ)
data_indexより，diとすることで，forの要素を分かりやすくした。

＊ 注意点 ＊
入力データの行数2780に対して，3分割などと指定した場合には，割り切れないため，割り切れない分をどのように処理するかがポイント。

"""