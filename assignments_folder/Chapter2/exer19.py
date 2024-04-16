"""
言語処理100本ノック 第2章課題

19. 各行の1コラム目の文字列の出現頻度を求め，出現頻度の高い順に並べるPermalink
各行の1列目の文字列の出現頻度を求め，その高い順に並べて表示せよ．確認にはcut, uniq, sortコマンドを用いよ．

"""
import subprocess
import pandas as pd
import collections

# ファイルパス指定
filepath = "./assignments_folder/Chapter2/popular-names.txt"    # ファイルパスを指定
df = pd.read_table(filepath, header=None)                     # pandasにおけるDataFrame形式に変更する

names_df = df.iloc[:, 0]    # 名前のみをdfから抜き出す
name_counts = collections.Counter(names_df) # collections.Counterを用いることで，pandasのSeriesから名前の出現回数をカウントしたものを返す。

sorted_counts = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)   # 出現頻度の高い順に並び替える

# 出力
for name, count in sorted_counts:  
    print(f"{name}: {count}")

print("UNIXコマンドで実現すると，")
# UNIX コマンドを定義

# コマンドを実行し、出力を取得
output = subprocess.check_output(["wsl", "cut", "-f", "1", "-d", "\t", filepath, "|", "sort", "|", "uniq", "-c", "|", "sort", "-nr"])

# 出力をデコードして表示
print(output.decode())

"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
withを用いて短くした。

変数や関数の命名(p.10 2.1 明確な単語を選ぶ):
変数名や関数名が具体的でわかりやすいものになっている。例えば、filepath、name_countsなどが挙げられる。
コメントを使って変数や処理の目的を説明している。

インデントとスペースの利用:
インデントが正確で揃っており、コードブロックの区切りが明確です。
演算子の周りにスペースを入れ、読みやすさを向上させています。

コードの短縮化(p.171～p.173 13.4 身近なライブラリに親しむ):
pandasの機能を使ってデータの読み込みや操作を行い，コードが短くした。
リスト内包表記やラムダ関数を用いて簡潔に処理を記述した。

コメントの利用(p.76 6.6コードの意図を書く):
コードの意図や処理内容を説明するコメントをつけた。



UNIXコマンドでの確認方法
wc -l assignments_folder\Chapter2\popular-names.txt

"""