"""
言語処理100本ノック 第3章課題

39. Zipfの法則
単語の出現頻度順位を横軸，その出現頻度を縦軸として，両対数グラフをプロットせよ．

"""

text = []   # テキスト
sentence = []   # 文
 
with open("./assignments_folder/Chapter4/neko.txt.mecab", "r", encoding="utf-8") as f:
    for line in f:
        line = line.split("\t")     # タブ文字で分割
        if len(line) == 2:          # タブ文字の有無の判別
            line[1] = line[1].split(",")    # コンマで分割
            sentence.append({"surface": line[0], "base": line[1][6], "pos": line[1][0], "pos1": line[1][1]})    # 辞書追加
            if line[1][1] == "句点":    # 句点の場合，textにsentenceの一文を追加
                text.append(sentence)
                sentence = []


# 課題35
from collections import Counter

word_list = []

for sentence in text:
    for word in sentence:
        if word["surface"] != "":
            word_list.append(word["surface"])
            
counted_words = Counter(word_list)

# 課題36
import matplotlib.pyplot as plt
import japanize_matplotlib

word_list, count_list = zip(*counted_words.most_common())     # zip(*) でリストを展開(個々の引数として入れるように展開)しながらアンパックできる。https://note.nkmk.me/python-zip-usage-for/


# 課題39

fig, axis = plt.subplots()
x, y = zip(*[(i, v) for i, v in enumerate(count_list, 1)])

print(x)
print(y)
axis.plot(x, y)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("単語の出現頻度順位")
plt.ylabel("出現頻度")
plt.show()

"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
リスト内包表記を使って、コンパクトで読みやすいコードを実現しています。
イギリスに関する記事を取得する際、filterとlambdaを使って処理を行っています。これにより、1行で必要なデータを抽出できます。

・p.10の「2.1 明確な単語を選ぶ」で，
変数名やコメントに明確な単語を使っています。例えば、"folderpath"や"filename"は、そのまま読んで意味が理解できる名前です。
"UK_article"という変数名は、イギリスに関する記事を表すため、そのままの名前を使用しています。

"""