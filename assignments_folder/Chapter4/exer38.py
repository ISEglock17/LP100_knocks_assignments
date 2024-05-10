"""
言語処理100本ノック 第3章課題

38. ヒストグラム
単語の出現頻度のヒストグラムを描け．ただし，横軸は出現頻度を表し，1から単語の出現頻度の最大値までの線形目盛とする．縦軸はx軸で示される出現頻度となった単語の異なり数（種類数）である．

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
print(word_list)
print(count_list)


# 課題38
fig, axis = plt.subplots()
axis.hist(count_list, bins=100)
plt.xlabel("出現頻度")
plt.ylabel("種類数")
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