"""
言語処理100本ノック 第3章課題

36. 頻度上位10語
出現頻度が高い10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

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

word_list, count_list = zip(*counted_words.most_common(10))     # zip(*) でリストを展開(個々の引数として入れるように展開)しながらアンパックできる。https://note.nkmk.me/python-zip-usage-for/
print(word_list)
print(count_list)

fig, axis = plt.subplots()
axis.bar(word_list, count_list)
plt.title("単語の出現頻度")
plt.xlabel("単語")
plt.ylabel("出現数")
plt.show()

"""
('の', '。', 'て', '、', 'は', 'に', 'を', 'と', 'が', 'た')
(9194, 7486, 6868, 6772, 6420, 6243, 6071, 5508, 5337, 3988)
"""


"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
リスト内包表記を使って、コンパクトで読みやすいコードを実現しています。
イギリスに関する記事を取得する際、filterとlambdaを使って処理を行っています。これにより、1行で必要なデータを抽出できます。

・p.10の「2.1 明確な単語を選ぶ」で，
変数名やコメントに明確な単語を使っています。例えば、"folderpath"や"filename"は、そのまま読んで意味が理解できる名前です。
"UK_article"という変数名は、イギリスに関する記事を表すため、そのままの名前を使用しています。

変数名の明確化: text や sentence、word_list、counted_words、word_list、count_listなど、変数名が意味を明確に伝えるように命名されています。
コメントの追加: コード内には、各行やブロックの目的や処理内容を説明するコメントがあります。これにより、コードの理解が容易になっています。
可読性の向上: インデントや空白行を使って、コードのブロックを視覚的に整理し、可読性を高めています。
適切なデータ構造の選択: 単語の出現頻度を求める際に、適切なデータ構造を選択しています。具体的には、単語の出現回数をカウントするためにCounterオブジェクトが使用されています。
グラフの表示: 単語の出現頻度上位10語をグラフで表示するために、matplotlibを使って棒グラフが作成されています。

"""