"""
言語処理100本ノック 第3章課題

37. 「猫」と共起頻度の高い上位10語
「猫」とよく共起する（共起頻度が高い）10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

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
# 課題37
from collections import Counter

word_list = []

for sentence in text:
    if any(word["base"] == "猫" for word in sentence):  # 文中に基本形が「猫」の単語が含まれているかどうか。  
        word_list += [word["surface"] for word in sentence]
            
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
('の', 'は', '', '、', 'に', '猫', 'を', 'て', '。', 'と')
(397, 284, 270, 265, 251, 248, 240, 236, 219, 210)
"""


"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
リスト内包表記を使って、コンパクトで読みやすいコードを実現しています。
イギリスに関する記事を取得する際、filterとlambdaを使って処理を行っています。これにより、1行で必要なデータを抽出できます。

・p.10の「2.1 明確な単語を選ぶ」で，
変数名やコメントに明確な単語を使っています。例えば、"folderpath"や"filename"は、そのまま読んで意味が理解できる名前です。
"UK_article"という変数名は、イギリスに関する記事を表すため、そのままの名前を使用しています。

"""