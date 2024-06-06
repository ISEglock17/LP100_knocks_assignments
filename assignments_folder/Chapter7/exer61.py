"""
言語処理100本ノック 第7章課題

61. 単語の類似度
“United States”と”U.S.”のコサイン類似度を計算せよ．

"""
import gensim
file = './assignments_folder/Chapter7/GoogleNews-vectors-negative300.bin.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)

# 課題61
print(model.similarity('United_States', 'U.S.'))

# 出力結果
"""
0.73107743
"""

"""
リーダブルコードの実践

・p.10の「2.1 明確な単語を選ぶ」
・p.171～p.173の「短いコードを書くこと」
・p.15の「ループイテレータ」
・p.76 の「6.6コードの意図を書く
コメントの追加:

コードの目的や各行の動作を説明するコメントがあります。例えば、課題番号や、print()文の直前に説明コメントがあります。
適切な変数名の使用:

fileやmodelなど、変数名がわかりやすく、かつ意味のあるものになっています。
冗長なコメントの削除:

コードが自己説明的であるため、冗長なコメントはありません。
適切な関数名の使用:

model.similarity() のように、関数やメソッドの名前が適切に使われています。これにより、コードを読む人が関数が何を行っているかを容易に理解できます。
コードの整形:

インデントが適切に行われ、コードが整形されています。
表示される出力も整形され、読みやすくなっています。
"""