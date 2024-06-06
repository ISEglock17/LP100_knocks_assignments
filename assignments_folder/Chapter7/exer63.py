"""
言語処理100本ノック 第7章課題

63. 加法構成性によるアナロジー
“Spain”の単語ベクトルから”Madrid”のベクトルを引き，”Athens”のベクトルを足したベクトルを計算し，そのベクトルと類似度の高い10語とその類似度を出力せよ．
"""
import gensim
file = './assignments_folder/Chapter7/GoogleNews-vectors-negative300.bin.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)

top10_similarity = model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'], topn=10)

# 出力部
print('類似度ランキング: 単語\t類似度')
for index, value in enumerate(top10_similarity):
    print('{}: {}\t{}'.format(index + 1, value[0], value[1]))


# 出力結果
"""
類似度ランキング: 単語  類似度
1: Greece       0.6898480653762817
2: Aristeidis_Grigoriadis       0.560684859752655
3: Ioannis_Drymonakos   0.5552908778190613
4: Greeks       0.545068621635437
5: Ioannis_Christou     0.5400862097740173
6: Hrysopiyi_Devetzi    0.5248445272445679
7: Heraklio     0.5207759737968445
8: Athens_Greece        0.516880989074707
9: Lithuania    0.5166865587234497
10: Iraklion    0.5146791338920593
"""

"""
リーダブルコードの実践

・p.10の「2.1 明確な単語を選ぶ」
・p.171～p.173の「短いコードを書くこと」
・p.15の「ループイテレータ」
・p.76 の「6.6コードの意図を書く
コメントの追加: コードの目的や各行の動作を説明するコメントがあります。出力部には、類似度ランキングの説明コメントがあります。
適切な変数名の使用: fileやmodelなど、変数名が適切でわかりやすいものになっています。
出力の整形: 類似度ランキングが適切に整形されており、見やすくなっています。
"""