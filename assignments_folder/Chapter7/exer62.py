"""
言語処理100本ノック 第7章課題

62. 類似度の高い単語10件
“United States”とコサイン類似度が高い10語と，その類似度を出力せよ．

"""
import gensim
file = './assignments_folder/Chapter7/GoogleNews-vectors-negative300.bin.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)

top10_similarity = model.most_similar('United_States', topn=10)

# 出力部
print('類似度ランキング: 単語\t類似度')
for index, value in enumerate(top10_similarity):
    print('{}: {}\t{}'.format(index + 1, value[0], value[1]))

# 出力結果
"""
類似度ランキング: 単語  類似度
1: Unites_States        0.7877248525619507
2: Untied_States        0.7541370987892151
3: United_Sates 0.7400724291801453
4: U.S. 0.7310774326324463
5: theUnited_States     0.6404393911361694
6: America      0.6178410053253174
7: UnitedStates 0.6167312264442444
8: Europe       0.6132988929748535
9: countries    0.6044804453849792
10: Canada      0.601906955242157
"""

"""
# 参考になるサイト
https://qiita.com/kenta1984/items/93b64768494f971edf86
gensimを使って，Word2Vecを活用する方法が記されている。

リーダブルコードの実践

・p.10の「2.1 明確な単語を選ぶ」
・p.171～p.173の「短いコードを書くこと」
・p.15の「ループイテレータ」
・p.76 の「6.6コードの意図を書く
コメントの追加: コードの目的や各行の動作を説明するコメントがあります。出力部には、類似度ランキングの説明コメントがあります。
適切な変数名の使用: fileやmodelなど、変数名が適切でわかりやすいものになっています。
出力の整形: 類似度ランキングが適切に整形されており、見やすくなっています。
"""