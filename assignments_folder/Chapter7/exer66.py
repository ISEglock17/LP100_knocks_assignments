"""
言語処理100本ノック 第7章課題

66. WordSimilarity-353での評価
The WordSimilarity-353 Test Collectionの評価データをダウンロードし，単語ベクトルにより計算される類似度のランキングと，人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ．

評価データの情報は次のリンク先に書かれている。
https://gabrilovich-com.translate.goog/resources/data/wordsim353/wordsim353.html?_x_tr_sl=en&_x_tr_tl=ja&_x_tr_hl=ja&_x_tr_pto=sc

"""
import gensim
from tqdm import tqdm
from scipy.stats import spearmanr

file = './assignments_folder/Chapter7/GoogleNews-vectors-negative300.bin.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)

file2 = './assignments_folder/Chapter7/wordsim353/combined.csv'

# 進捗率計算用処理
total = sum([1 for _ in open(file2)])

# 変数定義
human = []
word2vec = []

with open(file2, 'r', encoding='utf-8') as f:
    next(f)
    for list in tqdm(f, total=total):
        cols = list.rstrip().split(',')
        human.append(float(cols[2]))
        word2vec.append(model.similarity(cols[0], cols[1]))

correlation, pvalue = spearmanr(human, word2vec)

# 出力
print('スピアマン相関係数: {}'.format(correlation))

# 出力結果
"""
スピアマン相関係数: 0.7000166486272194
"""

"""
リーダブルコードの実践

・p.10の「2.1 明確な単語を選ぶ」
・p.171～p.173の「短いコードを書くこと」
・p.15の「ループイテレータ」
・p.76 の「6.6コードの意図を書く

コメントの追加: コードの目的や各行の動作を説明するコメントがあります。例えば、進捗率計算用の処理やスピアマン相関係数を計算する部分に対するコメントがあります。
適切な変数名の使用: human、word2vecなど、適切な変数名が使われています。これにより、コードの理解が容易になります。
出力の整形: スピアマン相関係数が計算され、わかりやすく表示されています。
"""