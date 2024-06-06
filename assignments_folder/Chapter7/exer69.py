"""
言語処理100本ノック 第7章課題

69. t-SNEによる可視化
ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ．

"""
import csv
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 出力ファイルのパス
inputfile = './assignments_folder/Chapter7/country_word2vec.csv'

# 国名とベクトルを格納するリスト
country_list = []
country_vec_list = []

# CSVファイルの読み込み
with open(inputfile, 'r', encoding='utf-8') as f_in:
    reader = csv.reader(f_in)
    next(reader)  # ヘッダーをスキップ
    for row in reader:
        country = row[0]
        country_vec_str = row[1]
        country_vec = np.fromstring(country_vec_str, sep=' ')  # ベクトルを文字列から数値に変換
        country_list.append(country)
        country_vec_list.append(country_vec)

# K-meansクラスタリングの実行
kmeans = KMeans(n_clusters=5)
kmeans.fit(country_vec_list)

# TSNEによる次元削減
tsne = TSNE(n_components=2, random_state=64)
X_reduced = tsne.fit_transform(np.array(country_vec_list))

# プロットの作成
plt.figure(figsize=(10, 10))
for x, country, color in zip(X_reduced, country_list, kmeans.labels_):
    plt.text(x[0], x[1], country, color='C{}'.format(color))
plt.xlim([-12, 15])
plt.ylim([-15, 15])
plt.savefig('./assignments_folder/Chapter7/fig69.png')
plt.show()


"""
リーダブルコードの実践

ｖ・p.10の「2.1 明確な単語を選ぶ」
・p.171～p.173の「短いコードを書くこと」
・p.15の「ループイテレータ」
・p.76 の「6.6コードの意図を書く」
適切なファイルパス: ファイルの場所を指定する際に、適切な相対パスが使用されています。
適切な変数名: 変数名は意味がありわかりやすいものになっています。例えば、country_list、country_vec_listなどが挙げられます。
t-SNEの実行: sklearnのTSNEクラスが使用され、t-SNEによる次元削減が行われます。
プロットの作成: t-SNEによって次元削減されたデータがプロットされ、各点に国名が表示され、クラスタリングの結果によって色分けされます。
"""