"""
言語処理100本ノック 第7章課題

68. Ward法によるクラスタリング
国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．さらに，クラスタリング結果をデンドログラムとして可視化せよ．

"""
import csv
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

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

# 階層的クラスタリングの実行
linkage_result = linkage(country_vec_list, method='ward')

# デンドログラムのプロット
plt.figure(figsize=(16, 9))
dendrogram(linkage_result, labels=country_list)
plt.savefig('./assignments_folder/Chapter7/fig68.png')
plt.show()

"""
リーダブルコードの実践

・p.10の「2.1 明確な単語を選ぶ」
・p.171～p.173の「短いコードを書くこと」
・p.15の「ループイテレータ」
・p.76 の「6.6コードの意図を書く
適切なファイルパス: ファイルの場所を指定する際に、適切な相対パスが使用されています。
適切な変数名: 変数名は意味がありわかりやすいものになっています。例えば、country_list、country_vec_listなどが挙げられます。
階層的クラスタリングの実行: scipyのlinkage関数が使用され、Ward法による階層型クラスタリングが行われます。
デンドログラムのプロット: クラスタリングの結果をデンドログラムとして視覚化し、matplotlibを用いて可視化されています。

"""