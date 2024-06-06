"""
言語処理100本ノック 第7章課題

67. k-meansクラスタリング
国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ．

国名に関する単語ベクトル生成をexer67-1.py，k-meansクラスタリングをexer67-2.pyで行う。

"""
import csv
from sklearn.cluster import KMeans
import numpy as np


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

# テスト出力
print(f"Total country_list: {len(country_list)}")
print(f"First country: {country_list[0]}, Vector: {country_vec_list[0]}")

# K-meansクラスタリングの実行
kmeans = KMeans(n_clusters=5)
kmeans.fit(country_vec_list)

# 各クラスタに属する国名を出力
for i in range(5):
    cluster = np.where(kmeans.labels_ == i)[0]
    print('クラスタ', i)
    print(', '.join([country_list[k] for k in cluster]))

# 出力結果
"""
クラスタ 0
Zimbabwe, Zambia, Senegal, Rwanda, Mali, Namibia, Uganda, Tuvalu, Sudan, Madagascar, Liberia, Ghana, Guinea, Morocco, Botswana, Nigeria, Gambia, Eritrea, Fiji, Somalia, Mauritania, Mozambique, Niger, Malawi, Kenya, Algeria, Gabon, Angola, Tunisia, Burundi
クラスタ 1
Brazil, Chile, Honduras, Argentina, Philippines, Ecuador, Cuba, Belize, Suriname, Peru, Uruguay, Mexico, Bahamas, Colombia, Venezuela, Samoa, Guyana, Jamaica, Nicaragua, Dominica
クラスタ 2
Portugal, Canada, Greenland, Japan, Germany, Sweden, Finland, England, Austria, Liechtenstein, USA, France, Norway, Spain, Denmark, Italy, Ireland, Switzerland, Belgium, Iceland, Netherlands, Europe, Australia
クラスタ 3
Romania, Estonia, Slovakia, Slovenia, Moldova, Malta, Armenia, Kazakhstan, Poland, Lithuania, Russia, Azerbaijan, Albania, Hungary, Montenegro, Cyprus, Ukraine, Georgia, Belarus, Bulgaria, Latvia, Macedonia, Serbia, Croatia, Greece, Turkey
クラスタ 4
Turkmenistan, Oman, Malaysia, Syria, Vietnam, Libya, Egypt, Uzbekistan, Cambodia, Iraq, Kyrgyzstan, Indonesia, India, Korea, Iran, Thailand, Bangladesh, Qatar, China, Pakistan, Bhutan, Nepal, Afghanistan, Laos, Jordan, Lebanon, Bahrain, Taiwan, Israel, Tajikistan
"""

"""
リーダブルコードの実践

・p.10の「2.1 明確な単語を選ぶ」
・p.171～p.173の「短いコードを書くこと」
・p.15の「ループイテレータ」
・p.76 の「6.6コードの意図を書く

適切なファイルパス: ファイルの場所を指定する際に、適切な相対パスが使用されています。
適切な変数名: 変数名は意味がありわかりやすいものになっています。たとえば、country_list、country_vec_listなどが挙げられます。
テスト出力の追加: デバッグのためにテスト出力が追加されており、変数の内容やデータの形式が正しいか確認できます。
K-meansクラスタリングの実行: sklearnのKMeansクラスが使用されています。
クラスタリング結果の出力: 各クラスタに属する国名が出力されており、クラスタリングの結果が分かりやすくなっています。

"""