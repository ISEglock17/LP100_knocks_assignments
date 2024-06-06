"""
言語処理100本ノック 第7章課題

64. アナロジーデータでの実験
単語アナロジーの評価データをダウンロードし，vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，そのベクトルと類似度が最も高い単語と，その類似度を求めよ．求めた単語と類似度は，各事例の末尾に追記せよ．

"""
import gensim
file = './assignments_folder/Chapter7/GoogleNews-vectors-negative300.bin.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)

from tqdm import tqdm   # 進捗率表示用
inputfile = './assignments_folder/Chapter7/questions-words.txt'
outputfile = './assignments_folder/Chapter7/questions-words_similarity.txt'

# 進捗率計算用処理
total = sum([1 for _ in open(inputfile)])
   
# 出力部
with open(inputfile, 'r', encoding='utf-8') as f_in, open(outputfile, 'w', encoding='utf-8') as f_out:
    for line in tqdm(f_in, total=total):
        if line[0] == ':':  # 行がカテゴリ名を示している場合
            f_out.write(line)
        else:
            words = line.rstrip().split()
            similar_word, similarity = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=1)[0]
            f_out.write('{}\t{}\t{}\n'.format(line.rstrip(), similar_word, similarity))

# 出力結果
""" /questions-words_similarity.txt
: capital-common-countries
Athens Greece Baghdad Iraq	Iraqi	0.635187029838562
Athens Greece Bangkok Thailand	Thailand	0.7137669324874878
Athens Greece Beijing China	China	0.7235778570175171
Athens Greece Berlin Germany	Germany	0.6734622716903687
Athens Greece Bern Switzerland	Switzerland	0.4919748306274414
Athens Greece Cairo Egypt	Egypt	0.7527808547019958
Athens Greece Canberra Australia	Australia	0.583732545375824
Athens Greece Hanoi Vietnam	Viet_Nam	0.6276341676712036
Athens Greece Havana Cuba	Cuba	0.6460990905761719
...
...
...
write writes speak speaks	speaks	0.654321551322937
write writes swim swims	swims	0.6643378734588623
write writes talk talks	talked	0.5447186231613159
write writes think thinks	thinks	0.6177733540534973
write writes vanish vanishes	disappear	0.6002705693244934
write writes walk walks	walks	0.5534339547157288
write writes work works	works	0.538760781288147
"""

"""
― 参考になるサイト
https://qiita.com/kuroitu/items/f18acf87269f4267e8c1
上記のサイトで記述されているtqdmを用いることで，進捗率をプログレスバーとして表示することができる。

リーダブルコードの実践

・p.10の「2.1 明確な単語を選ぶ」
・p.171～p.173の「短いコードを書くこと」
・p.15の「ループイテレータ」
・p.76 の「6.6コードの意図を書く

コメントの追加: コードの目的や各行の動作を説明するコメントがあります。進捗率計算用のコメントや出力部に関する説明があります。
適切な変数名の使用: fileやmodel、inputfile、outputfileなど、変数名が適切でわかりやすいものになっています。
出力の整形: 出力結果が適切に整形されており、見やすくなっています。各事例の末尾に類似度と単語が追記されています。
進捗率表示の利用: tqdmを使用して進捗率をプログレスバーとして表示しています。これにより処理の進行状況が可視化され、ユーザーにわかりやすくなっています。
"""