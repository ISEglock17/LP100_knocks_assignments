"""
    input_file = "./assignments_folder/Chapter6/specified_publishers_articles.csv"より，    
    抽出された事例をランダムに並び替える．
    抽出された事例の80%を学習データ，残りの10%ずつを検証データと評価データに分割し，それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する．
    ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ（このファイルは後に問題70で再利用する）．
    
    
    FORMAT: ID ︙TITLE ︙URL ︙PUBLISHER ︙CATEGORY ︙STORY ︙HOSTNAME ︙TIMESTAMP

    ＊詳細内容＊
    ID 数値ID
    TITLE ニュースのタイトル 
    URL URL
    PUBLISHER 出版社名
    CATEGORY ニュースのカテゴリー（b = ビジネス、t = 科学技術、e = エンターテインメント、m = 健康）
    STORY 同じストーリーに関するニュースを含むクラスターの英数字ID
    HOSTNAME URLホスト名
    TIMESTAMP 1970年1月1日00:00:00 GMTからのミリ秒数で表した、ニュースが発表されたおおよその時間。
"""

import random

# ファイル名
input_file = "./assignments_folder/Chapter6/specified_publishers_articles.csv"
train_file = "./assignments_folder/Chapter6/train.txt"
valid_file = "./assignments_folder/Chapter6/valid.txt"
test_file = "./assignments_folder/Chapter6/test.txt"

# 事例を格納するリスト
instances = []

# ファイルを読み込み、ランダムに並び替える
with open(input_file, 'r', encoding='utf-8') as file:
    instances = file.readlines()

random.shuffle(instances)

# データの数を取得
total_instances = len(instances)
train_size = int(0.8 * total_instances)
valid_size = int(0.1 * total_instances)

# データを学習データ、検証データ、評価データに分割
train_data = instances[:train_size]
valid_data = instances[train_size:train_size+valid_size]
test_data = instances[train_size+valid_size:]

# ファイルにデータを書き込む
def write_data(filename, data):
    with open(filename, 'w', encoding='utf-8') as file:
        for instance in data:
            article_data_list = instance.split('\t')  # CATEGORYとTITLEのみを取得
            title = article_data_list[1]
            category = article_data_list[4]
            file.write(f"{category}\t{title}\n")  # タブ区切り形式で書き込み

write_data(train_file, train_data)
write_data(valid_file, valid_data)
write_data(test_file, test_data)

print("データを学習データ、検証データ、評価データに分割し、ファイルに保存しました。")
