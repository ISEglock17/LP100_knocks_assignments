"""
言語処理100本ノック 第7章課題

65. アナロジータスクでの正解率
64の実行結果を用い，意味的アナロジー（semantic analogy）と文法的アナロジー（syntactic analogy）の正解率を測定せよ．

"""
import re
from tqdm import tqdm   # 進捗率表示用
inputfile = './assignments_folder/Chapter7/questions-words_similarity.txt'

# 進捗率計算用処理
total = sum([1 for _ in open(inputfile)])

# 変数定義
semantic_count = 0
semantic_true = 0
syntactic_count = 0
syntactic_true = 0

# 正解率算出
with open(inputfile, 'r', encoding='utf-8') as f_in:
    for line in tqdm(f_in, total=total):
        if line[0] == ':':  # 行がカテゴリ名を示している場合
            category = re.sub(r'[ :\n]', '', line)            
        else:
            target = line.strip().split('\t')[1]
            ans = line.strip().split('\t')[0].split()[-1]
            
            if "gram" in category:  # "gram"がカテゴリ名に含まれているときは，文法的アナロジー(syntactic)として判断
                syntactic_count += 1
                if ans == target:
                    syntactic_true += 1
            else:
                semantic_count += 1
                if ans == target:
                    semantic_true += 1

# 出力
print('意味的アナロジーの正解率: {}'.format(semantic_true / semantic_count))
print('文法的アナロジーの正解率: {}'.format(syntactic_true / syntactic_count))

# 出力結果
"""
意味的アナロジーの正解率: 0.7308602999210734
文法的アナロジーの正解率: 0.7400468384074942
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
コメントの追加: コードの目的や各行の動作を説明するコメントがあります。例えば、カテゴリ名が "gram" を含むかどうかで文法的アナロジーかどうかを判断する部分に対するコメントがあります。
適切な変数名の使用: semantic_count, semantic_true, syntactic_count, syntactic_trueなど、適切な変数名が使われています。これにより、コードの理解が容易になります。
出力の整形: 正解率が計算され、意味的アナロジーと文法的アナロジーの正解率がわかりやすく表示されています。
進捗率表示の利用: tqdmを使用して進捗率をプログレスバーとして表示しています。これにより、処理の進行状況が視覚的にわかりやすくなっています。

"""