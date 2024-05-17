"""
言語処理100本ノック 第3章課題

45. 動詞の格パターンの抽出Permalink
今回用いている文章をコーパスと見なし，日本語の述語が取りうる格を調査したい． 動詞を述語，動詞に係っている文節の助詞を格と考え，述語と格をタブ区切り形式で出力せよ． ただし，出力は以下の仕様を満たすようにせよ．

動詞を含む文節において，最左の動詞の基本形を述語とする
述語に係る助詞を格とする
述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
このプログラムの出力をファイルに保存し，以下の事項をUNIXコマンドを用いて確認せよ．

コーパス中で頻出する述語と格パターンの組み合わせ
「行う」「なる」「与える」という動詞の格パターン（コーパス中で出現頻度の高い順に並べよ）

MeCabの基本として，次のフォーマットにしたがって，記録される。
表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
        [0],　　[1]　  ,　　[2]　  ,　　[3]　 ,[4]　 ,[5]　,[6] ,[7] ,[8]
"""



class Morph:
    """
        形態素を示すクラス
    """
    def __init__(self, line):
        self.surface, other = line.split("\t")
        other = other.split(",")
        
        self.base = other[6]
        self.pos = other[0]
        self.pos1 = other[1]
        
# 課題41
class Chunk:
    """
        文節を表すクラス
    """
    def __init__(self, morphs, dst, chunk_id):
        self.morphs = morphs    # Morphリスト
        self.dst = dst          # 係り先文節インデックス番号
        self.srcs = []          # 係り元文節インデックス番号のリスト
        self.chunk_id = chunk_id

class Sentence:
    """
        文章を表すクラス
    """
    def __init__(self, chunks):
        self.chunks = chunks    # Chunkリスト
        
        for i, chunk in enumerate(self.chunks):
            if chunk.dst not in [None, -1]: # Noneや-1を除く
                self.chunks[chunk.dst].srcs.append(i)   # 係り元リストにインデックスを追加
 

 
# 初期化
sentences = []  # ※前問とはリスト名を変えた
morphs = []
chunks = []
chunk_id = 0

# メインルーチン
with open("./assignments_folder/Chapter5/ai.ja.txt.parsed", encoding="utf-8") as f:
    for line in f:
        if line[0] == "*":  # *の場合スキップする
            if morphs:
                chunks.append(Chunk(morphs, dst, chunk_id))
                chunk_id += 1
                morphs = []
            dst = int(line.split()[2].replace("D", ""))
        elif line != "EOS\n":   # 文末でない場合Morphのインスタンスを生成
            morphs.append(Morph(line))
        else:  # 文末の場合
            chunks.append(Chunk(morphs, dst, chunk_id))
            sentences.append(Sentence(chunks))
            
            morphs = []
            chunks = []
            dst = None
            chunk_id = 0

with open("./assignments_folder/Chapter5/result45.txt", "w", encoding="utf-8") as f:
    for i in range(len(sentences)):
        for chunk in sentences[i].chunks:
            for morph in chunk.morphs:
                if morph.pos == "動詞": 
                    particles = []  # 助詞
                    for src in chunk.srcs:
                        particles.append([morph.surface for morph in sentences[i].chunks[src].morphs if morph.pos == "助詞"])
                    if len(particles) > 1:
                        particles = set(particles)  # 重複を除去
                        particles = sorted(list(particles)) # 辞書順にソート
                        form = " ".join(particles)  # スペース区切りで文字列に変換
                        print(f"{morph.base}\t{form}", file=f)  # ファイルに出力
                        
# 出力結果
"""
する	て を
代わる	に を
行う	て に
せる	て に
する	と も
れる	と も
述べる	で に の は
いる	で に の は
する	で を
する	て を
する	て を
ある	が は
する	で に により
れる	で に により
する	と を
使う	で でも は
れる	で でも は
いる	で でも は
ある	て も
出す	が に
...
"""

# UNIXコマンドでの確認
"""
コーパス中で頻出する述語と格パターンの組み合わせ
cat ./result45.txt | sort | uniq -c | sort -nr | head -n 5

「行う」「なる」「与える」という動詞の格パターン（コーパス中で出現頻度の高い順に並べよ）
cat ./result45.txt |grep "行う" | sort |uniq -c | sort -nr |head -n 5
cat ./result45.txt |grep "なる" | sort |uniq -c | sort -nr |head -n 5
cat ./result45.txt |grep "与える" | sort |uniq -c | sort -nr |head -n 5
"""

# UNIX 出力結果
"""
     12 する    は を
     10 する    に を
      9 する    で を
      8 する    が に
      6 する    と は


      1 行う    まで を
      1 行う    は を をめぐって
      1 行う    は を
      1 行う    に を
      1 行う    に まで を
      

      4 なる    に は
      3 なる    が と
      1 異なる  が で
      1 なる    は
      1 なる    に は も
      
      
      1 与える  に は を
      1 与える  が に
      1 与える  が など に
"""


"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
リスト内包表記を使って、コンパクトで読みやすいコードを実現しています。
イギリスに関する記事を取得する際、filterとlambdaを使って処理を行っています。これにより、1行で必要なデータを抽出できます。

・p.10の「2.1 明確な単語を選ぶ」で，
変数名やコメントに明確な単語を使っています。例えば、"folderpath"や"filename"は、そのまま読んで意味が理解できる名前です。
"UK_article"という変数名は、イギリスに関する記事を表すため、そのままの名前を使用しています。



クラスとメソッドの説明的なコメント

Morph、Chunk、Sentence の各クラスには、役割や目的を説明するコメントが付いており、他の人が理解しやすい構造になっている。



変数名の意味の明確化

各変数の名前は、その役割や内容を反映している。例えば、self.surface、self.base、self.pos など、わかりやすい命名が行われている。



初期化とインスタンスの管理

Sentence クラスの初期化では、与えられた文節リストをそのまま属性として保持し、さらに係り元リストを構築する処理が行われている。



ファイルの読み込みとデータ処理の分離

with open() ブロックでファイルを開き、その後の処理では読み込んだデータに対する処理が行われている。これにより、ファイルの読み書きとデータの操作が明確に分離されている。



条件分岐と処理の意図の説明

if 文や for ループには、条件の意図を説明するコメントがあり、特定の条件下で処理が行われることが明確になっている。例えば、if morph.pos == "動詞": という条件が「動詞の場合に処理を行う」という意図であることが分かる。



出力のフォーマットとコメント

print() 文での出力部分には、出力のフォーマットやその内容がどのような情報を表しているのかを説明するコメントがある。例えば、print(f"{morph.base}\t{form}", file=f) の部分で、動詞とその後に係る助詞の組み合わせをファイルに出力していることが分かる。
"""