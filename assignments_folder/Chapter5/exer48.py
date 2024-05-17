"""
言語処理100本ノック 第5章課題

48. 名詞から根へのパスの抽出
文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出せよ． ただし，構文木上のパスは以下の仕様を満たすものとする．

各文節は（表層形の）形態素列で表現する
パスの開始文節から終了文節に至るまで，各文節の表現を” -> “で連結する

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


sentence = sentences[2]
for chunk in sentence.chunks:
    for morph in chunk.morphs:
        if "名詞" in morph.pos:
            path = ["".join(morph.surface for morph in chunk.morphs if morph.pos != "記号")]
            while chunk.dst != -1:
                path.append("".join(morph.surface for morph in sentence.chunks[chunk.dst].morphs if morph.pos != "記号"))
                chunk = sentence.chunks[chunk.dst]
            print("->".join(path))

                        
# 出力結果
"""
人工知能->語->研究分野とも->される
される
じんこうちのう->語->研究分野とも->される
される
AI->エーアイとは->語->研究分野とも->される
エーアイとは->語->研究分野とも->される
計算->という->道具を->用いて->研究する->計算機科学->の->一分野を->指す->語->研究分野とも->される
概念と->道具を->用いて->研究する->計算機科学->の->一分野を->指す->語->研究分野とも->される
コンピュータ->という->道具を->用いて->研究する->計算機科学->の->一分野を->指す->語->研究分野とも->される
道具を->用いて->研究する->計算機科学->の->一分野を->指す->語->研究分野とも->される
知能を->研究する->計算機科学->の->一分野を->指す->語->研究分野とも->される
研究する->計算機科学->の->一分野を->指す->語->研究分野とも->される
計算機科学->の->一分野を->指す->語->研究分野とも->される
される
される
一分野を->指す->語->研究分野とも->される
される
語->研究分野とも->される
言語の->推論->問題解決などの->知的行動を->代わって->行わせる->技術または->研究分野とも->される
理解や->推論->問題解決などの->知的行動を->代わって->行わせる->技術または->研究分野とも->される
推論->問題解決などの->知的行動を->代わって->行わせる->技術または->研究分野とも->される
問題解決などの->知的行動を->代わって->行わせる->技術または->研究分野とも->される
される
知的行動を->代わって->行わせる->技術または->研究分野とも->される
される
人間に->代わって->行わせる->技術または->研究分野とも->される
コンピューターに->行わせる->技術または->研究分野とも->される
技術または->研究分野とも->される
計算機->コンピュータによる->情報処理システムの->実現に関する->研究分野とも->される
される
コンピュータによる->情報処理システムの->実現に関する->研究分野とも->される
知的な->情報処理システムの->実現に関する->研究分野とも->される
情報処理システムの->実現に関する->研究分野とも->される
される
設計や->実現に関する->研究分野とも->される
実現に関する->研究分野とも->される
研究分野とも->される
される
"""
          

"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
リスト内包表記を使って、コンパクトで読みやすいコードを実現しています。
イギリスに関する記事を取得する際、filterとlambdaを使って処理を行っています。これにより、1行で必要なデータを抽出できます。

・p.10の「2.1 明確な単語を選ぶ」で，
変数名やコメントに明確な単語を使っています。例えば、"folderpath"や"filename"は、そのまま読んで意味が理解できる名前です。
"UK_article"という変数名は、イギリスに関する記事を表すため、そのままの名前を使用しています。



クラス定義とコメントの追加

Morph、Chunk、Sentence クラスには、役割と目的を説明するコメントが追加されており、各クラスの理解が容易になっています。


変数の意味の明確化

self.surface、self.base、self.pos などの変数名は、それぞれの役割を明確に表しており、コードの理解が容易です。


初期化とデータ処理の明確化

テキストファイルからのデータの読み込みと、それに基づく文節と形態素の構造化が分かりやすく、整理されています。with open() ブロックでファイルを開き、文節と形態素のリストを構築する処理が行われています。


文節のパスの取得処理

文節のパスを取得する処理がわかりやすく記述されています。特に、while chunk.dst != -1 のループを使用して、係り先が存在する限りパスを追加し続ける仕組みが明確に表現されています。


出力フォーマットとコメント

print("->".join(path)) の部分で、パスを矢印で区切って出力することが説明されています。これにより、出力のフォーマットや内容が明確になっています。


条件の意図の説明

if "名詞" in morph.pos の条件が「名詞を含む形態素に対しての処理」として説明されており、コードの意図が理解しやすいです。

"""