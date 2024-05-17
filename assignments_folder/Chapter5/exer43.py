"""
言語処理100本ノック 第3章課題

43. 名詞を含む文節が動詞を含む文節に係るものを抽出
名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．ただし，句読点などの記号は出力しないようにせよ．

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
        
# 出力
for chunk in sentences[2].chunks:  
    if chunk.dst in [None, -1]:
        continue
    
    # チャンク内の形態素の表層形を結合（記号を除く）
    surf = "".join([morph.surface for morph in chunk.morphs if morph.pos != "記号"])
    
    # 係り先チャンク
    dst_chunk = sentences[2].chunks[int(chunk.dst)]
    next_surf = "".join([morph.surface for morph in dst_chunk.morphs if morph.pos != "記号"])
    
    # 名詞と動詞のチェック
    has_noun = any(morph.pos == "名詞" for morph in chunk.morphs)
    has_verb = any(morph.pos == "動詞" for morph in dst_chunk.morphs)
    
    # 名詞が含まれるチャンクが動詞を含むチャンクに係る場合
    if has_noun and has_verb:
        print(f"{surf}\t{next_surf}")

# 出力結果
"""
道具を  用いて
知能を  研究する
一分野を        指す
知的行動を      代わって
人間に  代わって
コンピューターに        行わせる
研究分野とも    される
"""


"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
リスト内包表記を使って、コンパクトで読みやすいコードを実現しています。
イギリスに関する記事を取得する際、filterとlambdaを使って処理を行っています。これにより、1行で必要なデータを抽出できます。

・p.10の「2.1 明確な単語を選ぶ」で，
変数名やコメントに明確な単語を使っています。例えば、"folderpath"や"filename"は、そのまま読んで意味が理解できる名前です。
"UK_article"という変数名は、イギリスに関する記事を表すため、そのままの名前を使用しています。

クラス設計と命名規則

クラス名や変数名は説明的で意味が分かりやすいようにする。
各クラスには役割を説明するコメントを付ける。


メソッドとインスタンス変数の初期化

__init__ メソッドでインスタンス変数を明確に初期化する。


ループや条件分岐の明確化

条件分岐やループ部分には意図や処理の流れを示すコメントを追加する。
条件が特定の値をチェックする場合、それが何を意味するのかを説明する。


メイン処理の分離と関数化

メインの処理と結果の出力を分離し、それぞれの役割を明確にする。
関数やメソッドを用いて、処理の再利用性と可読性を高める。


コードのコメントと説明

コード全体に役割や目的を説明するコメントを適宜追加する。
関数やクラスの前に短いコメントを付け、その機能を理解しやすくする。

"""