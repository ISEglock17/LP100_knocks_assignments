"""
言語処理100本ノック 第3章課題

46. 動詞の格フレーム情報の抽出
45のプログラムを改変し，述語と格パターンに続けて項（述語に係っている文節そのもの）をタブ区切り形式で出力せよ．45の仕様に加えて，以下の仕様を満たすようにせよ．

項は述語に係っている文節の単語列とする（末尾の助詞を取り除く必要はない）
述語に係る文節が複数あるときは，助詞と同一の基準・順序でスペース区切りで並べる

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
            
with open("./assignments_folder/Chapter5/result46.txt", "w", encoding="utf-8") as f:
    for i in range(len(sentences)):
        for chunk in sentences[i].chunks:
            for morph in chunk.morphs:
                if morph.pos == "動詞": 
                    particles = []
                    items = []
                    for src in chunk.srcs:
                        particles += [morph.surface for morph in sentences[i].chunks[src].morphs if morph.pos == "助詞"]
                        items += ["".join([morph.surface for morph in sentences[i].chunks[src].morphs if morph.pos != "記号"])]
                    if len(particles) > 1 and len(items) > 1:
                        particles = sorted(set(particles))
                        items = sorted(set(items))
                        particles_form = " ".join(particles)
                        items_form = " ".join(items)
                        print(f"{morph.base}\t{particles_form}\t{items_form}", file=f)

                        
# 出力結果
"""
する	て を	用いて 知能を
代わる	に を	人間に 知的行動を
行う	て に	コンピューターに 代わって
せる	て に	コンピューターに 代わって
述べる	で に の は	佐藤理史は 次のように 解説で
いる	で に の は	佐藤理史は 次のように 解説で
する	で を	コンピュータ上で 知的能力を
する	て を	パターンを 解析して
する	て を	パターンを 解析して
ある	が は	応用例は 技術ソフトウェアコンピュータシステム 画像認識等が
する	で に により	1956年に ある ジョンマッカーシーにより ダートマス会議で
れる	で に により	1956年に ある ジョンマッカーシーにより ダートマス会議で
する	と を	主体と 記述を
使う	で でも は	命名された 意味あいでも 現在では
れる	で でも は	命名された 意味あいでも 現在では
いる	で でも は	命名された 意味あいでも 現在では
ある	て も	ことも 使われている
出す	が に	しばしば プログラム 人工無脳が 引き合いに
れる	が に	しばしば プログラム 人工無脳が 引き合いに
する	に を	役割を 計算機に
...
"""



"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
リスト内包表記を使って、コンパクトで読みやすいコードを実現しています。
イギリスに関する記事を取得する際、filterとlambdaを使って処理を行っています。これにより、1行で必要なデータを抽出できます。

・p.10の「2.1 明確な単語を選ぶ」で，
変数名やコメントに明確な単語を使っています。例えば、"folderpath"や"filename"は、そのまま読んで意味が理解できる名前です。
"UK_article"という変数名は、イギリスに関する記事を表すため、そのままの名前を使用しています。

適切な変数名とコメント: 変数名やコメントが明確であり、コードの理解を助けています。例えば、textやsentenceといった変数名は適切であり、コメントも関数の役割を説明しています。
可読性のためのコードの分割: プログラム全体が1つの長い関数ではなく、論理的なブロックに分割されています。これにより、各部分の機能を理解しやすくなります。
条件分岐の明確化: 条件分岐が必要な箇所で条件式が明確になっており、可読性が向上しています。例えば、if len(line) == 2といった条件はシンプルで理解しやすいです。
適切なデータ構造の使用: 形態素解析結果を格納するために辞書やリストといった適切なデータ構造が使用されています。これにより、データの扱いや操作が容易になります。
適切な関数の使用: ファイルの読み込みやデータの解析に適切な関数が使用されており、コードが効率的であり、可読性が高まっています。
一貫性のあるスタイル: インデントやスペースの使用が一貫しており、コードのスタイルが統一されています。

"""