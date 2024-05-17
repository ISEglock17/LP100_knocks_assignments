"""
言語処理100本ノック 第3章課題

47. 機能動詞構文のマイニング

動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい．46のプログラムを以下の仕様を満たすように改変せよ．

「サ変接続名詞+を（助詞）」で構成される文節が動詞に係る場合のみを対象とする
述語は「サ変接続名詞+を+動詞の基本形」とし，文節中に複数の動詞があるときは，最左の動詞を用いる
述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
述語に係る文節が複数ある場合は，すべての項をスペース区切りで並べる（助詞の並び順と揃えよ）

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

with open("./assignments_folder/Chapter5/result47.txt", "w", encoding="utf-8") as f:
    for sentence in sentences:
        for chunk in sentence.chunks:
            for morph in chunk.morphs:
                if morph.pos == "動詞": 
                    for src in chunk.srcs:
                        predicates = []
                        if (len(sentence.chunks[src].morphs) == 2 and 
                            sentence.chunks[src].morphs[0].pos1 == "サ変接続" and 
                            sentence.chunks[src].morphs[1].surface == "を"):
                            
                            predicates = "".join([
                                sentence.chunks[src].morphs[0].surface, 
                                sentence.chunks[src].morphs[1].surface, 
                                morph.base
                            ])
                            
                            particles = []
                            items = []
                            for src2 in chunk.srcs:
                                particles += [morph.surface for morph in sentence.chunks[src2].morphs if morph.pos == "助詞"]
                                item = "".join([morph.surface for morph in sentence.chunks[src2].morphs if morph.pos != "記号"])
                                item = item.rstrip()
                                if item not in predicates:
                                    items.append(item)
                            
                            if len(particles) > 1 and len(items) > 1:
                                particles = sorted(set(particles))
                                items = sorted(set(items))
                                particles_form = " ".join(particles)
                                items_form = " ".join(items)
                                predicate = " ".join(predicates)
                                print(f"{predicate}\t{particles_form}\t{items_form}", file=f)

                        
# 出力結果
"""
注 目 を 集 め る	が を	ある その後 サポートベクターマシンが
経 験 を 行 う	に を	元に 学習を
学 習 を 行 う	に を	元に 経験を
進 化 を 見 せ る	て において は を	加えて 敵対的生成ネットワークは 活躍している 特に 生成技術において
進 化 を い る	て において は を	加えて 敵対的生成ネットワークは 活躍している 特に 生成技術において
開 発 を 行 う	は を	エイダ・ラブレスは 製作した
命 令 を す る	で を	機構で 直接
運 転 を す る	に を	元に 増やし
特 許 を す る	が に まで を	2018年までに 日本が
特 許 を い る	が に まで を	2018年までに 日本が
運 転 を す る	て に を	基づいて 柔軟に
注 目 を 集 め る	から は を	ことから ファジィは 世界初であった 関わらず
制 御 を す る	から を	多少 少なさから
研 究 を 続 け る	が て を	ジェフホーキンスが 向けて
研 究 を い る	が て を	ジェフホーキンスが 向けて
投 資 を 行 う	で に を	全世界的に 民間企業主導で
探 索 を 行 う	で を	実装し 無報酬で
研 究 を 行 う	て を	いう 始めており
研 究 を い る	て を	いう 始めており
投 資 を す る	に は まで を	2022年までに 韓国は
反 乱 を 起 こ す	て に対して を	人間に対して 於いて
監 視 を 行 う	に まで を	人工知能に 推し進められ 歩行者まで
監 視 を せ る	に まで を	人工知能に 推し進められ 歩行者まで
禁 止 を 求 め る	が に は を	4月には ヒューマン・ライツ・ウォッチが 記された
禁 止 を い る	が に は を	4月には ヒューマン・ライツ・ウォッチが 記された
追 及 を 受 け る	て で と とともに は を	とともに なった 公聴会では 拒否すると 整合性で 暴露されており
解 任 を す る	て は を	Google社員らは しかし 含まれており 発表した
解 散 を す る	が で は を	4月4日 Googleは 倫理委員会が 理由で 要請した
話 を す る	は を	みんな 哲学者は
話 を い る	は を	みんな 哲学者は
議 論 を 行 う	まで を	けっこう これまで 長時間
議 論 を く る	まで を	けっこう これまで 長時間

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

Morph、Chunk、Sentence の各クラスには、役割や目的を説明するコメントが記述されている。これにより、各クラスの役割や責務が明確になっている。


変数名の意味の明確化

変数名は意味が理解しやすいように命名されており、例えば self.surface、self.base、self.pos などがこれに当たります。これにより、コードの読みやすさが向上しています。


初期化とインスタンスの管理

Sentence クラスの初期化では、与えられた文節リストをそのまま属性として保持し、さらに係り元リストを構築する処理が行われている。これにより、文節間の係り受け関係を効果的に管理できます。


ファイルの読み込みとデータ処理の分離

ファイルの読み込みとそれに続くデータ処理が明確に分離されています。with open() ブロックでファイルを開き、その後の処理では読み込んだデータに対する操作が行われています。


条件分岐と処理の意図の説明

if 文や for ループには、条件の意図を説明するコメントが付けられています。例えば、if morph.pos == "動詞": の条件が「動詞の場合に処理を行う」という意図であることが明確になっています。


出力のフォーマットとコメント

print() 文での出力部分には、出力のフォーマットやその内容がどのような情報を表しているのかを説明するコメントがあります。例えば、print(f"{predicate}\t{particles_form}\t{items_form}", file=f) の部分で、述語、助詞の組み合わせ、アイテムの組み合わせをファイルに出力していることが分かります。

"""