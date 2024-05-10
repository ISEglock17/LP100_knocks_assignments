"""
言語処理100本ノック 第3章課題

30. 形態素解析結果の読み込み
形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，1文を形態素（マッピング型）のリストとして表現せよ．第4章の残りの問題では，ここで作ったプログラムを活用せよ．


MeCabの基本として，次のフォーマットにしたがって，記録される。
表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
        [0],　　[1]　  ,　　[2]　  ,　　[3]　 ,[4]　 ,[5]　,[6] ,[7] ,[8]
"""

text = []   # テキスト
sentence = []   # 文
 
with open("./assignments_folder/Chapter4/neko.txt.mecab", "r", encoding="utf-8") as f:
    for line in f:
        line = line.split("\t")     # タブ文字で分割
        if len(line) == 2:          # タブ文字の有無の判別
            line[1] = line[1].split(",")    # コンマで分割
            sentence.append({"surface": line[0], "base": line[1][6], "pos": line[1][0], "pos1": line[1][1]})    # 辞書追加
            if line[1][1] == "句点":    # 句点の場合，textにsentenceの一文を追加
                text.append(sentence)
                sentence = []

# 出力確認
print(text)
"""
[[{'surface': '一', 'base': '一', 'pos': '名詞', 'pos1': '数'}, {'surface': '', 'base': '*\n', 'pos': '記号', 'pos1': '一般'}, {'surface': '', 'base': '*\n', 'pos': '記号', 'pos1': '一般'}, {'surface': '\u3000', 'base': 
'\u3000', 'pos': '記号', 'pos1': '空白'}, {'surface': '吾輩', 'base': '吾輩', 'pos': '名詞', 'pos1': '代名詞'}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': '猫', 'base': '猫', 'pos': '名
詞', 'pos1': '一般'}, {'surface': 'で', 'base': 'だ', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'ある', 'base': 'ある', 'pos': '助動詞', 'pos1': '*'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}], [{'surface': '', 'base': '*\n', 'pos': '記号', 'pos1': '一般'}, {'surface': '名前', 'base': '名前', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': 'まだ', 'base': 'まだ', 'pos': '副詞', 'pos1': '助詞類接続'}, {'surface': '無い', 'base': '無い', 'pos': '形容詞', 'pos1': '自立'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}], [{'surface': '', 'base': '*\n', 'pos': '記号', 'pos1': '一般'}, {'surface': '', 'base': '*\n', 'pos': '記号', 'pos1': '一般'}, {'surface': '\u3000', 'base': '\u3000', 'pos': '記号', 'pos1': '空白'}, {'surface': 'どこ', 'base': 'どこ', 'pos': '名詞', 'pos1': '代名詞'}, {'surface': 'で', 'base': 'で', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '生れ', 'base': '生れる', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'た', 'base': 'た', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'か', 'base': 'か', 'pos': '助詞', 'pos1': '副助詞／並立助詞／終助詞'}, {'surface': 'とんと', 'base': 'とんと', 'pos': '副詞', 'pos1': '一般'}, {'surface': '見当', 'base': '見当', 'pos': '名詞', 'pos1': 'サ変接続'}, 
{'surface': 'が', 'base': 'が', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': 'つか', 'base': 'つく', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'ぬ', 'base': 'ぬ', 'pos': '助動詞', 'pos1': '*'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}], [{'surface': '', 'base': '*\n', 'pos': '記号', 'pos1': '一般'}, {'surface': '何', 'base': '何', 'pos': '名詞', 'pos1': '代名詞'}, {'surface': 'でも', 'base': 'でも', 'pos': '助詞
', 'pos1': '副助詞'}, {'surface': '薄暗い', 'base': '薄暗い', 'pos': '形容詞', 'pos1': '自立'}, {'surface': 'じめじめ', 'base': 'じめじめ', 'pos': '副詞', 'pos1': '一般'}, {'surface': 'し', 'base': 'する', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'た', 'base': 'た', 'pos': '助動詞', 'pos1': '*'}, {'surface': '所', 'base': '所', 'pos': '名詞', 'pos1': '非自立'}, {'surface': 'で', 'base': 'で', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': 'ニャーニャー', 'base': '*\n', 'pos': '名詞', 'pos1': '一般'}, {'surface': '泣い', 'base': '泣く', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'て', 'base': 'て', 'pos': '助詞', 'pos1': '接続助詞'}, {'surface': 'い 
た事', 'base': 'いた事', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'だけ', 'base': 'だけ', 'pos': '助詞', 'pos1': '副助詞'}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': '記憶', 'base': '記憶', 'pos': '名詞', 'pos1': 'サ変接続'}, {'surface': 'し', 'base': 'する', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'て', 'base': 'て', 'pos': '助詞', 'pos1': '接続助詞'}, {'surface': 'いる', 'base': 'いる', 'pos': '動詞', 'pos1': '非自立'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}], [{'surface': '', 'base': '*\n', 'pos': '記号', 'pos1': '一般'}, {'surface': '吾輩', 'base': '吾輩', 'pos': '名詞', 'pos1': '代
名詞'}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': 'ここ', 'base': 'ここ', 'pos': '名詞', 'pos1': '代名詞'}, {'surface': 'で', 'base': 'で', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '始め', 'base': '始める', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'て', 'base': 'て', 'pos': '助詞', 'pos1': '接続助詞'}, {'surface': '人間', 'base': '人間', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'という', 'base': 'という', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': 'もの', 'base': 'もの', 'pos': '名詞', 'pos1': '非自立'}, {'surface': 'を', 'base': 'を', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '見', 'base': '見る', 
'pos': '動詞', 'pos1': '自立'}, {'surface': 'た', 'base': 'た', 'pos': '助動詞', 'pos1': '*'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}], [{'surface': '', 'base': '*\n', 'pos': '記号', 'pos1': '一般
'}, {'surface': 'しかも', 'base': 'しかも', 'pos': '接続詞', 'pos1': '*'}, {'surface': 'あと', 'base': 'あと', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'で', 'base': 'で', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '聞く', 'base': '聞く', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'と', 'base': 'と', 'pos': '助詞', 'pos1': '接続助詞'}, {'surface': 'それ', 'base': 'それ', 'pos': '名詞', 'pos1': '代名詞'}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': '書生', 'base': '書生', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'という', 'base': 'という', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '人間', 'base': '人間', 'pos': '名詞', 'pos1': '一般'}, {'surface': '中', 'base': '中', 'pos': '名詞', 'pos1': '接尾'}, {'surface': 'で', 'base': 'で', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '一番', 'base': '一番', 'pos': '名詞', 'pos1': 
'副詞可能'}, {'surface': '獰悪', 'base': '獰悪', 'pos': '名詞', 'pos1': '形容動詞語幹'}, {'surface': 'な', 'base': 'だ', 'pos': '助動詞', 'pos1': '*'}, {'surface': '種族', 'base': '種族', 'pos': '名詞', 'pos1': '一般'}, 
{'surface': 'で', 'base': 'だ', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'あっ', 'base': 'ある', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'た', 'base': 'た', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'そう', 'base': 'そう', 'pos': '名詞', 'pos1': '特殊'}, {'surface': 'だ', 'base': 'だ', 'pos': '助動詞', 'pos1': '*'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}], [{'surface': '', 'base': '*\n', 'pos': '記号', 'pos1': '一般'}, {'surface': 'この', 'base': 'この', 'pos': '連体詞', 'pos1': '*'}, {'surface': '書生', 'base': '書生', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'という', 'base': 'という', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': 'の', 'base': 'の', 'pos': '名詞', 'pos1': '非自立'}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': '時々', 'base': '時々', 'pos': '副詞', 'pos1': '一般'}, {'surface': '我々', 'base': '我々', 'pos': '名詞', 'pos1': '代名詞'}, {'surface': 'を', 'base': 'を', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '捕え', 'base': '捕える', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'て', 'base': 'て', 
'pos': '助詞', 'pos1': '接続助詞'}, {'surface': '煮', 'base': '煮る', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'て', 'base': 'て', 'pos': '助詞', 'pos1': '接続助詞'}, {'surface': '食う', 'base': '食う', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'という', 'base': 'という', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '話', 'base': '話', 'pos': '名詞', 'pos1': 'サ変接続'}, {'surface': 'で', 'base': 'だ', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'ある', 'base': 'ある', 'pos': '助動詞', 'pos1': '*'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}], [{'surface': '', 'base': '*\n', 'pos': '記号', 'pos1': '一般'}, {'surface': 'しかし', 'base': 'しかし', 'pos': '接続詞', 'pos1': '*'}, {'surface': 'その', 'base': 'その', 'pos': '連体詞', 'pos1': '*'}, {'surface': '当時', 'base': '当時', 'pos': '名詞', 'pos1': '副詞可能'}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': '何', 'base': '何', 'pos': '名詞', 'pos1': '代名詞'}, {'surface': 'という', 'base': 'という', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '考', 'base': '考', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'も', 'base': 'も', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': 'なかっ', 'base': 'ない', 'pos': '形容詞', 'pos1': '自立'}, {'surface': 'た', 'base': 'た', 'pos': '助動詞', 'pos1': '*'}, 
{'surface': 'から', 'base': 'から', 'pos': '助詞', 'pos1': '接続助詞'}, {'surface': '別段', 'base': '別段', 'pos': '副詞', 'pos1': '助詞類接続'}, {'surface': '恐し', 'base': '恐い', 'pos': '形容詞', 'pos1': '自立'}, {'surface': 'いとも', 'base': 'いとも', 'pos': '副詞', 'pos1': '一般'}, {'surface': '思わ', 'base': '思う', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'なかっ', 'base': 'ない', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'た
', 'base': 'た', 'pos': '助動詞', 'pos1': '*'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}], [{'surface': '', 'base': '*\n', 'pos': '記号', 'pos1': '一般'}, {'surface': 'ただ', 'base': 'ただ', 'pos': 
'接続詞', 'pos1': '*'}, {'surface': '彼', 'base': '彼', 'pos': '名詞', 'pos1': '代名詞'}, {'surface': 'の', 'base': 'の', 'pos': '助詞', 'pos1': '連体化'}, {'surface': '掌', 'base': '掌', 'pos': '名詞', 'pos1': '一般'}, 
{'surface': 'に', 'base': 'に', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '載せ', 'base': '載せる', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'られ', 'base': 'られる', 'pos': '動詞', 'pos1': '接尾'}, {'surface': ' 
て', 'base': 'て', 'pos': '助詞', 'pos1': '接続助詞'}, {'surface': 'スー', 'base': 'スー', 'pos': '名詞', 'pos1': '固有名詞'}, {'surface': 'と', 'base': 'と', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '持ち上げ', 'base': '持ち上げる', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'られ', 'base': 'られる', 'pos': '動詞', 'pos1': '接尾'}, {'surface': 'た', 'base': 'た', 'pos': '助動詞', 'pos1': '*'}, {'surface': '時', 'base': '時', 'pos': '名詞', 'pos1': '非自立'}, {'surface': '何だか', 'base': '何だか', 'pos': '副詞', 'pos1': '一般'}, {'surface': 'フワフワ', 'base': 'フワフワ', 'pos': '副詞', 'pos1': '助詞類接続'}, {'surface': 'し', 'base': 'する', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'た', 'base': 'た', 'pos': '助動詞', 'pos1': '*'}, {'surface': '感じ', 'base': '感じ', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'が', 'base': 'が', 'pos': '助詞', 'pos1': '格 
助詞'}, {'surface': 'あっ', 'base': 'ある', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'た', 'base': 'た', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'ばかり', 'base': 'ばかり', 'pos': '助詞', 'pos1': '副助詞'}, {'surface': 'で', 'base': 'だ', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'ある', 'base': 'ある', 'pos': '助動詞', 'pos1': '*'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}], [{'surface': '', 'base': '*\n', 
'pos': '記号', 'pos1': '一般'}, {'surface': '掌', 'base': '掌', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'の', 'base': 'の', 'pos': '助詞', 'pos1': '連体化'}, {'surface': '上', 'base': '上', 'pos': '名詞', 'pos1': '非
自立'}, {'surface': 'で', 'base': 'で', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '少し', 'base': '少し', 'pos': '副詞', 'pos1': '助詞類接続'}, {'surface': '落ちつい', 'base': '落ちつく', 'pos': '動詞', 'pos1': '自立
'}, {'surface': 'て', 'base': 'て', 'pos': '助詞', 'pos1': '接続助詞'}, {'surface': '書生', 'base': '書生', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'の', 'base': 'の', 'pos': '助詞', 'pos1': '連体化'}, {'surface': ' 
顔', 'base': '顔', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'を', 'base': 'を', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '見', 'base': '見る', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'た', 'base': 'た', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'の', 'base': 'の', 'pos': '名詞', 'pos1': '非自立'}, {'surface': 'が', 'base': 'が', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': 'いわゆる', 'base': 'いわゆる', 'pos': '連体詞', 'pos1': '*'}, {'surface': '人間', 'base': '人間', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'という', 'base': 'という', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': 'もの', 'base': 'もの', 'pos': '名詞', 'pos1': '非自 
立'}, {'surface': 'の', 'base': 'の', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '見', 'base': '見る', 'pos': '動詞', 'pos1': '自立'}, {'surface': '始', 'base': '始', 'pos': '名詞', 'pos1': '固有名詞'}, {'surface': ' 
で', 'base': 'だ', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'あろ', 'base': 'ある', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'う', 'base': 'う', 'pos': '助動詞', 'pos1': '*'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}], [{'surface': '', 'base': '*\n', 'pos': '記号', 'pos1': '一般'}, {'surface': 'この', 'base': 'この', 'pos': '連体詞', 'pos1': '*'}, {'surface': '時', 'base': '時', 'pos': '名詞', 'pos1': '非自立'}, {'surface': '妙', 'base': '妙', 'pos': '名詞', 'pos1': '形容動詞語幹'}, {'surface': 'な', 'base': 'だ', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'もの', 'base': 'もの', 'pos': '名詞', 'pos1': '非自立'}, {'surface': 'だ', 'base': 'だ', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'と', 'base': 'と', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '思っ', 'base': '思う', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'た', 'base': 'た', 'pos': '助動詞', 'pos1': '*'}, {'surface': '感じ', 'base': '感じ', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'が', 'base': 'が', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '今', 'base': '今', 'pos': '名詞', 'pos1': '副 
詞可能'}, {'surface': 'でも', 'base': 'でも', 'pos': '助詞', 'pos1': '副助詞'}, {'surface': '残っ', 'base': '残る', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'て', 'base': 'て', 'pos': '助詞', 'pos1': '接続助詞'}, {'surface': 'いる', 'base': 'いる', 'pos': '動詞', 'pos1': '非自立'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}], [{'surface': '', 'base': '*\n', 'pos': '記号', 'pos1': '一般'}, {'surface': '第', 'base': '第', 'pos': '接頭詞', 'pos1': '数接続'}, {'surface': '一', 'base': '一', 'pos': '名詞', 'pos1': '数'}, {'surface': '毛', 'base': '毛', 'pos': '名詞', 'pos1': '接尾'}, {'surface': 'をもって', 'base': 'をもって', 'pos': 
'助詞', 'pos1': '格助詞'}, {'surface': '装飾', 'base': '装飾', 'pos': '名詞', 'pos1': 'サ変接続'}, {'surface': 'さ', 'base': 'する', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'れ', 'base': 'れる', 'pos': '動詞', 'pos1': '接尾'}, {'surface': 'べき', 'base': 'べし', 'pos': '助動詞', 'pos1': '*'}, {'surface': 'はず', 'base': 'はず', 'pos': '名詞', 'pos1': '非自立'}, {'surface': 'の', 'base': 'の', 'pos': '助詞', 'pos1': '連体化'}, {'surface': '顔', 'base': '顔', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'が', 'base': 'が', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': 'つるつる', 'base': 'つるつる', 'pos': '副詞', 'pos1':
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