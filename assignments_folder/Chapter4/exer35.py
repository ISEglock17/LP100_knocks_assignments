"""
言語処理100本ノック 第3章課題

35. 単語の出現頻度
文章中に出現する単語とその出現頻度を求め，出現頻度の高い順に並べよ．

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


# 課題35
from collections import Counter

word_list = []

for sentence in text:
    for word in sentence:
        if word["surface"] != "":
            word_list.append(word["surface"])
            
counted_words = Counter(word_list)

# 出力確認
print(counted_words.most_common())

"""
[('の', 9194), ('。', 7486), ('て', 6868), ('、', 6772), ('は', 6420), ('に', 6243), ('を', 6071), ('と', 5508), ('が', 5337), ('た', 3988), ('で', 3806), ('「', 3231), ('」', 3225), ('も', 2479), ('ない', 2390), ('だ', 
2363), ('し', 2322), ('から', 2032), ('ある', 1728), ('な', 1613), ('ん', 1568), ('か', 1530), ('いる', 1249), ('事', 1207), ('へ', 1034), ('う', 992), ('する', 992), ('もの', 981), ('君', 973), ('です', 973), ('云う', 937), ('主人', 932), ('よう', 696), ('ね', 683), ('この', 649), ('御', 636), ('ば', 617), ('人', 602), ('その', 576), ('一', 554), ('そう', 546), ('何', 539), ('なる', 531), ('さ', 514), ('よ', 509), ('なら', 483), ('吾輩
', 481), ('い', 478), ('ます', 458), ('じゃ', 448), ('…', 433), ('これ', 414), ('\u3000', 411), ('なっ', 404), ('それ', 381), ('来', 364), ('れ', 356), ('見', 350), ('でも', 346), ('時', 345), ('迷亭', 343), ('ませ', 330), ('いい', 320), ('三', 319), ('——', 319), ('まで', 313), ('ところ', 313), ('方', 312), ('二', 303), ('ず', 299), ('上', 294), ('まし', 289), ('寒月', 286), ('顔', 282), ('ぬ', 277), ('先生', 274), ('見る', 273), ('人間
', 272), ('だろ', 270), ('くらい', 269), ('僕', 268), ('たら', 262), ('さん', 260), ('なく', 258), ('気', 250), ('あり', 249), ('猫', 248), ('だけ', 246), ('出', 245), ('出来', 244), ('云っ', 241), ('また', 238), ('中', 
234), ('思っ', 232), ('ばかり', 231), ('十', 231), ('ごとく', 225), ('あっ', 221), ('どう', 220), ('って', 216), ('細君', 213), ('など', 205), ('鼻', 199), ('今', 195), ('大', 195), ('や', 194), ('者', 194), ('そんな', 194), ('あの', 189), ('しかし', 185), ('てる', 182), ('より', 181), ('ながら', 179), ('自分', 175), ('ので', 175), ('少し', 172), ('頭', 169), ('ちょっと', 169), ('でしょ', 162), ('訳', 159), ('前', 158), ('日', 154), (' 
声', 154), ('かい', 153), ('うち', 152), ('ただ', 150), ('知れ', 150), ('ほど', 150), ('聞い', 150), ('として', 149), ('私', 149), ('だって', 148), ('男', 147), ('思う', 146), ('たい', 146), ('行っ', 144), ('せ', 143), ('家', 143), ('子', 143), ('眼', 142), ('？', 141), ('ため', 140), ('見え', 139), ('よく', 138), ('出し', 137), ('彼', 134), ('誰', 133), ('たり', 133), ('かも', 132), ('間', 131), ('所', 127), ('知ら', 127), ('女', 126), ('もう', 125), ('え', 125), ('られ', 121), ('こんな', 120), ('金田', 119), ('どこ', 118), ('東風', 118), ('たる', 117), ('という', 116), ('今日', 116), ('ねえ', 116), ('まだ', 115), ('いや', 114), ('通り', 114), ('なけ 
れ', 113), ('苦', 112), ('ざる', 111), ('的', 111), ('さえ', 109), ('くる', 109), ('れる', 109), ('第', 108), ('例', 108), ('こう', 107), ('口', 107), ('まあ', 107), ('聞く', 106), ('なかっ', 106), ('なり', 104), ('わ', 
104), ('持っ', 103), ('馬鹿', 103), ('あれ', 103), ('五', 102), ('行く', 101), ('本', 101), ('心', 100), ('年', 99), ('沙弥', 99), ('四', 98), ('ぜ', 98), ('ここ', 97), ('とか', 97), ('手', 97), ('ええ', 97), ('やる', 96), ('大きな', 95), ('度', 94), ('話し', 94), ('分ら', 93), ('やっ', 93), ('下', 93), ('今度', 93), ('ちゃ', 93), ('考え', 92), ('しまっ', 91), ('くれ', 90), ('少々', 90), ('云わ', 89), ('まい', 89), ('ござい', 89), ('妙', 88), ('大変', 88), ('昔', 88), ('る', 87), ('面白い', 87), ('いくら', 86), ('あ', 86), ('奴', 86), ('あまり', 85), ('あなた', 85), ('鈴木', 85), ('っ', 84), ('もっ', 84), ('六', 84), ('云い', 84), ('仙', 84), ('出来る', 83), ('独', 83), ('学校', 82), ('なかなか', 82), ('金', 82), ('もっとも', 81), ('出す', 80), ('やはり', 80), ('なるほど', 80), ('どうも', 79), ('小', 79), ('さあ', 78), ('話', 77), ('それから', 77), ('得', 76), ('目', 
76), ('つけ', 75), ('すれ', 75), ('ましょ', 75), ('まま', 74), ('彼等', 74), ('運動', 74), ('以上', 74), ('仕方', 73), ('もし', 73), ('来る', 73), ('ヴァイオリン', 73), ('もん', 73), ('全く', 73), ('あろ', 72), ('のみ', 
72), ('つい', 72), ('つもり', 72), ('内', 71), ('物', 71), ('だい', 71), ('名', 71), ('早く', 71), ('何だか', 70), ('決して', 70), ('のに', 70), ('見える', 70), ('ほか', 68), ('それで', 68), ('出る', 68), ('音', 67), (' 
かく', 67), ('右', 67), ('思わ', 66), ('なんか', 66), ('様', 66), ('あと', 65), ('べき', 65), ('這入っ', 65), ('寝', 65), ('入れ', 65), ('大分', 65), ('そんなに', 65), ('す', 65), ('八', 65), ('教師', 64), ('食っ', 64), 
('みんな', 64), ('たく', 63), ('必ず', 63), ('あら', 63), ('おい', 63), ('相違', 63), ('心配', 63), ('毛', 62), ('無', 62), ('急', 62), ('いつ', 62), ('ところが', 62), ('黒', 62), ('だから', 62), ('先', 61), ('行か', 61), ('そりゃ', 61), ('なさい', 61), ('思い', 60), ('すると', 60), ('書斎', 60), ('候', 60), ('ら', 60), ('敵', 60), ('同じ', 59), ('まず', 59), ('事件', 59), ('はず', 58), ('足', 58), ('始め', 57), ('知っ', 57), ('供', 57), ('いえ', 57), ('において', 57), ('なに', 57), ('り', 56), ('お', 56), ('鼠', 56), ('立て', 56), ('奥さん', 56), ('しばらく', 55), ('気の毒', 55), ('やり', 55), ('帰っ', 55), ('分', 55), ('泥棒', 55), ('あんな', 55), ('生徒', 55), ('るる', 54), ('駄目', 53), ('無論', 53), ('ようやく', 52), ('かけ', 52), ('返事', 52), ('すぐ', 52), ('なくっ', 52), ('とも', 52), ('云え', 52), ('そこ', 51), ('せる', 51), ('笑い', 51), ('々', 51), ('いよい
よ', 51), ('驚', 51), ('なぜ', 51), ('逆上', 51), ('らしい', 50), ('横', 50), ('真面目', 50), ('裏', 49), ('好い', 49), ('一つ', 49), ('屋', 49), ('ろ', 49), ('これから', 48), ('いろいろ', 48), ('しよ', 48), ('しきりに', 48), ('首', 48), ('立っ', 48), ('娘', 48), ('水', 48), ('名前', 47), ('おら', 47), ('無理', 47), ('笑っ', 47), ('世の中', 47), ('妻君', 47), ('しまう', 47), ('どんな', 47), ('向う', 46), ('不思議', 46), ('んで', 46), ('せんだって', 46), ('体', 46), ('是非', 46), ('勢', 46), ('頃', 45), ('風', 45), ('なし', 45), ('吾', 45), ('極', 45), ('実は', 45), ('悪い', 45), ('問題', 45), ('七', 45), ('博士', 45), ('感じ', 44), ('まるで', 44), ('上
っ', 44), ('とき', 44), ('癖', 44), ('相手', 44), ('いかに', 44), ('勝手', 44), ('食い', 44), ('め', 44), ('円', 44), ('近頃', 44), ('聞き', 44), ('違', 44), ('実に', 43), ('何とか', 43), ('こっち', 43), ('西洋', 43), ('相', 43), ('やら', 42), ('下女', 42), ('答え', 42), ('百', 42), ('客', 42), ('ああ', 42), ('館', 42), ('考える', 41), ('聞か', 41), ('ご', 41), ('元来', 41), ('買っ', 41), ('結果', 41), ('垣', 41), ('念', 41), ('様子', 41), ('つく', 41), ('け', 41), ('致し', 41), ('なんて', 41), ('雪江', 41), ('食う', 40), ('ことに', 40), ('平気', 40), ('よる', 40), ('自己', 40), ('長い', 40), ('とうとう', 40), ('困る', 40), ('障子', 40), ('だっ', 40), 
('士', 40), ('多々良', 40), ('当人', 39), ('挨拶', 39), ('たって', 39), ('読ん', 39), ('髯', 39), ('構わ', 39), ('必要', 39), ('研究', 39), ('わから', 39), ('おや', 39), ('鏡', 39), ('時々', 38), ('とにかく', 38), ('すで
に', 38), ('へえ', 38), ('きっと', 38), ('どうか', 38), ('神', 38), ('時代', 38), ('いけ', 38), ('うん', 38), ('飲ん', 37), ('椽側', 37), ('上げ', 37), ('とっ', 37), ('自然', 37), ('感心', 37), ('事実', 37), ('車屋', 37), ('えらい', 37), ('会', 37), ('着', 37), ('死ん', 37), ('分り', 37), ('生れ', 36), ('暗に', 36), ('よかろ', 36), ('腹', 36), ('色', 36), ('不', 36), ('後', 36), ('座敷', 36), ('やめ', 36), ('落ち', 36), ('随分', 36), ('骨', 36), ('国', 36), ('なお', 36), ('面', 36), ('死ぬ', 36), ('意味', 36), ('飯', 36), ('武', 36), ('書生', 35), ('一番', 35), ('いう', 35), ('ちと', 35), ('せっかく', 35), ('さすが', 35), ('ちょうど', 35), ('承知', 35), ('病気', 35), ('心得', 35), ('坊', 35), ('点', 35), ('字', 35), ('君子', 35), ('実業', 35), ('探偵', 35), ('陰', 35), ('とうてい', 35), ('落雲', 35), ('穴', 34), ('どうして', 34), ('どうしても', 34), ('おり', 34), ('過
ぎ', 34), ('忘れ', 34), ('力', 34), ('けれども', 34), ('すこぶる', 34), ('※', 34), ('共', 34), ('到底', 33), ('不平', 33), ('至っ', 33), ('等', 33), ('ごとき', 33), ('向っ', 33), ('日本', 33), ('笑う', 33), ('取っ', 33), ('よほど', 33), ('影', 33), ('おく', 33), ('天下', 33), ('衛門', 33), ('真中', 32), ('非常', 32), ('左', 32), ('ついに', 32), ('やがて', 32), ('我', 32), ('毎日', 32), ('失敬', 32), ('どうせ', 32), ('切っ', 32), ('おれ', 32), ('時分', 32), ('夫婦', 32), ('にゃ', 32), ('待っ', 32), ('突然', 32), ('代り', 32), ('安心', 32), ('に対して', 32), ('過ぎる', 32), ('伯父', 32), ('かかる', 31), ('分る', 31), ('いっ', 31), ('他', 31), ('こりゃ', 
31), ('について', 31), ('枚', 31), ('一体', 31), ('いら', 31), ('たろ', 31), ('込ん', 31), ('書い', 31), ('立派', 31), ('令嬢', 31), ('き', 31), ('引き', 31), ('飛ん', 31), ('行き', 31), ('湯', 31), ('しかも', 30), ('別 
段', 30), ('はなはだ', 30), ('付け', 30), ('向', 30), ('後ろ', 30), ('べから', 30), ('耳', 30), ('言葉', 30), ('付', 30), ('似', 30), ('残念', 30), ('普通', 30), ('姉', 30), ('晩', 30), ('辺', 30), ('いか', 30), ('作っ', 30), ('逆', 30), ('ども', 30), ('によって', 30), ('山の芋', 30), ('考', 29), ('おっ', 29), ('台所', 29), ('最後', 29), ('尻', 29), ('すまし', 29), ('質問', 29), ('杯', 29), ('尻尾', 29), ('大丈夫', 29), ('喧嘩', 29), ('愉快', 29), ('なさる', 29), ('でし', 29), ('なあ', 29), ('舌', 29), ('返', 29), ('御前', 29), ('町', 29), ('尺', 29), ('烏', 29), ('逢っ', 28), ('くれる', 28), ('帰る', 28), ('傍', 28), ('いっしょ', 28), ('畳', 28), ('白
', 28), ('たしかに', 28), ('発達', 28), ('愚', 28), ('変化', 28), ('時間', 28), ('負け', 28), ('にやにや', 28), ('起し', 28), ('結構', 28), ('全体', 28), ('撫で', 28), ('変', 28), ('悪', 28), ('つ', 28), ('茶碗', 28), ('つか', 27), ('付い', 27), ('それでも', 27), ('夜', 27), ('己', 27), ('爪', 27), ('画', 27), ('開い', 27), ('這入る', 27), ('夢', 27), ('世間', 27), ('受け', 27), ('羽織', 27), ('関係', 27), ('迷惑', 27), ('馳', 27), ('損
', 27), ('だんだん', 27), ('ま', 27), ('妻', 27), ('諸君', 27), ('文明', 27), ('三平', 27), ('狂', 27), ('そうして', 26), ('坐っ', 26), ('我慢', 26), ('善い', 26), ('得る', 26), ('次', 26), ('一向', 26), ('底', 26), ('○', 26), ('なれ', 26), ('黙っ', 26), ('注意', 26), ('充分', 26), ('利か', 26), ('二つ', 26), ('嫁', 26), ('説明', 26), ('なあに', 26), ('あんまり', 26), ('答える', 26), ('貰っ', 26), ('先方', 26), ('苦しい', 25), ('ものの', 25), ('眺め', 25), ('見せ', 25), ('実際', 25), ('朝', 25), ('わるい', 25), ('かける', 25), ('有し', 25), ('さっき', 25), ('なん', 25), ('がっ', 25), ('場', 25), ('冗談', 25), ('く', 25), ('箸', 25), ('山', 25), ('拝見', 25), ('餅', 25), ('かかっ', 25), ('生き', 25), ('変ら', 25), ('戦争', 25), ('述べ', 25), ('万', 25), ('法', 25), ('着物', 25), ('化物', 25), ('邸', 24), ('この間', 24), ('読む', 24), ('言っ', 24), ('玉', 24), ('御馳走', 24), ('なるべく', 24), ('心持ち', 24), ('場合', 24), ('語', 24), ('教え', 24), ('ありがたい', 24), ('動物', 24), ('婦人', 24), ('壺', 24), ('九', 24), ('もう少し', 24), ('嫌', 24), ('下さい', 24), ('分っ', 24), ('よっ 
ぽど', 24), ('困っ', 24), ('種', 24), ('月並', 24), ('給え', 24), ('坊主', 24), ('希', 24), ('真似', 24), ('本人', 24), ('こそ', 24), ('帽子', 24), ('寸', 24), ('向い', 24), ('結婚', 24), ('地蔵', 24), ('個性', 24), ('胃
弱', 23), ('毫も', 23), ('自ら', 23), ('同時に', 23), ('廻っ', 23), ('癪', 23), ('やろ', 23), ('銭', 23), ('懸け', 23), ('皆', 23), ('うまい', 23), ('功', 23), ('取ら', 23), ('立ち', 23), ('大抵', 23), ('連れ', 23), ('な
かろ', 23), ('出掛け', 23), ('説', 23), ('御覧', 23), ('違い', 23), ('身', 23), ('いいえ', 23), ('嬢', 23), ('千', 23), ('世紀', 23), ('松', 23), ('甘木', 23), ('解釈', 23), ('しか', 23), ('ー', 23), ('布団', 23), ('自身
', 23), ('東京', 23), ('禿', 23), ('なか', 23), ('蝉', 23), ('記憶', 22), ('廻る', 22), ('尊敬', 22), ('判然', 22), ('ついで', 22), ('調子', 22), ('そこで', 22), ('むずかしい', 22), ('能', 22), ('意', 22), ('散歩', 22), 
('皿', 22), ('ことごとく', 22), ('学者', 22), ('際', 22), ('師匠', 22), ('気味', 22), ('られる', 22), ('水島', 22), ('しまい', 22), ('茶の間', 22), ('たちまち', 22), ('罪', 22), ('やっぱり', 22), ('自覚', 22), ('たまえ', 22), ('叔父さん', 22), ('煙草', 21), ('乱暴', 21), ('表', 21), ('天', 21), ('膝', 21), ('動か', 21), ('主義', 21), ('滑稽', 21), ('かよう', 21), ('よし', 21), ('こと', 21), ('個', 21), ('一般', 21), ('なんぞ', 21), ('源
', 21), ('長く', 21), ('合', 21), ('詩', 21), ('天然', 21), ('高い', 21), ('竹', 21), ('帯', 21), ('爺さん', 21), ('あばた', 21), ('至る', 20), ('さて', 20), ('黒い', 20), ('置い', 20), ('滅多', 20), ('神経', 20), ('現に
', 20), ('隣り', 20), ('せら', 20), ('熱心', 20), ('美学', 20)
"""



"""
―リーダブルコードの内容で実践したこと―
・p.171～p.173の「短いコードを書くこと」で，
リスト内包表記を使って、コンパクトで読みやすいコードを実現しています。
イギリスに関する記事を取得する際、filterとlambdaを使って処理を行っています。これにより、1行で必要なデータを抽出できます。

・p.10の「2.1 明確な単語を選ぶ」で，
変数名やコメントに明確な単語を使っています。例えば、"folderpath"や"filename"は、そのまま読んで意味が理解できる名前です。
"UK_article"という変数名は、イギリスに関する記事を表すため、そのままの名前を使用しています。

"""