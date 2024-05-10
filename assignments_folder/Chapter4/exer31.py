"""
言語処理100本ノック 第3章課題

31. 動詞
動詞の表層形をすべて抽出せよ．

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


# 課題31
verb_set = set()   # 動詞の集合

for sentence in text:
    for word in sentence:
        if word['pos'] == '動詞':
            verb_set.add(word['surface'])

print(verb_set)
            

"""
{'忍び込む', '思い立っ', '癒す', '掻い込ん', 'ち', '断わり', '見計らっ', '追付こ', '散る', '勝れ', '有する', '殺せ', '与っ', '説く', '打ち殺す', '習お', '燃える', '滅入っ', '凌い', '手伝っ', '与える', '貪る', '示す', '渡
る', '睨め', '詰め', '泣かせ', '着せる', '説き', '突き', '誘い出す', 'とまっ', '返す', '送れ', '立ち上っ', '頼む', 'つまら', '清め', 'やり込める', '綻びる', '浮ぶ', '片づか', '決める', '述べよ', '走ら', 'いじっ', '下ら', 'こん', '引き起し', '舞っ', '降らせる', '上がら', '逆らわ', 'しつけ', '察し', '破れ', '控え', '在る', '入っ', '照らさ', '背く', '溢れる', '合わ', '押し倒し', '乗る', 'ひっくり返し', '添う', '逢い', '叩き', 'ござい', '奢
り', '据える', 'まします', '坐り込ん', '構える', 'おどし', 'なりゃ', '呑み', 'やってのけ', '伝っ', '云お', '取り去る', '差支え', '上げよ', '出る', '申し込ん', '外れ', 'ぬから', '担ぎ出す', 'いる', '捕り', '較べ', '落とし
', 'うむ', 'なやん', '騒ぎ立てる', 'ともり', '抱え込ん', 'さし', '直っ', '挟ま', '減ぜ', '悟っ', '巡り', '覗く', '散っ', '分け', 'おくれ', '扱き', 'ほてっ', '見舞う', '過し', '汚れ', '放し', '収め', 'おっ', '溶かし', '振
り向く', 'わかり', '仰向き', 'くれろ', '枯れ', '飛び降り', '掘り', '見向き', '奪い合っ', 'くるみ', '書か', '起きろ', '読める', '聞か', '捕える', '別れ', '斬り', 'つかまっ', '待たさ', '乗ん', '慣れ', '焦がれ', '隣り', '拡
げ', '着い', '絞', '違っ', '云い', 'もらう', '起き直っ', 'はじめる', '譲ら', '就く', 'で', 'のせ', '覚ます', '刈り', '流れる', '通り過ぎる', '届か', '聞かさ', '引っ掻き', '擦っ', '可愛がっ', '仕立てる', '澄まし', '聴き', '遺す', '起る', '訳す', '巻き', '涼み', '流れ', '書き', '名乗っ', '手こずっ', '驚きゃ', '試みよ', 'ちまっ', '捨て', 'あたら', '依る', '突き出し', '追い出さ', '近づく', '刈り込ん', 'くべる', '触れ', '帯び', '誂え', '嵌っ
', '似合う', '争う', '叶え', '痛み入る', '向い', '聞き合わせ', 'ありがたがる', 'くりゃる', '振り返る', '比べる', '見え', 'まえる', '磨く', 'いいかね', '途切れる', '出直す', '飲む', 'うなされる', '瞬く', '逢っ', '押す', 'ふわ', 'あく', 'やみ', '裏返し', '制する', '目だた', '伝わる', 'もぐり', '嗅い', '見廻', '暴く', '押しつけ', 'だまさ', '錬', 'もぎとる', 'よる', '逆上せん', '落ちつけ', '貪り', '勧める', '拭き込ん', '浮かれ', '凝らし', '暮す', '舞い込む', '持っ', '遂げ', '奮っ', '伺っ', '及ば', '振', '与えん', '承る', '送る', '透き', '消え', '突い', '鳴い', '吐か', '殺す', '蒙', 'つくばい', '取り留め', '降さ', '繰返す', '投げ込ん', '繋が', '撓め', '追い
やっ', 'おどる', '撲っ', '漂う', '見棄て', 'のしかから', '躍っ', '属し', 'ゆき', '弾ずる', '書こ', '過ぎ去っ', '変え', '寝かさ', '突っ込ん', 'かかる', '問わ', 'もらわ', '明い', '磨き', '書き流し', '挟ん', '踏ん', '苦しめ
ん', '立ち寄っ', '移す', '送ら', '返ら', 'くべ', 'やり', 'あらわれる', '動く', '負え', '染み', '始める', '引き戻し', 'ぶらさがっ', '気に入る', '入り込ん', 'かかわる', '反し', 'ふれ', '焚け', '切り抜ける', 'くらさ', '言い
付け', '込める', 'ほてる', '上っ', '持ち出し', '包む', '申し込も', '過ぎ', '漲っ', 'とりかかる', '目立っ', '吹き出し', '起こさ', '疲れる', 'くらむ', '気に入ら', '飛び越える', '立た', 'てっ', '埋', 'あい', '揃え', '逢う', '問い', 'もとづく', '引き込ん', '撓り', '採る', 'なれん', '来し', '乗っ', '立てれ', '踏ま', '陣取っ', '布く', '嘯い', '茂っ', '逸する', '入り乱れ', 'あぶり', '間に合う', '見下ろし', '適し', 'はおり', '役に立つ', '達せ', '司', '嵌め', '弄する', '浮ん', 'くう', '見せ', 'あせっ', 'あたる', 'いれ', '咽び', '覚し', '敲き', '焦げ', 'つかまえ', '出逢っ', 'まぜ返す', 'たくっ', 'なくなる', '押し出す', '戴い', '見る', '蒙る', '願う', '待ち兼ね', '添え', '遠のい', '縫いつけ', '呼ばわり', 'わる', '聞え', '足そ', '焼い', '溯ら', '吹かし', '張り', '剥い', '拵え', '冷え', 'ふさい', 'かぎっ', 'すくい', '続け', 'ぬっ', '有り', '思い立ち', '疑わ', '引き返す', 'ちまお', 'つとめ', 'つづける', '斬る', 'せよ', '弾か', '勝た', '食いつく', 'しい', '損なっ', 'まくっ', 'すぼめる', '受け取ら', '返る', '狂わせ', '踞っ', '告げ', 'つら', '扱が', 'まて', 'あける', 'つかれ', 'ひる', 'くれる', '称し
', 'たまり', 'ぱくつく', '廻っ', '卸せ', '断わら', '罵', '足りん', '響い', '生え', 'しでかす', '飛び出し', '迷っ', '渡す', '払う', '曲がろ', 'あやまっ', '産まれ', '律すれ', '張りつめ', 'ふう', '近づい', 'もうし', '上がろ
', 'かぶっ', 'うたう', '見縊', '引き抜く', 'とろ', '失せ', '処する', 'しゃべり', '戻す', '威張っ', '相成', '属する', '遮っ', 'あやまれ', 'つれ', '真似れ', 'とぼけ', '打ち消す', '引き立た', 'れりゃ', '燻り', 'おろす', '絶
つ', '討と', '煩わし', '立ち', '生き延び', 'つくし', '下り', '見付かる', '振り翳し', '名乗る', '撚っ', '味わう', 'そそのかし', 'こぼれ', 'ころがり', 'きつける', '奢る', '考えれ', '買い込ん', 'とばす', '打ち立て', '乗り越
える', '飛び上がっ', '叩い', '凌ぐ', 'あてつける', '劃し', 'おひゃらかす', 'でき', '置ける', '見当り', '流行る', '名づける', '頼ん', '仕入れ', 'しから', '集まる', '似', '持て余し', '釣っ', '聞こえよ', '当っ', '繰り返さ', '捌け', '化け', 'やめる', '詰る', '鳴らし', 'ならべ', '担い', '逢わ', '有ん', '含まっ', '見', '憤る', 'いん', '潰す', '切り落し', '具え', 'かろ', '通り掛っ', '食べる', '救っ', 'すべれ', 'しまえ', '縊れる', 'あびせる', '懸っ', '雇っ', 'もとめ', '覚め', '云う', 'しかり', '怒ら', '戴く', '壊れる', '逃がす', 'かしこまり', 'おける', 'ござっ', '飛び込み', 'なぐる', '切れ', 'とまり', '書い', '出れ', '召し上がり', '動ずる', '食わせれ', '飛込め
', '入れ', '刺し', '着く', '及ぼす', '思っ', '引張る', '去ら', 'あたわ', '食わせる', 'つかみ合い', 'たて', '止せ', 'くだけ', '見かね', '下ろし', '述べ', 'やむ', '隠す', '破', 'かい出す', '仰せつけ', '視る', '守る', 'やろ
', '咎め', '上げ', '盛り込ま', '取り掛る', '気が付き', '貫く', '張れ', '生やし', '憤っ', '引き寄せる', '寝転ん', '話し', '取り殺し', '汚す', 'ほかなら', '結え', 'かさね', 'ぐれ', '示し', '積ま', 'させ', '戻っ', 'さまし', '受け', '増す', '引き起す', '乗じ', 'わける', 'とおっ', 'こめ', 'やっつける', '略する', '運ば', '障っ', '留まり', '取りはずす', '恨む', '連れ出さ', 'おえ', '飛びつい', 'あら', 'くるん', '進ん', 'とまら', '好む', '儲かっ
', '引きずっ', 'れよ', 'のぼら', 'かくし', 'あつかう', '蒸し', '洗え', '結う', '帰そ', '吹き出す', '己惚れ', '合わせる', 'まかり', 'とん', '迂', '働か', '敷か', 'つかし', '見せびらかし', '生じ', '付か', '更け', 'しまっ', '言い触らす', 'からかい', '敗れ', '存じ', '掘っ', '置き', '下され', '断わる', 'あつかっ', '廻る', '生れ変っ', '蒙っ', '冠せ', '買え', '聞き', '引く', '振り撒い', '事足る', '説い', 'つ', '突きつけ', '寄りつい', '追っ', '注す', 'すう', 'あるく', '諭し', 'ふる', '仰', '余す', '構わ', '称せ', '蹴返す', '被る', '労する', '追いつく', '釣り合う', '寝つき', '附け', '改めれ', '察せ', 'やむを得ん', '始まら', '切ろ', '落ちれ', '尋ね', 'やん', '凝
っ', '這入', '困り', '写し', 'かおっ', '拭う', '究め', '限っ', '飾る', 'かい', '打ち抜い', '調子づく', '出来れ', '減る', '割り込ん', 'ぬう', 'なぐさみ', 'めで', '登る', '覚まし', 'とく', 'ごまかす', '貪', 'あばか', '切れ
る', '論ずる', '立ち至り', '禿げる', '渡り', '余っ', '飛び降りる', '抱え', 'むすん', '洩らし', '失う', '病み', '売り', '爛れ', '流す', '見張っ', '開く', '折れん', '済ま', '買う', '済み', '限ら', '言いつけ', '吐き', '降り
る', 'からまっ', 'しよ', '分りゃ', '脱する', '埋め', '鳴かし', '傭っ', 'はぐっ', '気がつく', '測る', '動い', 'ほのめかす', '仕切っ', '劣ら', '寄こし', 'くだら', '来い', '還せ', '隠', 'よす', 'こり', '食お', '睨みつける', '打ちゃ', '覚っ', 'やっ', '開ける', 'おき', 'ひやかす', '還す', '切らす', 'ひき', 'なっ', 'ぶらさげ', '張り詰め', '争っ', 'いたろ', '住ん', 'ぶら下げる', '過ぎん', '逃げ延び', '振り立て', '障り', 'なぐりつける', '使え', '吸っ', '洒落れ', '握ろ', 'かくそ', '利かし', '関せ', '極まっ', '括っ', 'とりゃ', '敲く', '吹い', '申し聞ける', '驚かし', '取り扱い', '繰り', '寝そべっ', '乗り越す', 'かかわら', '形づくる', '住め', '稼い', '見合せよ', '降る', '与え', '困りゃ', '探', '侮る', '忍ばし', '割っ', '契っ', 'せら', '担が', 'はいら', '取り合せ', '臨む', '食み出し', '廻さ', 'かむ', '差し支え', '巡っ', 'われ', '有る', '蹶', 'くるまっ', 'ねばり', '兼ねる', 'おこせ
', 'かきつけ', '滑る', '荒立て', '載せる', '踏む', '叶っ', '抛り', '咲い', 'しだす', 'たで', 'かこつ', 'たれる', '鳴り', 'ほ', '和する', 'すまさ', '繰り返し', '横切っ', '罵る', '真似る', '覆せ', '張っ', '洗う', '洗い', '取り立て', '息ん', '満ち', '比し', '迷わさ', '這入ろ', '栄える', '張り付ける', '弱っ', '戻る', 'あらわす', '返し', '噛め', '忍ん', '込み', '煎じ', '持ち上がっ', '連れ出し', 'まし', '問いかける', '置い', '整っ', 'くらん', 'とれん', '怒鳴る', '写し出す', '参る', 'はみ出し', '改める', '痛み入っ', '飛び', '見出す', '晴れ渡っ', '申し立て', '充たす', '話せ', '鳴', '詫び', 'たてる', '聞き糺し', '始まっ', '舂き', 'くらし', '飽き飽きし', '開き', '登り', '向け', 'おる', '割り切れる', 'もらっ', '割り込む', '乱れ', '開け放っ', '掛ける', '仕ら', '飛込ん', 'ひっくり返る', 'うたい', '吸い込ん', '集まれ', '湧き出る', '居', '消し', '心付い', '召し上がれ', '崩す', '関す
る', '救い出し', '落す', 'い', '思い出し', '列ね', 'はちきれん', 'かえし', '写せ', 'そり返っ', '持て余す', '駆ら', '振れ', '受け合わ', '布か', '響き', '延び', '利かす', 'さげ', '待て', 'つぶやい', '敷き', '抜く', '交ぜ', '足ら', '散らし', 'いりゃ', '落ちつき', '奪わ', '起き直り', '参っ', '冒し', 'か', 'き', 'れ', '潰れる', '蒙り', 'すい', 'かくす', '引ずり', '出合っ', '漕ぎつけ', '乞う', '仕込む', 'あまり', 'いける', '切っ', '透かし', '確かめ', '活かす', '弾きゃ', '違う', '馴らし', 'こせつい', '捕っ', '寄せる', '返り', 'つまり', '笑われる', '湧い', '扱か', '張付け', '学ぶ', '佇ん', '食い切っ', '迎える', '及ぶ', '退ける', '抜か', 'こたえる', '跳ね', '飲
ま', '忍び', '感じ', '抛', '参らせる', 'きる', '出来', '引き下がっ', 'かくれ', '押しやり', '済し', '堪り', '繙け', 'はいる', 'からかう', '用い', '念じ', 'ぶら下がる', '盗ら', '流行り', 'そそのかさ', '飛び離れ', '着こなし
', '印し', '寝る', '開い', '行き過ぎよ', '纏まる', 'こぼす', 'めぐらし', 'かけ', 'ひく', '追払わ', '引きかえそ', '取り巻い', '通り過ぎ', 'ひねくり', '心得', 'かき合せ', '気がつか', '刺し通し', '飲み干し', '残る', 'いらっ
しゃれ', '穿く', '消える', 'すら', 'かため', '塗りつけ', '拗じ', '裂き', 'ぶる', 'やい', '斬っ', '抜き', '見合せ', '転ずる', 'こしらえれ', '笑う', '嘆ぜ', 'しゃべら', '合っ', '触る', '捩じ', '擦り付け', '持ち上げ', '煮え
', 'ぬくもっ', '押し通そ', '消そ', 'しめ', '図ら', '真似', '害する', 'ととのっ', '見つめ', '飲み込ん', '混ぜ', '圧し', '飛ん', 'あるき', '焦れ', 'わか', '好か', '縫っ', 'やら', '言い兼ね', '拭い', '塗り', '腐る', 'なり', '飾っ', '薫ずる', '感じ入る', '取り残さ', 'よし', '見立て', '降り', '投げ込む', '欠ける', '若か', '及ぼ', '打ち解け', '衒う', 'かわし', '沈ん', '間に合わ', '怒鳴っ', '尽し', '直そ', 'た', 'つき合わ', 'あがり', '釣り込ま
', '驚か', '拝する', 'してやり', '見下し', '忘れろ', 'こし', '剰', '続か', '乗', 'とまる', 'ためし', '祟っ', 'のべる', '回っ', '引け', 'ぼ', '惹か', '縛せ', '聞き出し', '相成り', '遊ばし', 'とろけ', '怪しま', 'もったいぶ
っ', '垂らす', 'わり', '行き過ぎ', 'だまる', '通じ', 'やっつけ', '起きる', '擦り', '囃し', 'もが', '供し', '貰わ', '持ち直す', '朽ち', '収める', 'くれ', '食わ', '落ちつか', '守っ', 'ともっ', '引き付け', '見詰め', '値する
', 'ござる', '揚げ', '進む', '逃げ惑う', '思い込ん', 'ふて', '挙っ', '撫でる', '律する', '得よ', 'やり過ごし', '持ちかけ', 'とら', '磨る', '崩れ', '望ん', '食べ', 'でる', '画き', '踞る', 'うめろ', '伝え', '折ら', '叱りつ
け', '吸取り', '訳し', '移し', '下げる', '飲も', 'しょ', '欺く', 'ぬける', '試みる', '為し', '釣ら', 'はね', '伏せっ', 'あう', '当ら', 'あるい', '飛び込ん', 'ここ', '拾わ', '踏みつけ', 'され', '折れ', '能い', '止す', '誂
える', '費やさ', '見せる', '見透かさ', 'さておい', 'ある', 'はなす', 'とっ', '心付く', '寝かし', '除き', '近付く', '求め', 'なすっ', '心付か', '這っ', '有す', '漕ぎ', '祝す', '取る', 'なりすまし', '笑っ', '変じ', '差出す
', '踊り', '控える', '言う', '周章てる', '引い', '圧さ', '衰え', 'かじっ', '止めよ', '分る', '練り', 'ちまう', '愛し', '残っ', '有っ', 'ほめ', '込めん', 'ふくらし', '叩く', 'ふくれ', '納め', '放り出し', 'つけ', '教えろ', '振り落とそ', '折れる', '計ら', '肥っ', '坐', '捕れる', '据え付け', 'なさろ', '起す', '押し通す', '振りかけ', '呈し', 'いろ', '縛ら', '徹っ', '食わす', '重なる', '浚い', '垂れん', '明か', 'あき', '歌っ', '致そ', '渋り', '潰せる', '立ち退く', 'ひねくっ', '足り', '切る', '割り切れ', '引き上げる', '払っ', '話せる', '縮ま', '痛ん', 'ゆく', '据わっ', '動じ', '描く', '罵り', '取り違える', '目立つ', '給う', '就き', '咎める', '終', '捻り出し', 'こい', '濁す', '羨む', '笑い', '出よ', 'かね', '鳴らさ', '与', '絞める', 'かん', '恨も', 'あばれ', 'はいり', '惚れる', '寄せ', '返さ', 'ねじ上げ', '怒鳴りつけれ', '振り', 'すまい', 'なくなっ', 'たべる', '着ける', '陥る
', '這入ら', '建つ', '糺せ', '剥がれ', '判ぜ', 'あきれ返っ', '対し', '弾く', '償わ', '取', 'しらべ', '載っけ', '下さっ', 'かすん', '済ます', '太れる', 'いっし', '飲め', 'きか', 'くずれ', 'しなび', '植え付け', '舐め', '繙
く', '生れる', '断わっ', '問い返さ', '這う', '執る', '捕え', '得ろ', 'せる', '懸かっ', '加わっ', '遣わし', '上げる', '放と', '出す', '運ぶ', '吸い', '見つかっ', '魂消', '禁ずる', '燃え', 'はいっ', 'つっ', '附し', 'いつわ
り', 'やりゃ', 'してやる', '浴びせ', 'よそう', '競っ', '付い', '噛み', '話しかける', '問い掛ける', '見廻し', '窺う', '余る', '滑ら', '解し', 'はやる', '下る', '越せ', 'ふい', 'もっ', '離れ', 'あつまっ', '見付け', '生ぜ', '煙り', 'すわ', '弱ら', '売っ', 'きら', '訴え', '整え', 'つけよ', '明け', '考え直し', 'わめく', '開け放し', 'たつ', '迎えれ', '祈ら', '詰まら', '逃げ出す', '蘇', '呼ぶ', 'とき', '弾き出す', '捲い', '心得る', '研ぎ', 'う
く', 'むけ', 'あがる', '広げ', '誘い出し', 'けりゃ', '引っ張り', '浸っ', '留っ', '忌む', '散らす', '遊び', 'やめ', '造ろ', '取り払う', '陥れる', '推せ', 'たけ', '穿つ', 'うて', '届い', '儲け', 'まつわっ', 'られ', '描き出
し', '負け', 'あつまる', 'れん', '造る', '溜ら', '住む', '走る', 'あり', '優っ', '払え', '外し', '縮める', '見下せる', '叩き上げ', 'やってのける', '論じ', 'みよ', '致す', '片づい', '改まる', 'いに', '飲み下し', '呑ん', 'しきら', '貼っ', '切り抜けよ', '寄っ', '浴びせかけ', '廻り', '考え付い', '富ん', '売れ残っ', '捉え', '命ずる', '寝', 'こすっ', '思う', '吃', '抜け出し', 'じらせる', '預っ', 'ごまかし', '刻ん', 'あらそう', 'おけ', '話そ', '埋っ', '書き付け', '正し', '打ち殺し', '抑える', '罹る', '並みいる', '浮ば', '臨ま', '絶っ', '連ね', 'きまっ', 'ぶらつい', '侵し', '運ん', '恐れ', '気取っ', '痛み', '張り込ん', 'いそが', '下がれ', '開か', '呼び立て', 'まかり間違っ', 'あび', '開け', 'あらわ', 'よみ', 'しごき', '生き返る', 'すも', '片づけ', '誘う', '沁む', '承', '振ら', 'からげる', '包み', '利い', '浮く', 'いこ', '済む', 'ぬかり', '浮かし', 'おどかさ', '潜ら', '引き受け
る', '使う', 'おっしゃる', '曲がり', '転がっ', '取払っ', '写っ', '入ら', '抑え', '沁み', '送っ', '立て', 'れろ', '指す', 'とる', 'いけ', '忌み嫌っ', '寝過ごし', '遊ば', '越える', '放り込ん', '入る', '結っ', '泣い', '帰れ
', '選ん', '群がる', '放さ', '転じ', '申さ', '設け', '引っ込ま', 'いらっしゃる', '浮き', 'きつけ', '罵っ', '磨っ', 'なし', '評する', '反り', '劣る', '有し', '起き', '目する', '振っ', '敷い', '焚い', '解く', '取り上げる', '読め', '引き', '思いつい', '引っ張る', '謹ん', 'たべ', '浮べ', '片付ける', '叱る', '抜い', '果せる', 'なる', 'かから', 'いたっ', '眩む', '潜っ', 'くっ', '下りる', '差し出す', '書き入れ', '敬い', '見離さ', '知らせ', 'わ
かる', '死な', '奪っ', '待ち受け', '起り', 'とりとめ', '聞きつけ', '売', '言いつける', '泊っ', '窘め', '歩い', '跳ねのけ', '向き直っ', '任じ', '進ぜる', '読み', '浴びろ', '凝らす', '占め', '暮らさ', '通ら', '引きずり', 'くずさ', '抱く', '担ぎ', '知らし', 'すぎる', '頬張る', '減ら', '折っ', '追い出し', '聞き合せ', '取りあげ', '待ち合せ', '間に合っ', '売る', '障る', 'やる', 'いえ', 'ゆか', 'つき', '好み', 'きい', '撓る', 'ひ', '起っ', '伝
える', '買わ', '陥っ', '娶る', '悔やん', '被れ', 'ねむり', 'いき', '始め', '引き込ま', '延びる', '取り乱さ', '見識張っ', '進も', 'はち切れる', '褒め', '分り', '称え', '譲る', 'すべり', '相成っ', 'ひそめ', '並べ', '揉み', 'ぼり', '転がる', '下げ', '変っ', '叫ん', '備われ', '着', '見合わせ', '動き出し', '飲ん', '証拠立て', 'のぼる', 'つくっ', '寄れ', '尖ん', '脱ぐ', '期し', '付く', '逆立て', '飛び越え', '申し聞け', 'くぐっ', 'すれ', '立っ
', 'むき出し', 'なめる', '撮っ', 'あっ', 'かざら', '飲み', 'きき', '進める', '弾け', 'ゆるん', '流れれ', 'うなり', '交ぜ返し', '湧き出', 'けれ', '来れ', '蓄える', 'なさる', 'なげ', '鳴き', '綴る', '集まっ', 'せめ', '向け
る', '捲く', '通り抜け', '呼びかけ', '萌す', '併せ', '怒っ', 'せい', '飛び上がろ', '遠ざかる', '積ん', '差し出し', '見せびらかす', '逸す', '過ぎる', '延ばし', '届け', '行わ', '輝い', '喰う', '尋ねる', '見や', '砕けよ', '攻め', '逆らっ', '沈み', '翳す', '据え', '刈り込ま', 'すかし', '至る', 'らっしゃい', '涸らす', '承り', '喚び', 'うつし', '犯し', '移ら', '防ぐ', '推し', '睨ま', '出会っ', '漏る', '向き直る', '古る', 'ぬか', '明けれ', '追
う', 'うけ', '分っ', '帰る', '通そ', '泣か', '悲しむ', '擦る', '消す', '従っ', '過す', '割り出し', '接い', 'やれん', '消え失せ', '切り上げ', '犯さ', '障ら', '考え出す', '突く', '障', '計り', '途切れ', '祝し', '行う', '取
り寄せ', '打つ', '働け', '踏み付け', '営む', '絞め', '生ずる', '怒る', '発する', '信じ', 'あれ', '給え', '上ら', '集っ', '擦', '記し', 'のしかかる', '立ち上がる', 'せき込む', '直る', '勤まる', '務める', 'ならべる', '引き
裂い', '降っ', '経っ', '講じ', 'しくじっ', '為す', 'やって退け', '吹きかける', '抛り出さ', '込む', 'すん', '読み出し', 'られる', 'きかかっ', '愛する', '違い', 'さがし', 'ふかし', '力め', '計っ', '利く', 'ねぼけ', '追っか
ける', '已めろ', '参り', '角張っ', '押しつける', '腐っ', '喰わ', '突こ', 'さばけ', '踊ら', '懸かり', 'ぶら下げ', '立ち行か', '捧げる', 'たたく', '彩っ', '上る', '失し', '衝く', '噛ま', 'れれ', '詰め込む', '知れる', '近寄
り', '響け', '申し込ま', '後れ', 'もぐれ', '古ぼけ', '点じ', '詠ん', '見捨て', '揚げる', '仰ぎ', '震わせ', '隠れ', '顧み', '醒ます', '待ちかね', '施し', '問い返す', '使いこなせ', 'もぐり込ん', '初め', '這出', 'なん', '叱
りつける', '持ち上がる', '這入れ', '割る', 'つきつける', '洩らさ', '坐っ', 'むける', '抜け', 'がっ', '嫌っ', 'やられる', '臨め', '履い', '流れ込む', '喩える', '浴びる', 'かしこまる', '出さ', 'あらし', 'あやまる', 'てりゃ
', '伺う', '気が付い', '制さ', '変ずる', '書き散らし', '驚かさ', '考えつい', '認め', '棄て', '張る', '退', '翳し', '案ずる', 'やき', '慮', 'すくん', '押し寄せ', '呼ば', '坐せ', '気がつい', 'やむをえ', 'まがっ', 'あぶら', '取り極め', '焦る', '現われ', '勤め', '叱', '越え', '嘲',
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