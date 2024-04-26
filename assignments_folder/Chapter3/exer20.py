"""
言語処理100本ノック 第3章課題

20. JSONデータの読み込み
Wikipedia記事のJSONファイルを読み込み，「イギリス」に関する記事本文を表示せよ．問題21-29では，ここで抽出した記事本文に対して実行せよ．

"""
import json

# 初期化
folderpath = "./assignments_folder/Chapter3/"
filename = "jawiki-country.json"

# JSONファイル読み込み
with open(folderpath + filename, "r", encoding="utf-8") as f:
    data_list = f.readlines()
    article_list = [json.loads(data) for data in data_list] # 複数のJSON形式の表記があるため，それぞれリスト化する。

# イギリスに関する記事JSONファイルの抽出
    UK_article = list(filter(lambda x: x["title"] == "イギリス", article_list))[0]

# 出力
print(UK_article)

"""
＊　出力結果　＊
{'title': 'イギリス', 'text': '{{redirect|UK}}\n{{redirect|英国|春秋時代の諸侯国|英 (春秋)}}\n{{Otheruses|ヨーロッパの国|長崎県・熊本県の郷土料理|いぎりす}}\n{{基礎情報 国\n|略名  =イギリス\n|日本語国名 = グレートブリテ
ン及び北アイルランド連合王国\n|公式国名 = {{lang|en|United Kingdom of Great Britain and Northern Ireland}}<ref>英語以外での正式国名:<br />\n*{{lang|gd|An Rìoghachd Aonaichte na Breatainn Mhòr agus Eirinn mu Thuath}}（[[ 
スコットランド・ゲール語]]）\n*{{lang|cy|Teyrnas Gyfunol Prydain Fawr a Gogledd Iwerddon}}（[[ウェールズ語]]）\n*{{lang|ga|Ríocht Aontaithe na Breataine Móire agus Tuaisceart na hÉireann}}（[[アイルランド語]]）\n*{{lang|kw|An Rywvaneth Unys a Vreten Veur hag Iwerdhon Glédh}}（[[コーンウォール語]]）\n*{{lang|sco|Unitit Kinrick o Great Breetain an Northren Ireland}}（[[スコットランド語]]）\n**{{lang|sco|Claught Kängrick o Docht Brätain an Norlin Airlann}}、{{lang|sco|Unitet Kängdom o Great Brittain an Norlin Airlann}}（アルスター・スコットランド語）</ref>\n|国旗画像 = Flag of the United Kingdom.svg\n|国章画像 = [[ファイル:Royal Coat of Arms of the United Kingdom.svg|85px|イギリスの国章]]\n|国章リンク =（[[イギリスの国章|国章]]）\n|標語 = {{lang|fr|[[Dieu et mon droit]]}}<br />（[[フランス語]]:[[Dieu et mon droit|神と我が権利]]）\n|国歌 = [[女王陛下万歳|{{lang|en|God Save the Queen}}]]{{en icon}}<br />\'\'神よ女王を護り賜え\'\'<br />{{center|[[ファイル:United States Navy Band - God Save the Queen.ogg]]}}\n|地図画像 = Europe-UK.svg\n|位置画像 = United Kingdom (+overseas territories) in the World (+Antarctica claims).svg\n|公用語 = [[英語]]\n|首都 = [[ロンドン]]（事実上）\n|最大都市 = ロンドン\n|元首等肩書 = [[イギリスの君主|女王]]\n|元首等氏名 = [[エリザベス2世]]\n|首相等肩書 = [[イギリスの首相|首相]]\n|首相等氏名 = [[ボリス・ジョンソン]]\n|他元首等肩書1 = [[貴族院 (イギリス)|貴族院議長]]\n|他元首等氏名1 = [[:en:Norman Fowler, Baron Fowler|ノーマン・ファウラー]]\n|他元首等肩書2 = [[庶民院 (イギリス)|庶民院議長]]\n|他元
首等氏名2 = {{仮リンク|リンゼイ・ホイル|en|Lindsay Hoyle}}\n|他元首等肩書3 = [[連合王国最高裁判所|最高裁判所長官]]\n|他元首等氏名3 = [[:en:Brenda Hale, Baroness Hale of Richmond|ブレンダ・ヘイル]]\n|面積順位 = 76\n|面積 
大きさ = 1 E11\n|面積値 = 244,820\n|水面積率 = 1.3%\n|人口統計年 = 2018\n|人口順位 = 22\n|人口大きさ = 1 E7\n|人口値 = 6643万5600<ref>{{Cite web|url=https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates|title=Population estimates - Office for National Statistics|accessdate=2019-06-26|date=2019-06-26}}</ref>\n|人口密度値 = 271\n|GDP統計年元 = 2012\n|GDP値元 = 1兆5478億<ref name="imf-statistics-gdp">[http://www.imf.org/external/pubs/ft/weo/2012/02/weodata/weorept.aspx?pr.x=70&pr.y=13&sy=2010&ey=2012&scsm=1&ssd=1&sort=country&ds=.&br=1&c=112&s=NGDP%2CNGDPD%2CPPPGDP%2CPPPPC&grp=0&a=IMF>Data and Statistics>World Economic Outlook Databases>By Countrise>United Kingdom]</ref>\n|GDP統計年MER = 2012\n|GDP順位MER = 6\n|GDP値MER = 2兆4337億<ref name="imf-statistics-gdp" />\n|GDP統計年 = 2012\n|GDP順位 = 6\n|GDP値 = 2兆3162億<ref name="imf-statistics-gdp" />\n|GDP/人 = 36,727<ref name="imf-statistics-gdp" />\n|建国形態 = 建国\n|確立形態1 = [[イングランド王国]]／[[スコットランド王国]]<br />（両国とも[[合同法 (1707年)|1707年合同法]]まで）\n|確立年月日1 = 927年／843年\n|確立形態2 = [[グレートブリテン王国]]成立<br />（1707年合同法）\n|確立年月日2 = 1707年{{0}}5月{{0}}1日\n|確立形態3 = [[グレートブリテン及びアイルランド連合王国]]成立<br />（[[合同法 (1800年)|1800年合同法]]）\n|確立年月日3 = 1801年{{0}}1月{{0}}1日\n|確立形態4 = 現在の国号「\'\'\'グレートブリテン及び北アイルランド連合王国\'\'\'」に変更\n|確立年月日4 = 1927年{{0}}4月12日\n|通貨 = [[スターリング・ポンド|UKポンド]] (£)\n|通貨コー 
ド = GBP\n|時間帯 = ±0\n|夏時間 = +1\n|ISO 3166-1 = GB / GBR\n|ccTLD = [[.uk]] / [[.gb]]<ref>使用は.ukに比べ圧倒的少数。</ref>\n|国際電話番号 = 44\n|注記 = <references/>\n}}\n\n\'\'\'グレートブリテン及び北アイルランド連 
合王国\'\'\'（グレートブリテンおよびきたアイルランドれんごうおうこく、{{lang-en-short|United Kingdom of Great Britain and Northern Ireland}}: \'\'\'UK\'\'\'）は、[[ヨーロッパ大陸]]の北西岸に位置し、[[グレートブリテン島]]・[[アイルランド島]]北東部・その他多くの島々から成る[[立憲君主制]][[国家]]。首都は[[ロンドン]]。[[日本語]]における[[通称]]の一例として\'\'\'イギリス\'\'\'、\'\'\'英国\'\'\'（えいこく）がある（→[[#国名]]）。\n\n\'\'\'[[イ
ングランド]]\'\'\'、\'\'\'[[ウェールズ]]\'\'\'、\'\'\'[[スコットランド]]\'\'\'、\'\'\'[[北アイルランド]]\'\'\'という歴史的経緯に基づく4つの[[イギリスのカントリー|「カントリー」と呼ばれる「国」]]が、[[同君連合]]型の単一の
[[主権国家体制|主権国家]]を形成する<ref name="page823">{{cite web |url=http://webarchive.nationalarchives.gov.uk/+/http://www.number10.gov.uk/Page823 |title=Countries within a country |publisher=Prime Minister\'s Office 
|accessdate=10 January 2003}}</ref>独特の統治体制を採るが、一般的に[[連邦]]国家とは区別される。\n\n[[国際連合安全保障理事会常任理事国]]の一国（五大国）であり、[[G7]]・[[G20]]に参加する。GDPは世界10位以内に位置する巨大な 
市場を持ち、ヨーロッパにおける四つの大国「[[ビッグ4 (ヨーロッパ)|ビッグ4]]」の一国である。[[ウィーン体制]]が成立した[[1815年]]以来、世界で最も影響力のある国家・[[列強]]の一つに数えられる。また、[[民主主義]]、[[立憲君主制
]]など近代国家の基本的な諸制度が発祥した国でもある。\n\nイギリスの[[擬人化]]としては[[ジョン・ブル]]、[[ブリタニア (女神)|ブリタニア]]が知られる。\n\n==国名==\n正式名称は英語で「{{Lang|en|\'\'\'United Kingdom of Great Britain and Northern Ireland\'\'\'}}（ユナイテッド・キングダム・オヴ・グレイト・ブリテン・アンド・ノーザン・アイルランド）」であり、日本語では、「\'\'\'グレート・ブリテン及び北部アイルランド連合王国\'\'\'」とする場合（法文
など）と「\'\'\'グレート・ブリテン及び北アイルランド連合王国\'\'\'」とする場合（条約文など）がある。\n\n英語での略称は「{{Lang|en|\'\'\'United Kingdom\'\'\'}}」、「{{Lang|en|\'\'\'UK\'\'\'}}」。[[日本語]]における一般的な
通称は「\'\'\'イギリス\'\'\'」もしくは「\'\'\'英国\'\'\'」であるが、稀に「{{Lang|en|United Kingdom}}」の[[直訳と意訳|直訳]]である「\'\'\'[[連合王国]]\'\'\'（れんごうおうこく）」が用いられることもある。現在の公用文では「 
英国」が使用されており、「イギリス」は口語で用いられることが多い<ref>[[日本放送協会|NHK]]で採用している他、原則として「英国」を用いるメディアでも「[[イギリス英語]]」のような形では使用する。</ref>。「連合王国」は2003年ま 
で法文において用いられていた<ref>[http://warp.da.ndl.go.jp/info:ndljp/pid/1368617/www.meti.go.jp/policy/anpo/moto/topics/country/country.pdf 輸出貿易管理令等における国名表記の変更について]（[[経済産業省]]） 国立国会図書 
館のアーカイブより\'\'2019-2-5閲覧\'\'</ref>。\n\n「イギリス」は、[[ポルトガル語]]で[[イングランド]]を指す「{{Lang|pt|Inglez}}（イングレス）」が語源で、戦国時代にポルトガル人が来航した事に起源を持つ。原義にかかわらず連合
王国全体を指して使われており、連合王国の構成体たる「イングランド」とは区別される。[[江戸時代]]には、[[オランダ語]]の「{{Lang|nl|Engelsch}}（エングルシュ）」を語源とする「\'\'\'エゲレス\'\'\'」という呼称も広く使用された<ref>[https://kotobank.jp/word/%E3%82%A8%E3%82%B2%E3%83%AC%E3%82%B9-444373 コトバンク「エゲレス」]</ref>。[[幕末]]から[[明治]]・[[大正]]期には「\'\'\'英吉利\'\'\'（えいぎりす）」や「大不列顛（だいふれつてん、大ブリテン）」
と[[国名の漢字表記一覧|漢字で表記]]されることもあったが、前者が「英国」という略称の語源である。ただし「英国」は、狭義に連合王国全体でなくイングランド（\'\'\'英格蘭\'\'\'）のみを指す場合もある<ref>また、[[アメリカ合衆国]]に渡ることを「渡米」と言うように、イギリス、特にイングランドへ渡ることを「渡英」と言う（[[二字熟語による往来表現の一覧]]を参照）。</ref>。\n\n[[合同法 (1707年)|1707年合同法]]においては、[[イングランド王国]]および[[スコッ
トランド王国]]を一王国に統合すると宣言する。同法において、新国家名称は「[[グレートブリテン王国]]」または「グレートブリテン連合王国」および「連合王国」とすると述べている<ref>{{cite web |url=http://www.scotshistoryonline.co.uk/union.html |title=Treaty of Union, 1706 |publisher=Scots History Online |accessdate=23 August 2011}}</ref><ref>{{cite book |url=http://books.google.com/?id=LYc1tSYonrQC&pg=PA165 |title=Constitutional & Administrative Law |page=165 |author=Barnett, Hilaire |author2=Jago, Robert |edition=8th |year=2011 |isbn=978-0-415-56301-7 |publisher=Routledge |location=Abingdon }}</ref>。しかしながら、「連合王国」という用語は18世紀における非公式 
の使用にのみ見られ、「長文式」でない単なる「グレート・ブリテン」であった1707年から1800年まで、同国はごくまれに正式名称である「グレート・ブリテン連合王国」と言及された<ref>See [[s:Act of Union 1707#Article 1 (name of the 
new kingdom)|Article One]] of the Act of Union 1707.</ref><ref name=name>"After the political union of England and Scotland in 1707, the nation\'s official name became \'Great Britain\'", \'\'The American Pageant, Volume 1\'\', Cengage Learning (2012)</ref><ref name=name2>"From 1707 until 1801 \'\'Great Britain\'\' was the official designation of the kingdoms of England and Scotland". \'\'The Standard Reference Work:For the Home, School and Library, Volume 3\'\', Harold Melvin Stanford (1921)</ref><ref name=name3>"In 1707, on the union with Scotland, \'Great Britain\' became the official name of the British Kingdom, and so continued until the union with Ireland in 1801". \'\'United States Congressional serial set, Issue 10;Issue 3265\'\' (1895)</ref><ref>{{cite web |url=http://www.historyworld.net/wrldhis/PlainTextHistories.asp?historyid=ab07 |title=History of Great Britain (from 1707) |authorlink=w:Bamber Gascoigne |author=Gascoigne, Bamber |publisher=History World |accessdate=18 July 2011}}</ref>。[[合同法 (1800年)|1800年合同法]]では、1801年にグレート・ブリテン王国と[[アイルランド 
王国]]が統合し、[[グレート・ブリテン及びアイルランド連合王国]]が成立した。現在の正式国名である「グレート・ブリテン及び北（部）アイルランド連合王国」は、[[北アイルランド]]のみが連合王国の一部としてとどまった1922年の[[アイ
ルランド自由国]]独立および{{仮リンク|アイルランド分裂|en|Partition of Ireland}}後に採用された<ref>{{cite book |title=The Irish Civil War 1922–23 |author=Cottrell, P. |year=2008 |page=85 |isbn=1-84603-270-9}}</ref>。\n\n 
イギリスは主権国家として国であるが、イングランド、[[スコットランド]]、[[ウェールズ]]、それほどの段階ではないが北アイルランドも、主権国家ではないが[[イギリスのカントリー|「国」（country）]]と呼ばれる<ref name="alphabeticalNI">{{citation |author1=S. Dunn |author2=H. Dawson|year=2000 |title=An Alphabetical Listing of Word, Name and Place in Northern Ireland and the Living Language of Conflict |publisher=Edwin Mellen Press |location=Lampeter |quote=One specific problem&nbsp;— in both general and particular senses&nbsp;— is to know what to call Northern Ireland itself:in the general sense, it is not a country, or a province, or a state&nbsp;— although some 
refer to it contemptuously as a statelet:the least controversial word appears to be jurisdiction, but this might change.}}</ref><ref>{{cite web |url=http://www.iso.org/iso/iso_3166-2_newsletter_ii-3_2011-12-13.pdf |title=Changes in the list of subdivision names and code elements |work=ISO 3166-2 |publisher=International Organization for Standardization |date=15 December 2011 |accessdate=28 May 2012}}</ref>。スコットランド、ウェールズ、 
北アイルランドは、権限の委譲による自治権を有する<ref>[http://books.google.com/?id=gPkDAQAAIAAJ Population Trends, Issues 75–82, p.38], 1994, UK Office of Population Censuses and Surveys</ref><ref name="citizenship">[http://books.google.com/?id=2u8rD6F-yg0C&pg=PA7 Life in the United Kingdom:a journey to citizenship, p. 7], United Kingdom Home Office, 2007, ISBN 978-0-11-341313-3.</ref>。イギリス首相のウェブサイトでは、連合王国の説明とし 
て「1国内の国々」という言葉が用いられていた<ref name="page823"/>。{{仮リンク|イギリスの12のNUTS1地域|en|NUTS of the United Kingdom}}統計のような複数の統計的概要において、スコットランド、ウェールズ、北アイルランドを「region」と言及している<ref>{{cite web |url=http://www.ons.gov.uk/ons/dcp171778_346117.xml |title=Statistical bulletin:Regional Labour Market Statistics |accessdate=5 March 2014 |archiveurl=https://web.archive.org/web/20141224045523/http://www.ons.gov.uk/ons/dcp171778_346117.xml |archivedate=2014年12月24日 |deadlinkdate=2018年3月 }}</ref><ref>{{cite web |url=http://www.gmb.org.uk/newsroom/fall-in-earnings-value-during-recession |title=13.4% 
Fall In Earnings Value During Recession |accessdate=5 March 2014}}</ref>。北アイルランドは「province」とも言及される<ref name="alphabeticalNI"/><ref name="placeApart">{{cite book |author=Murphy, Dervla |title=A Place Apart |year=1979 |publisher=Penguin |place=London |isbn=978-0-14-005030-1}}</ref>。北アイルランドに関しては、記述名の使用が「多くの場合、個人の政治的選好を明らかにする選択で議論の的になり得る」<ref>{{Cite book |last1=Whyte 
|first1=John |authorlink1=w:John Henry Whyte |last2=FitzGerald |first2=Garret|authorlink2=w:Garret FitzGerald|year=1991 |title=Interpreting Northern Ireland |location=Oxford |publisher=Clarendon Press |isbn=978-0-1 
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