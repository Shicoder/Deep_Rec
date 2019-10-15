#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
from pyspark import SparkContext, HiveContext, SQLContext,SparkConf

stopwords_list = [u")",u"(",u",", u"」", u"「", u"|", u"&", u"amp", u"·", u"【", u"】", u"/", u")", u"(", u"（", u"）", u"$", u"0",
                  u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"?", u"_", u"“", u"”", u"、", u"。", u"《", u"》",
                  u"一", u"一些", u"一何", u"一切", u"一则", u"一方面", u"一旦", u"一来", u"一样", u"一般", u"一转眼", u"万一", u"上", u"上下",
                  u"下", u"不", u"不仅", u"不但", u"不光", u"不单", u"不只", u"不外乎", u"不如", u"不妨", u"不尽", u"不尽然", u"不得", u"不怕",
                  u"不惟", u"不成", u"不拘", u"不料", u"不是", u"不比", u"不然", u"不特", u"不独", u"不管", u"不至于", u"不若", u"不论", u"不过",
                  u"不问", u"与", u"与其", u"与其说", u"与否", u"与此同时", u"且", u"且不说", u"且说", u"两者", u"个", u"个别", u"临", u"为",
                  u"为了", u"为什么", u"为何", u"为止", u"为此", u"为着", u"乃", u"乃至", u"乃至于", u"么", u"之", u"之一", u"之所以", u"之类",
                  u"乌乎", u"乎", u"乘", u"也", u"也好", u"也罢", u"了", u"二来", u"于", u"于是", u"于是乎", u"云云", u"云尔", u"些", u"亦",
                  u"人", u"人们", u"人家", u"什么", u"什么样", u"今", u"介于", u"仍", u"仍旧", u"从", u"从此", u"从而", u"他", u"他人", u"他们",
                  u"以", u"以上", u"以为", u"以便", u"以免", u"以及", u"以故", u"以期", u"以来", u"以至", u"以至于", u"以致", u"们", u"任", u"任何",
                  u"任凭", u"似的", u"但", u"但凡", u"但是", u"何", u"何以", u"何况", u"何处", u"何时", u"余外", u"作为", u"你", u"你们", u"使",
                  u"使得", u"例如", u"依", u"依据", u"依照", u"便于", u"俺", u"俺们", u"倘", u"倘使", u"倘或", u"倘然", u"倘若", u"借", u"假使",
                  u"假如", u"假若", u"傥然", u"像", u"儿", u"先不先", u"光是", u"全体", u"全部", u"兮", u"关于", u"其", u"其一", u"其中", u"其二",
                  u"其他", u"其余", u"其它", u"其次", u"具体地说", u"具体说来", u"兼之", u"内", u"再", u"再其次", u"再则", u"再有", u"再者", u"再者说",
                  u"再说", u"冒", u"冲", u"况且", u"几", u"几时", u"凡", u"凡是", u"凭", u"凭借", u"出于", u"出来", u"分别", u"则", u"则甚",
                  u"别", u"别人", u"别处", u"别是", u"别的", u"别管", u"别说", u"到", u"前后", u"前此", u"前者", u"加之", u"加以", u"即", u"即令",
                  u"即使", u"即便", u"即如", u"即或", u"即若", u"却", u"去", u"又", u"又及", u"及", u"及其", u"及至", u"反之", u"反而", u"反过来",
                  u"反过来说", u"受到", u"另", u"另一方面", u"另外", u"另悉", u"只", u"只当", u"只怕", u"只是", u"只有", u"只消", u"只要", u"只限",
                  u"叫", u"叮咚", u"可", u"可以", u"可是", u"可见", u"各", u"各个", u"各位", u"各种", u"各自", u"同", u"同时", u"后", u"后者",
                  u"向", u"向使", u"向着", u"吓", u"吗", u"否则", u"吧", u"吧哒", u"吱", u"呀", u"呃", u"呕", u"呗", u"呜", u"呜呼", u"呢",
                  u"呵", u"呵呵", u"呸", u"呼哧", u"咋", u"和", u"咚", u"咦", u"咧", u"咱", u"咱们", u"咳", u"哇", u"哈", u"哈哈", u"哉",
                  u"哎", u"哎呀", u"哎哟", u"哗", u"哟", u"哦", u"哩", u"哪", u"哪个", u"哪些", u"哪儿", u"哪天", u"哪年", u"哪怕", u"哪样",
                  u"哪边", u"哪里", u"哼", u"哼唷", u"唉", u"唯有", u"啊", u"啐", u"啥", u"啦", u"啪达", u"啷当", u"喂", u"喏", u"喔唷", u"喽",
                  u"嗡", u"嗡嗡", u"嗬", u"嗯", u"嗳", u"嘎", u"嘎登", u"嘘", u"嘛", u"嘻", u"嘿", u"嘿嘿", u"因", u"因为", u"因了", u"因此",
                  u"因着", u"因而", u"固然", u"在", u"在下", u"在于", u"地", u"基于", u"处在", u"多", u"多么", u"多少", u"大", u"大家", u"她",
                  u"她们", u"好", u"如", u"如上", u"如上所述", u"如下", u"如何", u"如其", u"如同", u"如是", u"如果", u"如此", u"如若", u"始而",
                  u"孰料", u"孰知", u"宁", u"宁可", u"宁愿", u"宁肯", u"它", u"它们", u"对", u"对于", u"对待", u"对方", u"对比", u"将", u"小",
                  u"尔", u"尔后", u"尔尔", u"尚且", u"就", u"就是", u"就是了", u"就是说", u"就算", u"就要", u"尽", u"尽管", u"尽管如此", u"岂但",
                  u"己", u"已", u"已矣", u"巴", u"巴巴", u"并", u"并且", u"并非", u"庶乎", u"庶几", u"开外", u"开始", u"归", u"归齐", u"当",
                  u"当地", u"当然", u"当着", u"彼", u"彼时", u"彼此", u"往", u"待", u"很", u"得", u"得了", u"怎", u"怎么", u"怎么办", u"怎么样",
                  u"怎奈", u"怎样", u"总之", u"总的来看", u"总的来说", u"总的说来", u"总而言之", u"恰恰相反", u"您", u"惟其", u"慢说", u"我", u"我们",
                  u"或", u"或则", u"或是", u"或曰", u"或者", u"截至", u"所", u"所以", u"所在", u"所幸", u"所有", u"才", u"才能", u"打", u"打从",
                  u"把", u"抑或", u"拿", u"按", u"按照", u"换句话说", u"换言之", u"据", u"据此", u"接着", u"故", u"故此", u"故而", u"旁人", u"无",
                  u"无宁", u"无论", u"既", u"既往", u"既是", u"既然", u"时候", u"是", u"是以", u"是的", u"曾", u"替", u"替代", u"最", u"有",
                  u"有些", u"有关", u"有及", u"有时", u"有的", u"望", u"朝", u"朝着", u"本", u"本人", u"本地", u"本着", u"本身", u"来", u"来着",
                  u"来自", u"来说", u"极了", u"果然", u"果真", u"某", u"某个", u"某些", u"某某", u"根据", u"欤", u"正值", u"正如", u"正巧", u"正是",
                  u"此", u"此地", u"此处", u"此外", u"此时", u"此次", u"此间", u"毋宁", u"每", u"每当", u"比", u"比及", u"比如", u"比方", u"没奈何",
                  u"沿", u"沿着", u"漫说", u"焉", u"然则", u"然后", u"然而", u"照", u"照着", u"犹且", u"犹自", u"甚且", u"甚么", u"甚或", u"甚而",
                  u"甚至", u"甚至于", u"用", u"用来", u"由", u"由于", u"由是", u"由此", u"由此可见", u"的", u"的确", u"的话", u"直到", u"相对而言",
                  u"省得", u"看", u"眨眼", u"着", u"着呢", u"矣", u"矣乎", u"矣哉", u"离", u"竟而", u"第", u"等", u"等到", u"等等", u"简言之",
                  u"管", u"类如", u"紧接着", u"纵", u"纵令", u"纵使", u"纵然", u"经", u"经过", u"结果", u"给", u"继之", u"继后", u"继而",
                  u"综上所述", u"罢了", u"者", u"而", u"而且", u"而况", u"而后", u"而外", u"而已", u"而是", u"而言", u"能", u"能否", u"腾", u"自",
                  u"自个儿", u"自从", u"自各儿", u"自后", u"自家", u"自己", u"自打", u"自身", u"至", u"至于", u"至今", u"至若", u"致", u"般的",
                  u"若", u"若夫", u"若是", u"若果", u"若非", u"莫不然", u"莫如", u"莫若", u"虽", u"虽则", u"虽然", u"虽说", u"被", u"要", u"要不",
                  u"要不是", u"要不然", u"要么", u"要是", u"譬喻", u"譬如", u"让", u"许多", u"论", u"设使", u"设或", u"设若", u"诚如", u"诚然",
                  u"该", u"说来", u"诸", u"诸位", u"诸如", u"谁", u"谁人", u"谁料", u"谁知", u"贼死", u"赖以", u"赶", u"起", u"起见", u"趁",
                  u"趁着", u"越是", u"距", u"跟", u"较", u"较之", u"边", u"过", u"还", u"还是", u"还有", u"还要", u"这", u"这一来", u"这个",
                  u"这么", u"这么些", u"这么样", u"这么点儿", u"这些", u"这会儿", u"这儿", u"这就是说", u"这时", u"这样", u"这次", u"这般", u"这边",
                  u"这里", u"进而", u"连", u"连同", u"逐步", u"通过", u"遵循", u"遵照", u"那", u"那个", u"那么", u"那么些", u"那么样", u"那些",
                  u"那会儿", u"那儿", u"那时", u"那样", u"那般", u"那边", u"那里", u"都", u"鄙人", u"鉴于", u"针对", u"阿", u"除", u"除了", u"除外",
                  u"除开", u"除此之外", u"除非", u"随", u"随后", u"随时", u"随着", u"难道说", u"非但", u"非徒", u"非特", u"非独", u"靠", u"顺",
                  u"顺着", u"首先", u"！", u"，", u"：", u"；", u"？",
                  u"测试",u":",u"!",u"~",u"、",u"ml",u"5折",u"%",u"/",u"cm",u"★",u"*",u"〖",u"",u" ",u"-",u"+",u"'发货'",u"'下单'",u'.'
                  ]
def do_sql_job(ssc, sql, tmp_table=None, target_table=None, is_show=False, recorder=None, coal_num=32, is_par=True):
    print(u"executing sql:")
    print(sql)
    tmp_df = ssc.sql(sql)
    if is_show:
        print(u"sql result:")
        # safe_show(tmp_df)
    if tmp_table:
        print(u"register to tmp table:{table}".format(table=tmp_table))
        # tmp_df = tmp_df.coalesce(coal_num)
        tmp_df.registerTempTable(tmp_table)
        print(u"tmp table size: {} ".format(tmp_df.count()))

    if target_table:
        print(u"write into table {target_table} with par={is_par}"
                   .format(target_table=target_table, is_par=is_par))
        recorder.record2table(tmp_table, target_table, is_par)
    print(u"sql job finished")
    return tmp_df
def get_sc(env, job_name,jar_name=None):
    if env == u'dev':
        spark_conf = SparkConf().\
            setAppName(job_name).\
            set(u"spark.jars",jar_name )
        # sc = SparkContext(u'local[4]',appName=job_name)
        sc = SparkContext("local[4]",conf=spark_conf)
    else:
        spark_conf = SparkConf().\
            setAppName(job_name).\
            set(u"spark.driver.maxResultSize",u'5g')
        sc = SparkContext(appName=job_name)
    return sc
def get_ssc(env, job_name,jar_name=None):
    sc = get_sc(env, job_name,jar_name)
    if env == u'dev':
        ssc = SQLContext(sc)
    else:
        ssc = HiveContext(sc)
    return sc, ssc
def get_dataframe(env, ssc,schema,sql_or_path=None,nrows=None,header=None):
    if env == u'online':
        sql = sql_or_path
        # sql = u"select goods_id, fx_num_30 from dm_ai.fx_gy_goods_profile_d where par='{the_day}'".format(
        #     the_day=the_day)
        return do_sql_job(ssc, sql)
    if env == u'dev':
        if schema != None:
            item_df = (ssc.read.schema(schema).format("com.databricks.spark.csv").options(header="true")
                   .load(sql_or_path))
        else:
            item_df = ssc.read.csv(sql_or_path,header=True)
        # item_df = pd.read_csv(sql_or_path,nrows=nrows,header=header)
        # item_df = ssc.createDataFrame(item_df)
        # print(item_df.dtypes)
        # null view add schema
        #https://stackoverflow.com/questions/38794522/pandas-dataframe-to-spark-dataframe-can-not-merge-type-error


        # item_df = ssc.read.parquet(sql_or_path)
        return item_df
def write2table(sc, ssc, result, dist_table_name, the_day=None):
    # eval_rdd = sc.parallelize(eval_rs_list)
    # eval_df = ssc.createDataFrame(eval_rdd)
    # result_df[['kdt_id','gy_goods_id','gy_kdt_id','prob']].show()
    tmp_table_name = u"result_df"
    result.registerTempTable(tmp_table_name)
    print(u"write to dist table: {}".format(dist_table_name))
    if the_day == None:
        sql = u'''
            insert overwrite table {table_name}
            select * from {src_table}
            '''.format(table_name=dist_table_name, src_table=tmp_table_name)
    else :
        sql = u'''
            insert overwrite table {table_name} partition (par = '{par}')
            select * from {src_table}
            '''.format(table_name=dist_table_name, par=the_day, src_table=tmp_table_name)
    return do_sql_job(ssc, sql)