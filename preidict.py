# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:28:41 2020

@author: cm
"""

from networks import SentimentAnalysis
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import sklearn.metrics as ms


SA = SentimentAnalysis()


def predict(sent):
    """
    1: positif
    0: neutral
    -1: negatif
    """
    score1, score0 = SA.normalization_score(sent)
    # print(score1, score0)
    if score1 == score0:
        result = 0
    elif score1 > score0:
        result = 1
    elif score1 < score0:
        result = -1
    return result, score1, score0


def sentiment_tagging(data_df):
    pred = []
    scores_pos_all, scores_neg_all = [], []
    for i in tqdm(range(len(data_df))):
        try:
            text = data_df['评论详情'].iloc[i]
            sent_tag, scores_pos, scores_neg = predict(text)
            pred.append(sent_tag)
            scores_pos_all.append(scores_pos)
            scores_neg_all.append(scores_neg)
        except:
            print('error')
            pred.append(0)    
            scores_pos_all.append(None)
            scores_neg_all.append(None)
    data_df['sentiment_tag_dict'] = pred
    data_df['scores_pos_dict'] = scores_pos_all
    data_df['scores_neg_dict'] = scores_neg_all

    return data_df


def distribution_plot_TextLength(mark_lens):
    # 评论长度分布图：用于确定 过长文本 and 分词的 max_len
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats
    sns.distplot(mark_lens, fit=stats.norm, color='g')  # 正太概率密度 / 核密度估计图
    plt.tick_params(labelsize=20)
    plt.show()


if __name__ =='__main__':
    # text = '说实话，我觉得这里的卫生很差，位置难道不偏僻？，我难道会喜欢这种酒店？'

    # SA.normalization_score(text)

    # path = 'D:/competition/酒店数据情感分类项目/data_raw/'
    # hotel_mark = pd.read_excel(path+'ChnSentiCorp_htl_all.xlsx')
    # hotel_mark.head()
    # pred = []

    path = 'D:/competition/泰迪杯_C/'
    hotel_mark_tag = pd.read_excel(open(path+'data_raw/hotel_mark_tag.xlsx', 'rb'), index_col=0)
    scenic_mark_tag = pd.read_excel(open(path+'data_raw/scenic_mark_tag.xlsx', 'rb'), index_col=0)

    ''' T4特色分析 '''
    import pickle
    # scenic 
    df_topsis_scenic = pd.read_excel(open(path+'doc/T4/熵权-tposis_scenic.xlsx', 'rb'), index_col=0)
    high = df_topsis_scenic.index[np.where(df_topsis_scenic['C'] >= 0.5)[0]]
    median = df_topsis_scenic.index[np.where((df_topsis_scenic['C'] >= 0.3) & (df_topsis_scenic['C'] < 0.5))[0]]
    low = df_topsis_scenic.index[np.where(df_topsis_scenic['C'] < 0.3)[0]]

    # 酒店
    df_topsis_hotel = pd.read_excel(open(path+'doc/T4/熵权-tposis_hotel.xlsx', 'rb'), index_col=0)
    high = df_topsis_hotel.index[np.where(df_topsis_hotel['C'] >= 0.5)[0]]
    median = df_topsis_hotel.index[np.where((df_topsis_hotel['C'] >= 0.2) & (df_topsis_hotel['C'] < 0.5))[0]]
    low = df_topsis_hotel.index[np.where(df_topsis_hotel['C'] < 0.2)[0]]

    from gensim.corpora import Dictionary
    from gensim import corpora
    import jieba.posseg as pseg 
    import jieba

    data_all = scenic_mark_tag
    d_name = '景区名称'
    data_all = hotel_mark_tag
    d_name = '酒店名称'


    object_k = high[2]

    data_k = data_all.iloc[np.where(data_all[d_name] == object_k)[0], :]

    # 热词提取
    import re
    dict_hotel_5 = {'服务':['服务', '前台','员工','人员', '态度','热情','升级','升房','经理'],
              '位置':['交通', '位置', '距离', '地铁', '近', '远', '商场','地理位置','周边','附近',
              '便利','出行','中心区','地段','视野'],
              '卫生':['环境', '卫生', '干净', '整洁','霉味'],
              '设施':['设施', '泳池','健身', '热水', '床', '停车场', '暖气', '空调', '隔音', '安静',
              '舒适','装修','旧','餐厅','浴室'],
              '性价比':['价格', '性价比', '贵', '便宜','免费','优惠','实惠']}
    
   
    def fun_sentimen_compute(dict_hotel_5, asp, object_k, d_name='酒店名称'):
        data_k = data_all.iloc[np.where(data_all[d_name] == object_k)[0], :]
        words_text = []
        for i in data_k.index:
            text = data_k.loc[i, '评论详情']
            text_small = re.split(r'[。！；？， ]', text)
            for t in text_small:
                for w in dict_hotel_5[asp]:
                    if w in t:
                        words_text.append(t)
                        continue
        words_text = list(set(words_text))
        tag_all = []
        for t in words_text:
            tag, _, _ = predict(t)
            tag_all.append(tag)
        tag_all = np.array(tag_all)
        pos = len(np.where(tag_all==1)[0]) / len(tag_all)
        neu = len(np.where(tag_all==0)[0]) / len(tag_all)
        neg = len(np.where(tag_all==-1)[0]) / len(tag_all)
        print(pos, neu, neg)
        return pos, neu, neg, words_text

    def sentiment_socres_compute(level_, d_name):
        df_high_hotel = pd.DataFrame(index=level_, columns=dict_hotel_5.keys())
        texts_all_hotel = {}
        for object_k in level_:
            texts_all_hotel.get(object_k, 0)
            asp_dict = {}
            for k in dict_hotel_5.keys():
                asp_dict.get(k, 0)
                pos, neu, neg, text_all = fun_sentimen_compute(dict_hotel_5, asp=k, object_k=object_k, d_name=d_name)
                asp_dict[k] = text_all
                df_high_hotel.loc[object_k, k] = [pos, neu, neg]
            texts_all_hotel[object_k] = asp_dict

        return df_high_hotel, texts_all_hotel

    df_high_hotel, texts_all_high_hotel = sentiment_socres_compute(high, '酒店名称')
    df_high_hotel.to_excel(open(path+'doc/T4/df_high_hotel.xlsx', 'wb'))
    pickle.dump(texts_all_high_hotel, open(path+'doc/T4/texts_all_high_hotel.pkl', 'wb'))
    
    df_median_hotel, texts_all_median_hotel = sentiment_socres_compute(median, '酒店名称')
    df_median_hotel.to_excel(open(path+'doc/T4/df_median_hotel.xlsx', 'wb'))
    pickle.dump(texts_all_median_hotel, open(path+'doc/T4/texts_all_median_hotel.pkl', 'wb'))

    df_low_hotel, texts_all_low_hotel = sentiment_socres_compute(low, '酒店名称')
    df_low_hotel.to_excel(open(path+'doc/T4/df_low_hotel.xlsx', 'wb'))
    pickle.dump(texts_all_low_hotel, open(path+'doc/T4/texts_all_low_hotel.pkl', 'wb'))

    '''' 词云 5-asp '''

    import jieba
    import csv
    # load 常用停用词表
    stop_words = pd.read_csv('D:/VSCode/pyStudy/NLP/stopwords/hit_stopwords.txt', 
                sep='\t', header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')
    stop_words = list(stop_words.iloc[:,0])
    
    # 13/20/35
    # 低11/38/39
    # 中 2/4/49
    def wordcloud_plot(doc_all):
        myfont = 'D:/VSCode/pyStudy/NLP/simsun.ttc'
        import wordcloud
        from PIL import Image
        import matplotlib.pyplot as plt
        mask = np.array(Image.open(path+'data_raw/pic_cloud.jfif'))
        # doc_all = [[w for w in sen if w != '很'] for sen in doc_all]
        # data_cloud = ' '.join([' '.join(sen) for sen in np.array(doc_all)])
        data_cloud = ' '.join(doc_all)
        # data_cloud = ' '.join([' '.join(sen) for sen in np.array(doc_all)])
        cloudobj = wordcloud.WordCloud(font_path=myfont,
                        width=1200, height=800,
                        mode="RGBA", background_color=None,
                        mask=mask,
                        stopwords=stop_words).generate(data_cloud)
        plt.imshow(cloudobj)
        plt.axis('off') 
        plt.show()


    texts_all_high_hotel = pickle.load(open(path+'doc/T4/texts_all_high_hotel.pkl', 'rb'))
    texts_all_median_hotel = pickle.load(open(path+'doc/T4/texts_all_median_hotel.pkl', 'rb'))
    texts_all_low_hotel = pickle.load(open(path+'doc/T4/texts_all_low_hotel.pkl', 'rb'))

    k = '服务'
    high
    for k in texts_all_high_hotel['H20'].keys():
        text_k = texts_all_high_hotel['H13'][k]
        # doc_k_all = [[w for w in jieba.lcut(sen) if w not in stop_words] for sen in text_k]
        wordcloud_plot(text_k)

    # 频率计算
    def get_freq(text_k, search_words_list):
        words_freq = []
        for w in search_words_list:
            w_list = list(w)
            freq = 0
            for text in text_k:
                count = 0
                for s in w_list:
                    if s in text:
                        count += 1
                    else:
                        continue
                if count == len(w_list):
                    freq += 1
            words_freq.append((w, freq))    
        words_freq_sorted = sorted(words_freq,  key=lambda x:x[1], reverse=True)
        return words_freq_sorted

    # 13/20/35
    search_words_list_pos = ['舒适','用心','体贴不错','五星','贴心','到位','舒服','态度好','服务好',
            '友好','细致','温馨','温暖','热心','周到','热情','升级','礼貌','赞','很棒','棒',
            '一流','佳']
    search_words_list_neg = ['有待提高','不怎么样','服务差']  
    text_k = texts_all_high_hotel['H35']['服务']
    get_freq(text_k, search_words_list_neg)
    get_freq(text_k, search_words_list_pos)

    search_words_list_pos = ['位置好', '靠海', '位置棒', '周边餐饮酒吧近', '视野好', '周边环境优雅', 
        '交通方便', '附近酒吧街', '离海近', '附近景色好', '出行方便', '便利', '视野无敌', '视野广', 
        '附近吃饭方便', '地段好', '视野棒', '视野美', '附近食肆', '离澳门近', '港珠澳大桥', '亲子', 
        '离关口近', '位置好找', '周边配套齐全', '周边方便', '周边近', '离近', '附近玉市',
        '亚洲第一喷泉近', '周边美食多', '离江边不远', '玉市']

    search_words_list_neg = ['周边配套不方便', '位置稍偏', '晚上没位置停车']
    text_k = texts_all_high_hotel['H35']['位置']
    get_freq(text_k, search_words_list_neg)
    get_freq(text_k, search_words_list_pos)

    search_words_list_pos = ['完备','新装修','餐厅nice','餐厅棒','设施不错','设施高端上档次','设施五星',
                '设施好','设施齐全','设施完备','设施新','设施没的说','设施棒','床舒适','床舒服',
                '无可挑剔','游泳池','健身房','游乐场','床大','床不错','床软','装修喜欢','装修好',
                '装修年代感','空调给力','热水充足','ktv']
    search_words_list_neg = ['空调冷','餐厅挤','设施老','隔音差','不太隔音','隔音不好',
            '没有热空调','设施太差','加不了床', '停车场不太行']
    text_k = texts_all_high_hotel['H35']['设施']
    get_freq(text_k, search_words_list_neg)
    get_freq(text_k, search_words_list_pos)



    search_words_list_pos = ['环境好','环境赞','环境舒适','环境不错','环境棒','环境优美','环境佳',
            '干净','干湿分离','卫生不错','卫生好','整洁']
    search_words_list_neg = ['环境一般','细节上有欠缺','卫生状况堪忧', '有烟味'] 
    text_k = texts_all_high_hotel['H35']['卫生']
    get_freq(text_k, search_words_list_neg)
    get_freq(text_k, search_words_list_pos)


    search_words_list_pos = ['免费','便宜','优惠','性价比高','性价比好','价格合理','价格适中',
            '价格合适','早餐丰富', '免费儿童游乐场', '免费接送', '免费升级', '便宜', '免费烘干服务', 
            '价格适中', '免费加床', '早餐免费', '优惠', '免费的健身房', '免费宵夜', '免费洗衣', 
            '性价比杠的', '免费延迟退房', '电动窗帘', '免费停车场', ]

    search_words_list_neg = ['吃饭不便宜', '很贵', '饮食贵', ' 价格偏高', '价格不便宜', '不便宜']
 
    text_k = texts_all_high_hotel['H35']['性价比']
    get_freq(text_k, search_words_list_neg)
    get_freq(text_k, search_words_list_pos)


    
    
    ''' 词频 '''
    # 13/20/35
    # 11/38/39
    text_k = texts_all_high_hotel['H35']['位置']
    for i in range(len(text_k)):
        print(i, ': ', text_k[i])

    search_words_list_pos = ['位置好', '靠海', '位置棒', '周边餐饮酒吧近', '视野好', '周边环境优雅', 
        '交通方便', '附近酒吧街', '离海近', '附近景色好', '出行方便', '便利', '视野无敌', '视野广', 
        '附近吃饭方便', '地段好', '视野棒', '视野美', '附近食肆', '离澳门近', '港珠澳大桥', '亲子', 
        '离关口近', '位置好找', '周边配套齐全', '周边方便', '周边近', '离近', '附近玉市',
        '亚洲第一喷泉近', '周边美食多', '离江边不远']

    search_words_list_neg = ['周边配套不方便', '位置稍偏', '晚上没位置停车']

    text_k = texts_all_high_hotel['H35']['性价比']
    for i in range(len(text_k)):
        print(i, ': ', text_k[i])

    search_words_list_pos = ['免费','便宜','优惠','性价比高','性价比好','价格合理','价格适中',
            '价格合适','早餐丰富', '免费儿童游乐场', '免费接送', '免费升级', '便宜', '免费烘干服务', 
            '价格适中', '免费加床', '早餐免费', '优惠', '免费的健身房', '免费宵夜', '免费洗衣', 
            '性价比杠的', '免费延迟退房', '电动窗帘', '免费停车场', ]

    search_words_list_neg = ['吃饭不便宜', '很贵', '饮食贵', ' 价格偏高', '价格不便宜', '不便宜']


    # 11/38/39
    text_k = texts_all_low_hotel['H11']['服务']
    for i in range(len(text_k)):
        print(i, ': ', text_k[i])

    search_words_list_pos = ['升级', '服务态度好', '态度好', '热情', '满意', '超五星', '友好', '整体不错', 
                '服务及时', '服务棒', '服务好', '服务到位', '服务在线', '送果盘和巧克力', '服务意识好',
                '延迟退房', '送早餐券', '服务细致周到', '体验不错', '亲子的好去处', '高效', '专业', 
                '贴心', '到位', '优质', '品牌', '耐心', '提供私人管家的服务', '送的快', '周到', '周全']

    search_words_list_neg = ['人员配置不够', '说话不清晰', '没有服务意识', '质量下降', '有待加强', 
            '没有提醒可以减免费用', '加收服务费', '不太人性化', '态度傲慢', '需提高', '餐厅服务意识全无', 
            '服务不行', '个别服务态度不是很棒', '差评', '态度差', '员工缺乏活力', '服务员少', 
            '差到忍无可忍', '落后', '服务一般', '服务跟不上', '需要提高', '对不起这个价格', 
            '服务不到位', '不给开发票', '倒数前三', '前台的男孩子好拽', '跟不上', '态度高冷', 
            '前台非常多管闲事', '不咋的', '态度恶劣', '本上没有服务', '免费停车']

    get_freq(text_k, search_words_list_neg)
    get_freq(text_k, search_words_list_pos)

    # 11/38/39
    text_k = texts_all_low_hotel['H39']['位置']
    for i in range(len(text_k)):
        print(i, ': ', text_k[i])

    search_words_list_pos = ['位置好', '门口有地铁', '周边各种景点', '交通方便', '位置没得说', 
            '地铁站附近', '适合带孩子', '便利', '地理位置得天独厚', '位置优越', '位置棒', '附近美食多', 
            '周边空气清新', '近景区', '周边逛的挺多', '地铁口', '欢乐谷很近', '周边环境优美', '位置不错', 
            '视野很好', '世界之窗近', '临近几大景点', '视野棒', '位置佳', '周边景点多', '离机场近',
            '配的上不便宜的价格', '视野开阔', '位置不错', '位置在海边', '周边食饭方便', '位置靠海', 
            '停车场位置多', '海滩很近', '周边有地方特色', '周边吃的多','附近小吃多','附近美食多',
            '夜市近', '周边景点多', '出行方便', '位置优越', '周边餐饮多', '交通便捷', '繁华地段']

    search_words_list_neg = ['差', '位置糟糕', '周边混乱不堪', '偏远', '位置偏', '离海远', ]
    
    get_freq(text_k, search_words_list_neg)
    get_freq(text_k, search_words_list_pos)

    # 11/38/39
    text_k = texts_all_low_hotel['H39']['设施']
    for i in range(len(text_k)):
        print(i, ': ', text_k[i])

    search_words_list_pos = ['泳池', '设施不错', '人造沙滩', '中餐厅', '儿童游乐设施', '空调不吵', 
            '床舒服', '床舒适', '设施好', '维护不错', '早餐品种足够',  '舒适', '车位足够', '设施齐全', 
            '餐厅美味', '床大', '设施棒', '设施完善','安静', '有停车场', '健身房', '西餐厅','温泉', 
            '设施齐全', '茶餐厅', '餐厅菜品正宗', '宽敞舒适', '四星酒店设施', '隔音好', '水床', ]

    search_words_list_neg = ['设施老旧', '陈旧', '空调冷气不够', '隔音差', '老化', '床垫太高',
            '隔音导致无眠', '装修风格老', '旧', '床头灯坏',  '空调冷', '很多设施不开', '床一般', 
            '房间旧', '房间差', '臭味', '设施维护跟不上', '设施老', '床小', '空调除湿一般', '简陋', 
            '装修落后', '空调失灵', '泳池小', '设施差', '质量差', '床褥潮湿', '空调坏', '健身房小',
            '空调不够凉', '设施一般', '热水不热', '中餐厅菜式少', '差', '设施差', '空调噪音大', 
            '空调不好使', '没有暖气', '被子潮湿', '床垫破损', '没电梯']

    get_freq(text_k, search_words_list_neg)
    get_freq(text_k, search_words_list_pos)

    # 11/38/39
    text_k = texts_all_low_hotel['H39']['卫生']
    for i in range(len(text_k)):
        print(i, ': ', text_k[i])

    search_words_list_pos = ['环境好', '环境不错', '整齐干净', '卫生间宽敞', '卫生好', '环境舒适', 
            '干净', '空气净化器', '环境优美', '卫生', '环境优雅', ]

    search_words_list_neg = ['霉味', '不干净', '整洁', '蚊虫叮咬', '环境恶劣', '卫生差', '卫生堪忧', 
            '卫生要加强', '蟑螂', '希望改善卫生', '床单不干净', '厕所熏人', '卫生一般', '卫生欠佳']

    get_freq(text_k, search_words_list_neg)
    get_freq(text_k, search_words_list_pos)

    # 11/38/39
    text_k = texts_all_low_hotel['H39']['性价比']
    for i in range(len(text_k)):
        print(i, ': ', text_k[i])
    
    search_words_list_pos = ['实惠', '免费升级', '性价比高', '性价比好', '免费', '优惠活动', '价格值得', 
            '免费停车', '价格还可以', '便宜', '不贵', '价格合理','免费提车', '价格公道', '价格亲民', 
            ]

    search_words_list_neg = ['价格贵', '性价比不高', '自助晚餐贵', '价格不便宜', '贵', '不便宜', '没有优惠', 
            '没有免费早餐', '性价比一般', ]
    
    get_freq(text_k, search_words_list_neg)
    get_freq(text_k, search_words_list_pos)

    # =====================================================================

    doc_k_all = [[w for w in jieba.lcut(sen) if w not in stop_words] for sen in text_k]
    dic_k = Dictionary(doc_k_all)
    words_freq_dict = dic_k.dfs
    w_freq = []
    for w in dic_k.token2id.keys():
        freqs = words_freq_dict[dic_k.token2id[w]]
        w_freq.append((w, freqs))
    w_freq_sorted = sorted(w_freq, key=lambda x:x[1], reverse=True)


    ''' eda大作业 '''
    path = 'D:/competition/酒店数据情感分类项目(EDA大作业)/temp_results/'
    hotel_review = pd.read_csv(open(path+'hotel_mark_tag.csv', 'rb'), index_col=0)
    hotel_review = hotel_review.loc[:,  ['label', 'review', 'sentiment_tag_model']]
    hotel_review.head()
    hotel_review['label'].value_counts()
    review_len = [len(x) for x in hotel_review['review']]
    distribution_plot_TextLength(review_len)
    np.max(review_len)
    np.median(review_len)
    np.min(review_len)
    np.mean(review_len)


    # 举个例子、
    i = 14
    text = hotel_review['review'].iloc[i]
    predict(text)

    # 情感词典标注

    hotel_review_dict = sentiment_tagging(hotel_review)
    # hotel_review.to_csv(open(path+'mark_hotel_tag.csv', 'w'))
    
    pred = hotel_review_dict['sentiment_tag_dict'].values
    ms.accuracy_score(hotel_review['label'], pred)
    ms.recall_score(hotel_review['label'], pred)
    ms.precision_score(hotel_review['label'], pred)
    ms.f1_score(hotel_review['label'], pred)
    
    ms.confusion_matrix(hotel_review['label'], pred)
    id_all = [i for i in range(len(hotel_review)) if hotel_review['label'].iloc[i] != hotel_review['sentiment_tag'].iloc[i]]

    # 实体发生变化
    hotel_review['review'].iloc[17]
    hotel_review['review'].iloc[29] 

    id_ = id_all[8]
    text = hotel_review['review'].iloc[322]
    # 机械压缩
    text
    SA.normalization_score(text)
    hotel_review['label'].iloc[id_]




