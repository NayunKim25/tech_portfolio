#!/usr/bin/env python
# coding: utf-8

# #### 1. 데이터 불러오기

# ##### 1) 리뷰 데이터 합치기

# In[ ]:


import pandas as pd

carsales = pd.read_csv('/content/drive/MyDrive/Document/carsales_carwale review.csv', header=None)
topgear = pd.read_csv('/content/drive/MyDrive/Document/Top gear review.csv', header=None)
edmunds = pd.read_csv('/content/drive/MyDrive/Document/Edmunds review.csv', header=None)

review_ls = [carsales, topgear, edmunds]
review = pd.concat(review_ls, ignore_index=True)
review.head()


# In[ ]:


review.info()


# ##### 2) 특허 데이터 불러오기

# In[ ]:


patents = pd.read_csv('/content/drive/MyDrive/Document/gp_제외.csv', encoding='cp949')
patents.head(3)


# In[ ]:


patents.info()


# ### 추가 1-1. 특허 데이터 클러스터링

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.DataFrame(patents, columns=['title', 'abstract'])
df['title'] = df['title'].fillna("")
df['abstract'] = df['abstract'].fillna("")
df['combined_text'] = df['title'] + " " + df['abstract']

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

min_clusters = 2
max_clusters = 15

wcss = []
silhouette_scores = []

for num_clusters in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)

    wcss.append(kmeans.inertia_)

    if num_clusters > 1:
        silhouette_avg = silhouette_score(tfidf_matrix, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(10, 5))
plt.plot(range(min_clusters, max_clusters + 1), wcss, marker='o', label='WCSS')
plt.xlim([2, 15])
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(min_clusters + 1, max_clusters + 1), silhouette_scores[1:], marker='o', color='green', label='Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Analysis for Optimal Number of Clusters')
plt.legend()
plt.show()


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

num_clusters = 8
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(tfidf_matrix)

df['cluster'] = kmeans.labels_

print(df[['title', 'abstract', 'cluster']])


# In[ ]:


patents.columns


# In[ ]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

score_df = patents[['claims', 'citedby', 'ipc', 'family']]

# 스케일링
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(score_df)

df_scaled = pd.DataFrame(scaled_features, columns=score_df.columns)

# 각 특성에 가중치를 곱하여 점수 계산
df_scaled['score'] = (
    df_scaled['claims'] * 0.2 +
    df_scaled['citedby'] * 0.3 +
    df_scaled['family'] * 0.3 +
    df_scaled['ipc'] * 0.2
)

df.reset_index(inplace=True)
df['id'] = patents['id']
df['patent_score'] = df_scaled['score']
print(df['patent_score'])


# In[ ]:


# 클러스터별 데이터 출력
for cluster_num in sorted(df.index.unique()):
    print(f"\nCluster {cluster_num}")
    display(df.loc[cluster_num][['id', 'title', 'abstract', 'patent_score']])


# In[ ]:


# 각 클러스터의 평균 점수 계산
cluster_scores = df.groupby('cluster')['patent_score'].mean().reset_index()
cluster_scores.columns = ['cluster', 'average_score']

print("Patent Scores by Cluster:")
cluster_scores


# ### 추가 1-2. 리뷰 데이터 키워드 추출

# In[ ]:


review[0]


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('vader_lexicon')

r_df = pd.DataFrame(review)

custom_stop_words = ["car", 'charging', 'driver', 'driving', 'electric', 'ev', 'features', 'great', 'hyundai', 'ioniq', 'just', 'kona', 'like', 'miles', 'model', 'new', 'range',
                     'rear', 'seats', 'standard', 'vehicle', 'cars', 'comes', 'comfortable', 'does', 'easy', 'fast', 'good', 'inch', 'lane', 'little', 'road', 'small', 'suv', 'tesla',
                     'time', 'assist', 'auto', 'don', 'evs', 'far', 'feel', 'fun', 'high', 'll', 'really', 'tech', 'use', 'way', 'years', '000', '10', 'available', 'better', 'bit',
                     'blind', 'braking', 'buy', 'cabin', 'charger', 'dual', 'feels', 'hybrid', 'isn', 'level', 'limited', 'long', 'looking', 'lot', 'love', 'make', 'models', 'motor',
                     'need', 'plenty', 'plus', 'powered', 'ride', 'set', 'sound', 'technology', 'test', 'trim', 've', 'wheels', 'year','best', 'big', 'class', 'come', 'console', 'doesn',
                     'family', 'got', 'head', 'home', 'inside', 'kia', 'look', 'looks', 'low', 'makes', 'mode', 'pack', 'point', 'premium', 'pretty', 'quite', 'real', 'right', 'screen',
                     'sel', 'spot', 'start', 'things', 'want', 'world']

all_stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) + custom_stop_words

tfidf_vectorizer = TfidfVectorizer(stop_words=all_stop_words, max_features=50)

tfidf_matrix = tfidf_vectorizer.fit_transform(r_df[0])
keywords = tfidf_vectorizer.get_feature_names_out()

sia = SentimentIntensityAnalyzer()

keyword_sentiment_scores = {}

for keyword in keywords:
    keyword_reviews = r_df[r_df[0].str.contains(keyword, case=False, na=False)]

    scores = [sia.polarity_scores(review)['compound'] for review in keyword_reviews[0]]
    if scores:  # 키워드가 포함된 리뷰가 있는 경우에만 평균 계산
        keyword_sentiment_scores[keyword] = sum(scores) / len(scores)
    else:
        keyword_sentiment_scores[keyword] = None  # 키워드가 포함된 리뷰가 없는 경우

print("Keyword Sentiment Scores:")
for keyword, score in keyword_sentiment_scores.items():
    print(f"{keyword}: {score}")


# ### 2. 데이터 분석

# #### 2.1 리뷰 데이터 분석

# ##### 1) 토픽모델링

# In[ ]:


review.head(3)


# In[ ]:


review_series = pd.Series(review[0])
review_series.head(3)


# In[ ]:


review_series = review_series.apply(lambda x: x.lower())


# In[ ]:


import nltk
nltk.download('punkt')

review_txt = review_series.apply(lambda row: nltk.word_tokenize(row))
review_txt.head(3)


# In[ ]:


from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
review_txt = review_txt.apply(lambda x: [word for word in x if word not in (stop)])
review_txt.head(3)


# In[ ]:


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
review_txt = review_txt.apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])
review_txt.head()


# In[ ]:


#1글자 단어 삭제
tokenized_review = review_txt.apply(lambda x: [word for word in x if len(word) > 1])
tokenized_review[:3]


# In[ ]:


from string import punctuation
for p in punctuation :
  tokenized_review = tokenized_review.replace(p, "")

tokenized_review[:3]


# In[ ]:


words_list = ["\'s", "fe", "car", "hyundai", 'one', 'miles', 'ioniq', 'kona', 'i30n', 'maybe', "n\'t", 'like', 'santa', 'tucson', "get", 'ev', "--", 'would', 'drive', 'vw', "``", '2022', 'two', 'also',
              'even', 'i20n', 'rn22e', 'cars', "\'\'", 'much', 'say', 'go', 'tesla', 'better', '..', 'i30', 'electric', 'take', 'make', 'every', 'around', 'come', 'work', 'lot', 'look', 'vehicle',
              'something', 'new', 'buy', 'new', 'really', 'great', 'love', 'well', 'come', 'need','little', 'give', 'think', 'everything', 'see', 'second', 'first', 'nexo', 'evs', '2024', 'happier',
              'help', 'point', 'feature', 'standard', 'system', 'model', 'i10', 'hybrid', 'still', 'include', 'put', 'rival', '2023', 'feel', 'never', 'rear', 'front', 'many', 'ioniq5', 'use', 'back',
              'part', 'want', 'truck', 'way', 'offer', 'pickup', 'know', 'range', 'issue', 'problem', 'zero', 'rodgers', 'yes', 'seem', 'perfect', 'different', 'unit', 'easy', 'less', 'hear', 'deal',
              'dealer', 'degrees', 'top', 'months', 'suv', '-the', 'assist', 'bring', 'gt', 'base', 'another', 'longer', 'i20', 'nice', 'fiesta', 'bite', 'life', 'cent', 'far', 'hydrogen', 'fcv',
              'tell', 'van', 'staria', 'uk', 'keep', 'kia', 'brand', 'good', 'best', 'estimate', 'per', 'three', 'almost', 'purchase', 'sel', 'hope', 'pony', 'epiq', 'park', 'plus', 'available',
              'ev6', 'long', 'high', 'test', 'full', 'limit', "\'re", 'amaze', 'years', 'combination', 'sport', 'right', 'people', 'year', 'update', '2019', 'former', 'sort', 'maverick', 'order', 'ford', 'city',
              'excellent', 'try', 'epa', 'without', 'things', 'number', 'enough', 'pretty', 'may', 'ionic', 'though', '...', 'indeed', 'level', 'se', 'close', 'least', '2021', 'replace', 'leaf', 'days', 'disappoint',
              'i40', 'versions', '300', 'world', 'via', 'anything', 'free', 'please', 'ride', 'jenni', 'day', 'pay', 'behind', 'motor', 'side', 'decent', 'n-line', '8.0', 'winter', 'expect', 'review', 'us', 'sell',
              'could', 'fix', 'tech', 'support', 'let', 'find', 'communicate', 'effective', 'worry', 'money', 'version', 'claim', 'technology', 'ever', 'pull', 'extra', "\'m", 'learn', 'wrong', 'fill', 'reason',
              'koera', 'status', 'flaw', 'wait', 'rev', 'bayon', 'build', 'mean', 'mode', 'diesel', 'market', 'fact', 'major', 'specs', 'mile', 'show', 'choice', 'average', 'techniq', 'live', 'set', 'kmph', 'spec',
              'sound+', 'low', 'pair', 'hit', 'touch', 'week', "\'ve", 'since', 'dealership', '40a', '1.6-litre', 'road', 'ensure', 'allow', 'mini-mpv', 'start', 'course', 'yet', 'thing', 'inside', 'provide', 'actually',
              'launch', 'previous', 'cold', 'hot', '8.5', 'repair', 'change', 'six-speed', 'seven-speed', 'elite', 'value', 'view', 'lower', 'extend', 'weather', 'concern', 'plenty', 'rid', 'haul', 'follow', 'save', 'total',
              'class', 'audi', 'smaller', 'cargo', 'rat', 'quite', 'leave', 'km', 'although', 'whole', 'ultimate', 'buyers', 'end', 'amperage', 'variants', 'nothing', 'bother', 'compare', 'several', 'contrast', 'able', 'real',
              'computer', 'bmw', 'anxiety', 'pack', 'proper', 'i800', 'plan', 'enjoy', 'complete', 'import', 'away', 'sit', 'four', 'instead', 'track', 'bin', 'underneath', 'firm', 'fit', 'eye', 'always', 'open',
              'except', 'overall', 'bolt', 'chevy', 'wife', 'i-pedal', 'default', 'sure', 'division', 'fall', 'easily', '258', 'become', 'genesis', 'roll', 'current', 'nissan', 'thank', 'call', 'continue', 'public',
              'mondeo', 'visit', 'korea', 'seven', 'discovery', 'finally', 'super', 'multilink', 'clean', '7500', 'rather', 'canada', 'already', 'hours', '8k', 'regret', 'luxury', 'mercedes', 'lifetime', 'msrp',
              'move', 'run', 'add', 'lead', 'lease', 'perk', 'reduce', 'credit', 'pant', 'mi', 'terrific', 'head', 'options', 'seriously', 'con', 'strange', 'require', '12v', 'die', 'bad', 'unlike', 'port', 'either', 'grown-up',
              'fresh', '64kwh', 'might', 'ahead', 'convince', 'recall', '50', 'remain', 'turn']
tokenized_review = tokenized_review.apply(lambda x: [words for words in x if words not in (words_list)])


# In[ ]:


from gensim import corpora
dictionary = corpora.Dictionary(tokenized_review)
corpus = [dictionary.doc2bow(txt) for txt in tokenized_review]
corpus[10]


# In[ ]:


len(dictionary)


# In[ ]:


#Coherence 및 Perplexity score 계산
import gensim
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel

def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    coherence_values = []
    perplexity_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=tokenized_review, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        perplexity_values.append(model.log_perplexity(corpus))

    return model_list, coherence_values, perplexity_values

def find_optimal_number_of_topics(dictionary, corpus, processed_data):
    limit = 20;
    start = 2;
    step = 2;

    model_list, coherence_values, perplexity_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=processed_data, start=start, limit=limit, step=step)
    x = range(start, limit, step)

    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    plt.plot(x, perplexity_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Perplexity score")
    plt.legend(("perplexity_values"), loc='best')
    plt.show()


# In[ ]:


find_optimal_number_of_topics(dictionary, corpus, tokenized_review)


# In[ ]:


NUM_TOPICS = 14
ldamodel = gensim.models.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
topics = ldamodel.print_topics()
for topic in topics:
  print(topic)


# Cruise & Engine Performance
# 주요 키워드: cruz, engine, battery, wheel, style, light, crossovers, hatch, gearbox, manual
# 
# Charging & Performance Experience
# 주요 키워드: charge, experience, engineer, time, power, performance, driver, sound, steer, production
# 
# Camper Design & Battery Life
# 주요 키워드: seat, carrier, battery, design, camper, size, light, price, experience, headlights
# 
# Power & Speed Control
# 주요 키워드: performance, power, charge, control, speed, wheel, time, battery, function, seat
# 
# Touring & Petrol Systems
# 주요 키워드: tourer, systems, fun, petrol, seat, power, interior, driver, lane, brake
# 
# Charging & Battery Efficiency
# 주요 키워드: charge, battery, time, hatch, seat, space, circuit, locate, grille, trip
# 
# Charging Service & Cost Efficiency
# 주요 키워드: charge, service, price, interior, space, home, quality, time, cost, speed
# 
# Comfort & Charging Features
# 주요 키워드: seat, charge, battery, control, power, interior, comfortable, small, lane, brake
# 
# Driver Control & Safety
# 주요 키워드: seat, charge, battery, steer, wheel, control, power, driver, brake, safety
# 
# Battery & Driver Interaction
# 주요 키워드: battery, charge, screen, control, seat, wheel, brake, sound, time, driver
# 
# Cost & Gas Efficiency
# 주요 키워드: price, gas, cost, maintenance, fun, charge, power, fast, comfortable, speed
# 
# Family & Charging Needs
# 주요 키워드: charge, seat, family, time, station, power, price, charger, wheel, fast
# 
# Fuel & Battery Costs
# 주요 키워드: price, charge, petrol, seat, expensive, style, wheel, fuel, load, battery
# 
# Petrol & Vehicle Design
# 주요 키워드: petrol, light, trim, seat, hatch, engines, rational, gearbox, slow, facelift

# #### 추가. 단어 간 유사도

# In[ ]:


df.reset_index(inplace=True)


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(text_data, top_n=10):
    custom_stop_words = ['input', 'second', 'connected', 'vehicle', 'output', 'hybrid', 'rotary', 'element', 'mode', 'torque', 'planetary', 'selectively', 'set', 'electric', 'fixed',
                         'current', 'ev', 'method', 'apparatus', 'contract', 'certificate', 'pnc', 'cs', 'secc', 'line', 'step', 'based', 'traffic', 'present', 'invention', 'symmetric',
                         'soc', 'required', 'wireless', 'pairing', 'receiving', 'cross', 'service', 'chain', 'authorization', 'provider', 'message', 'including', 'unit', 'hydrogen',
                         'order', 'solar', 'distance', 'section', 'road', 'eco', 'user', 'supply', 'v2g', 'evcc', 'certification', 'root', 'supply', 'authentication', '100', 'value',
                         'includes', 'friendly', 'authority', 'sub', 'steps', 'setup', 'verifying', 'mutual', 'relates', 'according', 'request', 'key', 'list', 'session', 'response',
                         'station', 'evse', 'equipment', 'rootca', 'comprises', 'communication', 'result', 'used', 'configured', 'verification', 'providing', 'accepting',  'providers',
                         'using', 'situation', 'associated', 'supporting', 'pad', 'state']

    all_stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) + custom_stop_words

    vectorizer = TfidfVectorizer(stop_words=all_stop_words, max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform(text_data)

    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    keyword_scores = [(feature_names[i], scores[i]) for i in range(len(feature_names))]

    keyword_scores = sorted(keyword_scores, key=lambda x: x[1], reverse=True)

    return keyword_scores[:top_n]

# 특허 제목과 초록 텍스트에서 주요 키워드 추출
cluster0_keyword = extract_keywords(df[df['cluster']==0]['combined_text'], top_n=10)
cluster1_keyword = extract_keywords(df[df['cluster']==1]['combined_text'], top_n=10)
cluster2_keyword = extract_keywords(df[df['cluster']==2]['combined_text'], top_n=10)
cluster3_keyword = extract_keywords(df[df['cluster']==3]['combined_text'], top_n=10)
cluster4_keyword = extract_keywords(df[df['cluster']==4]['combined_text'], top_n=10)
cluster5_keyword = extract_keywords(df[df['cluster']==5]['combined_text'], top_n=10)
cluster6_keyword = extract_keywords(df[df['cluster']==6]['combined_text'], top_n=10)
cluster7_keyword = extract_keywords(df[df['cluster']==7]['combined_text'], top_n=10)

print("Top 10 Keywords from cluster0:")
for keyword, score in cluster0_keyword:
    print(f"{keyword}")

print("Top 10 Keywords from cluster1:")
for keyword, score in cluster1_keyword:
    print(f"{keyword}")

print("Top 10 Keywords from cluster2:")
for keyword, score in cluster2_keyword:
    print(f"{keyword}")

print("Top 10 Keywords from cluster3:")
for keyword, score in cluster3_keyword:
    print(f"{keyword}")

print("Top 10 Keywords from cluster4:")
for keyword, score in cluster4_keyword:
    print(f"{keyword}")

print("Top 10 Keywords from cluster5:")
for keyword, score in cluster5_keyword:
    print(f"{keyword}")

print("Top 10 Keywords from cluster6:")
for keyword, score in cluster6_keyword:
    print(f"{keyword}")

print("Top 10 Keywords from cluster7:")
for keyword, score in cluster7_keyword:
    print(f"{keyword}")


# In[ ]:


keyword0 = [keyword for keyword, score in cluster0_keyword]
keyword1 = [keyword for keyword, score in cluster1_keyword]
keyword2 = [keyword for keyword, score in cluster2_keyword]
keyword3 = [keyword for keyword, score in cluster3_keyword]
keyword4 = [keyword for keyword, score in cluster4_keyword]
keyword5 = [keyword for keyword, score in cluster5_keyword]
keyword6 = [keyword for keyword, score in cluster6_keyword]
keyword7 = [keyword for keyword, score in cluster7_keyword]


# In[ ]:


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cluster_keywords = {0: keyword0, 1: keyword1, 2: keyword2, 3: keyword3, 4: keyword4, 5: keyword5, 6: keyword6, 7: keyword7}

review_keywords = {0: ['cruz', 'engine', 'battery', 'wheel', 'style', 'light', 'crossovers', 'hatch', 'gearbox', 'manual'],
    1: ['charge', 'experience', 'engineer', 'time', 'power', 'performance', 'driver', 'sound', 'steer', 'production'],
    2: ['seat', 'carrier', 'battery', 'design', 'camper', 'size', 'light', 'price', 'experience', 'headlights'],
    3: ['performance', 'power', 'charge', 'control', 'speed', 'wheel', 'time', 'battery', 'function', 'seat'],
    4: ['tourer', 'systems', 'fun', 'petrol', 'seat', 'power', 'interior', 'driver', 'lane', 'brake'],
    5: ['charge', 'battery', 'time', 'hatch', 'seat', 'space', 'circuit', 'locate', 'grille', 'trip'],
    6: ['charge', 'service', 'price', 'interior', 'space', 'home', 'quality', 'time', 'cost', 'speed'],
    7: ['seat', 'charge', 'battery', 'control', 'power', 'interior', 'comfortable', 'small', 'lane', 'brake'],
    8: ['seat', 'charge', 'battery', 'steer', 'wheel', 'control', 'power', 'driver', 'brake', 'safety'],
    9: ['battery', 'charge', 'screen', 'control', 'seat', 'wheel', 'brake', 'sound', 'time', 'driver'],
    10: ['price', 'gas', 'cost', 'maintenance', 'fun', 'charge', 'power', 'fast', 'comfortable', 'speed'],
    11: ['charge', 'seat', 'family', 'time', 'station', 'power', 'price', 'charger', 'wheel', 'fast'],
    12: ['price', 'charge', 'petrol', 'seat', 'expensive', 'style', 'wheel', 'fuel', 'load', 'battery'],
    13: ['petrol', 'light', 'trim', 'seat', 'hatch', 'engines', 'rational', 'gearbox', 'slow', 'facelift']}

def get_tfidf_vectorizer(keywords_list):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    return tfidf_vectorizer.fit_transform([' '.join(keywords) for keywords in keywords_list])

all_keywords = list(cluster_keywords.values()) + list(review_keywords.values())
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_vectorizer.fit([' '.join(keywords) for keywords in all_keywords])

cluster_tfidf = tfidf_vectorizer.transform([' '.join(keywords) for keywords in cluster_keywords.values()])
review_tfidf = tfidf_vectorizer.transform([' '.join(keywords) for keywords in review_keywords.values()])

def compute_cosine_similarity(tfidf_matrix1, tfidf_matrix2):
    return cosine_similarity(tfidf_matrix1, tfidf_matrix2)

similarity_matrix = compute_cosine_similarity(cluster_tfidf, review_tfidf)

print("Cosine Similarity between clusters and review topics:")
for i, cluster_sim in enumerate(similarity_matrix):
    print(f"Cluster {i} vs Review Topics:")
    for j, score in enumerate(cluster_sim):
        print(f"  Review Topic {j}: Similarity = {score:.4f}")


# #### 추가. 리뷰 키워드 감정 점수 계산

# In[ ]:


pip install vaderSentiment


# In[ ]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

analyzer = SentimentIntensityAnalyzer()

analyzer = SentimentIntensityAnalyzer()

r_df = pd.DataFrame(review)

review_keywords = {0: ['cruz', 'engine', 'battery', 'wheel', 'style', 'light', 'crossovers', 'hatch', 'gearbox', 'manual'],
    1: ['charge', 'experience', 'engineer', 'time', 'power', 'performance', 'driver', 'sound', 'steer', 'production'],
    2: ['seat', 'carrier', 'battery', 'design', 'camper', 'size', 'light', 'price', 'experience', 'headlights'],
    3: ['performance', 'power', 'charge', 'control', 'speed', 'wheel', 'time', 'battery', 'function', 'seat'],
    4: ['tourer', 'systems', 'fun', 'petrol', 'seat', 'power', 'interior', 'driver', 'lane', 'brake'],
    5: ['charge', 'battery', 'time', 'hatch', 'seat', 'space', 'circuit', 'locate', 'grille', 'trip'],
    6: ['charge', 'service', 'price', 'interior', 'space', 'home', 'quality', 'time', 'cost', 'speed'],
    7: ['seat', 'charge', 'battery', 'control', 'power', 'interior', 'comfortable', 'small', 'lane', 'brake'],
    8: ['seat', 'charge', 'battery', 'steer', 'wheel', 'control', 'power', 'driver', 'brake', 'safety'],
    9: ['battery', 'charge', 'screen', 'control', 'seat', 'wheel', 'brake', 'sound', 'time', 'driver'],
    10: ['price', 'gas', 'cost', 'maintenance', 'fun', 'charge', 'power', 'fast', 'comfortable', 'speed'],
    11: ['charge', 'seat', 'family', 'time', 'station', 'power', 'price', 'charger', 'wheel', 'fast'],
    12: ['price', 'charge', 'petrol', 'seat', 'expensive', 'style', 'wheel', 'fuel', 'load', 'battery'],
    13: ['petrol', 'light', 'trim', 'seat', 'hatch', 'engines', 'rational', 'gearbox', 'slow', 'facelift']}

keyword_sentiment = {i: {keyword: [] for keyword in review_keywords[i]} for i in range(len(review_keywords))}

for review in r_df[0]:
    for topic_id, keywords in review_keywords.items():
        for keyword in keywords:
            if keyword in review.lower():
                sentiment = analyzer.polarity_scores(review)
                keyword_sentiment[topic_id][keyword].append(sentiment['compound'])

keyword_avg_sentiment = {
    topic_id: {keyword: np.mean(scores) if scores else None for keyword, scores in keywords.items()}
    for topic_id, keywords in keyword_sentiment.items()
}

print("Average Sentiment Scores for Each Keyword by Review Topic:")
for topic_id, topic_keywords in keyword_avg_sentiment.items():
    print(f"Topic {topic_id}:")
    for keyword, avg_score in topic_keywords.items():
        if avg_score is not None:
            print(f"  Keyword '{keyword}': Average Sentiment Score = {avg_score:.4f}")
        else:
            print(f"  Keyword '{keyword}' not found in any review.")


# topic 0 : 0.92581
# topic 1 : 0.8943
# topic 2 : 0.90916
# topic 3 : 0.86387
# topic 4 : 0.81259
# topic 5 : 0.87062
# topic 6 : 0.86563
# topic 7 : 0.88003
# topic 8 : 0.87218
# topic 9 : 0.85974
# topic 10 : 0.87058
# topic 11 : 0.88159
# topic 12 : 0.80742
# topic 13 : 0.91754

# In[ ]:


import pandas as pd

topics = ['topic 0', 'topic 1', 'topic 2', 'topic 3', 'topic 4', 'topic 5', 'topic 6', 'topic 7', 'topic 8', 'topic 9', 'topic 10', 'topic 11', 'topic 12', 'topic 13']
ss = [0.92851, 0.8943, 0.90916, 0.86387, 0.81259, 0.87062, 0.86563, 0.88003, 0.87218, 0.85974, 0.87058, 0.88159, 0.80742, 0.91754]

s_t_df = pd.DataFrame({'topic': topics, 'average_score': ss})
s_t_df


# #### 추가. 이종 네트워크

# In[ ]:


cluster_scores['average_score']


# In[ ]:


review_keywords


# In[ ]:


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G = nx.Graph()

for i in range(similarity_matrix.shape[0]):
    G.add_node(f'Cluster_{i}', size=similarity_matrix[i][i] * 8000, node_type='cluster')
for j in range(similarity_matrix.shape[1]):
    G.add_node(f'Topic_{j}', size=similarity_matrix[:, j].max() * 8000, node_type='topic')

for i in range(similarity_matrix.shape[0]):
    for j in range(similarity_matrix.shape[1]):
        if similarity_matrix[i][j] > 0.01:
            G.add_edge(f'Cluster_{i}', f'Topic_{j}', weight=similarity_matrix[i][j])

node_sizes = [G.nodes[node]['size'] for node in G.nodes]
node_colors = ['blue' if G.nodes[node]['node_type'] == 'cluster' else 'green' for node in G.nodes]
edge_widths = [G[u][v]['weight'] * 50 for u, v in G.edges]

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="gray", alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

plt.title("Cluster-Topic Heterogeneous Network")
plt.show()


# In[ ]:


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G = nx.Graph()

for i in range(similarity_matrix.shape[0]):
    G.add_node(f'Cluster_{i}', size=similarity_matrix[i][i] * 8000, node_type='cluster')
for j in range(similarity_matrix.shape[1]):
    G.add_node(f'Topic_{j}', size=similarity_matrix[:, j].max() * 8000, node_type='topic')

for i in range(similarity_matrix.shape[0]):
    for j in range(similarity_matrix.shape[1]):
        if similarity_matrix[i][j] > 0.05:
            G.add_edge(f'Cluster_{i}', f'Topic_{j}', weight=similarity_matrix[i][j])

node_sizes = [G.nodes[node]['size'] for node in G.nodes]
node_colors = ['blue' if G.nodes[node]['node_type'] == 'cluster' else 'green' for node in G.nodes]
edge_widths = [G[u][v]['weight'] * 50 for u, v in G.edges]

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="gray", alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

plt.title("Cluster-Topic Heterogeneous Network")
plt.show()


# In[ ]:


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G = nx.Graph()

for i in range(similarity_matrix.shape[0]):
    G.add_node(f'Cluster_{i}', size=similarity_matrix[i][i] * 8000, node_type='cluster')
for j in range(similarity_matrix.shape[1]):
    G.add_node(f'Topic_{j}', size=similarity_matrix[:, j].max() * 8000, node_type='topic')

for i in range(similarity_matrix.shape[0]):
    for j in range(similarity_matrix.shape[1]):
        if similarity_matrix[i][j] > 0.1:
            G.add_edge(f'Cluster_{i}', f'Topic_{j}', weight=similarity_matrix[i][j])

node_sizes = [G.nodes[node]['size'] for node in G.nodes]
node_colors = ['blue' if G.nodes[node]['node_type'] == 'cluster' else 'green' for node in G.nodes]
edge_widths = [G[u][v]['weight'] * 50 for u, v in G.edges]

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="gray", alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

plt.title("Cluster-Topic Heterogeneous Network")
plt.show()


# In[ ]:


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

patent_cluster_scores = {0: 0.278754, 1: 0.306699, 2: 0.321152, 3: 0.313098, 4: 0.304844, 5: 0.282546, 6: 0.223950, 7: 0.247878}
review_topic_sentiments = {0: 0.92581, 1: 0.8943,  2: 0.90916, 3: 0.86387, 4: 0.81259, 5: 0.87062, 6: 0.86563, 7: 0.88003, 8: 0.87218, 9: 0.85974, 10: 0.87058, 11: 0.88159, 12: 0.80742, 13: 0.91754}
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def min_max_scaling(value, old_min, old_max, new_min, new_max):
    return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

cluster_min = min(patent_cluster_scores.values())
cluster_max = max(patent_cluster_scores.values())

sentiment_min = min(review_topic_sentiments.values())
sentiment_max = max(review_topic_sentiments.values())

min_size = 100
max_size = 2000

G = nx.Graph()

for i in range(similarity_matrix.shape[0]):
    scaled_size = min_max_scaling(patent_cluster_scores[i],
                                cluster_min, cluster_max,
                                min_size, max_size)
    G.add_node(f'Cluster_{i}', size=scaled_size, node_type='cluster')

for j in range(similarity_matrix.shape[1]):
    scaled_size = min_max_scaling(review_topic_sentiments[j],
                                sentiment_min, sentiment_max,
                                min_size, max_size)
    G.add_node(f'Topic_{j}', size=scaled_size, node_type='topic')


for i in range(similarity_matrix.shape[0]):
    for j in range(similarity_matrix.shape[1]):
        if similarity_matrix[i][j] > 0.05:
            G.add_edge(f'Cluster_{i}', f'Topic_{j}', weight=similarity_matrix[i][j])

node_sizes = [G.nodes[node]['size'] for node in G.nodes]
node_colors = ['blue' if G.nodes[node]['node_type'] == 'cluster' else 'green' for node in G.nodes]
edge_widths = [G[u][v]['weight'] * 60 for u, v in G.edges]

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="gray", alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

plt.title("Cluster-Topic Heterogeneous Network")
plt.show()


# ##### 2) 키워드 분석

# In[ ]:


review_token = []
for i in range(len(tokenized_review)):
   review_token.extend(tokenized_review[i])

review_token[:5]


# In[ ]:


words_to_delete = ["\'s", "fe", "car", "hyundai", 'one', 'miles', 'ioniq', 'kona', 'i30n', 'maybe', "n\'t", 'like', 'santa', 'tucson', "get", 'ev', "--", 'would', 'drive', 'vw', "``", '2022', 'two', 'also',
              'even', 'i20n', 'rn22e', 'cars', "\'\'", 'much', 'say', 'go', 'tesla', 'better', '..', 'i30', 'electric', 'take', 'make', 'every', 'around', 'come', 'work', 'lot', 'look', 'vehicle',
              'something', 'new', 'buy', 'new', 'really', 'love', 'well', 'come', 'need','little', 'give', 'think', 'everything', 'see', 'second', 'first', 'nexo', 'evs', '2024', 'happier',
              'help', 'point', 'feature', 'standard', 'system', 'model', 'i10', 'hybrid', 'still', 'include', 'put', 'rival', '2023', 'feel', 'never', 'rear', 'front', 'many', 'ioniq5', 'use', 'back',
              'part', 'want', 'truck', 'way', 'offer', 'pickup', 'know', 'range', 'issue', 'problem', 'zero', 'rodgers', 'yes', 'seem', 'different', 'unit', 'easy', 'less', 'hear', 'deal',
              'dealer', 'degrees', 'top', 'months', 'suv', '-the', 'assist', 'bring', 'gt', 'base', 'another', 'longer', 'i20', 'fiesta', 'bite', 'life', 'cent', 'far', 'hydrogen', 'fcv',
              'tell', 'van', 'staria', 'uk', 'keep', 'kia', 'brand', 'good', 'best', 'estimate', 'per', 'three', 'almost', 'purchase', 'sel', 'hope', 'pony', 'epiq', 'park', 'plus', 'available',
              'ev6', 'long', 'high', 'test', 'full', 'limit', "\'re", 'amaze', 'years', 'combination', 'sport', 'right', 'people', 'year', 'update', '2019', 'former', 'sort', 'maverick', 'order', 'ford', 'city',
              'try', 'epa', 'without', 'things', 'number', 'enough', 'pretty', 'may', 'ionic', 'though', '...', 'indeed', 'level', 'se', 'close', 'least', '2021', 'replace', 'leaf', 'days',
              'i40', 'versions', '300', 'world', 'via', 'anything', 'free', 'please', 'ride', 'jenni', 'day', 'pay', 'behind', 'motor', 'side', 'decent', 'n-line', '8.0', 'winter', 'expect', 'review', 'us', 'sell',
              'could', 'fix', 'tech', 'support', 'let', 'find', 'communicate', 'effective', 'worry', 'money', 'version', 'claim', 'technology', 'ever', 'pull', 'extra', "\'m", 'learn', 'wrong', 'fill', 'reason',
              'koera', 'status', 'flaw', 'wait', 'rev', 'bayon', 'build', 'mean', 'mode', 'diesel', 'market', 'fact', 'major', 'specs', 'mile', 'show', 'choice', 'average', 'techniq', 'live', 'set', 'kmph', 'spec',
              'sound+', 'low', 'pair', 'hit', 'touch', 'week', "\'ve", 'since', 'dealership', '40a', '1.6-litre', 'road', 'ensure', 'allow', 'mini-mpv', 'start', 'course', 'yet', 'thing', 'inside', 'provide', 'actually',
              'launch', 'previous', 'cold', 'hot', '8.5', 'repair', 'change', 'six-speed', 'seven-speed', 'elite', 'value', 'view', 'lower', 'extend', 'weather', 'concern', 'plenty', 'rid', 'haul', 'follow', 'save', 'total',
              'class', 'audi', 'smaller', 'cargo', 'rat', 'quite', 'leave', 'km', 'although', 'whole', 'ultimate', 'buyers', 'end', 'amperage', 'variants', 'nothing', 'bother', 'compare', 'several', 'contrast', 'able', 'real',
              'computer', 'bmw', 'anxiety', 'pack', 'proper', 'i800', 'plan', 'enjoy', 'complete', 'import', 'away', 'sit', 'four', 'instead', 'track', 'bin', 'underneath', 'firm', 'fit', 'eye', 'always', 'open',
              'except', 'overall', 'bolt', 'chevy', 'wife', 'i-pedal', 'default', 'sure', 'division', 'fall', 'easily', '258', 'become', 'genesis', 'roll', 'current', 'nissan', 'thank', 'call', 'continue', 'public',
              'mondeo', 'visit', 'korea', 'seven', 'discovery', 'finally', 'super', 'multilink', 'clean', '7500', 'rather', 'canada', 'already', 'hours', '8k', 'regret', 'luxury', 'mercedes', 'lifetime', 'msrp',
              'move', 'run', 'add', 'lead', 'lease', 'perk', 'reduce', 'credit', 'pant', 'mi', 'head', 'options', 'seriously', 'con', 'strange', 'require', '12v', 'die', 'bad', 'unlike', 'port', 'either', 'grown-up',
              '64kwh', 'might', 'ahead', 'convince', 'recall', '50', 'remain', 'turn']

token = list(filter(lambda word: word not in words_to_delete, review_token))


# In[ ]:


token[:3]


# In[ ]:


from collections import Counter

token_counter = Counter(token)

new_wordInfo = dict()
print("\n--Token : Freq--")
for tags, counts in token_counter.most_common(50):
    new_wordInfo[tags] = counts
    print("%6s : %d" % (tags, counts))


# In[ ]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(relative_scaling = 0.2,
                      background_color='white',
                      colormap="twilight"
                      ).generate_from_frequencies(new_wordInfo)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud)
plt.axis("off")
#plt.savefig("before_gpt_wordcloud.png")
plt.show()


# ##### 3) 텍스트 연관 분석(네트워크 분석)

# ##### 4) 감성분석

# ##### VADER

# In[ ]:


import nltk
nltk.download('vader_lexicon')


# In[ ]:


review.info()


# In[ ]:


review.head(3)


# In[ ]:


review.columns


# In[ ]:


review.rename(columns={review.columns[0]: 'txt'}, inplace=True)


# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

sia = SentimentIntensityAnalyzer()

def sentiment_category(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

review['compound'] = review['txt'].apply(lambda x: sia.polarity_scores(x)['compound'])
review['sentiment'] = review['compound'].apply(sentiment_category)

print(review)

print(review['sentiment'].value_counts())

average_sentiment = review['compound'].mean()
print(f"Average Sentiment Score: {average_sentiment}")


# In[ ]:


review.info()


# In[ ]:


# 키워드 분석 결과 상위 15개 단어
keywords = ['seat', 'charge', 'battery', 'power', 'wheel', 'control', 'price', 'steer', 'time', 'driver', 'brake', 'interior', 'safety', 'speed', 'performance']

results = {}
for keyword in keywords:
    keyword_reviews = review[review['txt'].str.contains(keyword, case=False)]
    avg_sentiment = keyword_reviews['compound'].mean()
    results[keyword] = avg_sentiment

print("기능별 평균 감성 점수:")
for keyword, avg_sentiment in results.items():
    print(f"{keyword}: {avg_sentiment}")


# In[ ]:


pd.DataFrame(results.items())[1]


# In[ ]:


top_keyword_15 = pd.DataFrame(results.items()).sort_values(by=1, ascending=False)
top_keyword_15.head(3)


# ##### 감성 점수 부여 후 키워드 분석 결과의 상위 15개 단어에 대한 평균 감성 점수 확인

# In[ ]:


import matplotlib.pyplot as plt
plt.barh(top_keyword_15[0], top_keyword_15[1])
plt.show()


# ##### 감성 분석 후 긍정, 부정, 중립 각각에 대하여 키워드 분석

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# 긍정, 부정, 중립 리뷰로 분류
positive_reviews = review[review['sentiment'] == 'Positive']['txt']
negative_reviews = review[review['sentiment'] == 'Negative']['txt']
neutral_reviews = review[review['sentiment'] == 'Neutral']['txt']

custom_stopwords = stopwords.words('english')
custom_stopwords.extend(["\'s", "fe", "car", "hyundai", 'one', 'miles', 'ioniq', 'kona', 'i30n', 'maybe', "n\'t", 'like', 'santa', 'tucson', "get", 'ev', "--", 'would', 'drive', 'vw', "``", '2022', 'two', 'also',
              'even', 'i20n', 'rn22e', 'cars', "\'\'", 'much', 'say', 'go', 'tesla', 'better', '..', 'i30', 'electric', 'take', 'make', 'every', 'around', 'come', 'work', 'lot', 'look', 'vehicle',
              'something', 'new', 'buy', 'new', 'really', 'love', 'well', 'come', 'need','little', 'give', 'think', 'everything', 'see', 'second', 'first', 'nexo', 'evs', '2024', 'happier',
              'help', 'point', 'feature', 'standard', 'system', 'model', 'i10', 'hybrid', 'still', 'include', 'put', 'rival', '2023', 'feel', 'never', 'rear', 'front', 'many', 'ioniq5', 'use', 'back',
              'part', 'want', 'truck', 'way', 'offer', 'pickup', 'know', 'range', 'issue', 'problem', 'zero', 'rodgers', 'yes', 'seem', 'different', 'unit', 'easy', 'less', 'hear', 'deal',
              'dealer', 'degrees', 'top', 'months', 'suv', '-the', 'assist', 'bring', 'gt', 'base', 'another', 'longer', 'i20', 'fiesta', 'bite', 'life', 'cent', 'far', 'hydrogen', 'fcv',
              'tell', 'van', 'staria', 'uk', 'keep', 'kia', 'brand', 'good', 'best', 'estimate', 'per', 'three', 'almost', 'purchase', 'sel', 'hope', 'pony', 'epiq', 'park', 'plus', 'available',
              'ev6', 'long', 'high', 'test', 'full', 'limit', "\'re", 'amaze', 'years', 'combination', 'sport', 'right', 'people', 'year', 'update', '2019', 'former', 'sort', 'maverick', 'order', 'ford', 'city',
              'try', 'epa', 'without', 'things', 'number', 'enough', 'pretty', 'may', 'ionic', 'though', '...', 'indeed', 'level', 'se', 'close', 'least', '2021', 'replace', 'leaf', 'days',
              'i40', 'versions', '300', 'world', 'via', 'anything', 'free', 'please', 'ride', 'jenni', 'day', 'pay', 'behind', 'motor', 'side', 'decent', 'n-line', '8.0', 'winter', 'expect', 'review', 'us', 'sell',
              'could', 'fix', 'tech', 'support', 'let', 'find', 'communicate', 'effective', 'worry', 'money', 'version', 'claim', 'technology', 'ever', 'pull', 'extra', "\'m", 'learn', 'wrong', 'fill', 'reason',
              'koera', 'status', 'flaw', 'wait', 'rev', 'bayon', 'build', 'mean', 'mode', 'diesel', 'market', 'fact', 'major', 'specs', 'mile', 'show', 'choice', 'average', 'techniq', 'live', 'set', 'kmph', 'spec',
              'sound+', 'low', 'pair', 'hit', 'touch', 'week', "\'ve", 'since', 'dealership', '40a', '1.6-litre', 'road', 'ensure', 'allow', 'mini-mpv', 'start', 'course', 'yet', 'thing', 'inside', 'provide', 'actually',
              'launch', 'previous', 'cold', 'hot', '8.5', 'repair', 'change', 'six-speed', 'seven-speed', 'elite', 'value', 'view', 'lower', 'extend', 'weather', 'concern', 'plenty', 'rid', 'haul', 'follow', 'save', 'total',
              'class', 'audi', 'smaller', 'cargo', 'rat', 'quite', 'leave', 'km', 'although', 'whole', 'ultimate', 'buyers', 'end', 'amperage', 'variants', 'nothing', 'bother', 'compare', 'several', 'contrast', 'able', 'real',
              'computer', 'bmw', 'anxiety', 'pack', 'proper', 'i800', 'plan', 'enjoy', 'complete', 'import', 'away', 'sit', 'four', 'instead', 'track', 'bin', 'underneath', 'firm', 'fit', 'eye', 'always', 'open',
              'except', 'overall', 'bolt', 'chevy', 'wife', 'i-pedal', 'default', 'sure', 'division', 'fall', 'easily', '258', 'become', 'genesis', 'roll', 'current', 'nissan', 'thank', 'call', 'continue', 'public',
              'mondeo', 'visit', 'korea', 'seven', 'discovery', 'finally', 'super', 'multilink', 'clean', '7500', 'rather', 'canada', 'already', 'hours', '8k', 'regret', 'luxury', 'mercedes', 'lifetime', 'msrp', 'confortable',
              'move', 'run', 'add', 'lead', 'lease', 'perk', 'reduce', 'credit', 'pant', 'mi', 'head', 'options', 'seriously', 'con', 'strange', 'require', '12v', 'die', 'bad', 'unlike', 'port', 'either', 'grown-up',
              '64kwh', 'might', 'ahead', 'convince', 'recall', '50', 'remain', 'turn', 'great', 'died', 'hits', 'levels', 'drove', 'preferred', 'found', 'features', 'within', 'charging', 'nice', 'towed', 'lines', 'time', 'sleek',
              'disappointed', 'driver', 'shop', 'slow', '25', 'ac', 'adults', 'affordable', 'cramped', 'certain', 'waiting', 'handles', 'seat', 'hoped', 'fun', 'comfortable', 'deceptive', 'decrease', 'draining', 'difficult',
              'expecting', 'fast', 'limited', 'saw', 'bit', 'person', 'expecting', 'smooth', 'ago', 'equivalent', 'refuses', 'fantastic', 'issues', 'ft', 'gas', 'conditions', 'clearance'])

def extract_avg_keywords(reviews, stopwords, num_keywords=10):
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix = vectorizer.fit_transform(reviews)
    feature_names = vectorizer.get_feature_names_out()

    avg_tfidf_scores = tfidf_matrix.mean(axis=0).A1

    avg_keywords = {feature_names[i]: avg_tfidf_scores[i] for i in range(len(feature_names))}

    sorted_avg_keywords = sorted(avg_keywords.items(), key=lambda item: item[1], reverse=True)
    return sorted_avg_keywords[:num_keywords]

positive_keywords = extract_avg_keywords(positive_reviews, custom_stopwords)
negative_keywords = extract_avg_keywords(negative_reviews, custom_stopwords)
neutral_keywords = extract_avg_keywords(neutral_reviews, custom_stopwords)

print("긍정 리뷰 키워드:")
for keyword, score in positive_keywords:
    print(f"{keyword}: {score}")

print("\n부정 리뷰 키워드:")
for keyword, score in negative_keywords:
    print(f"{keyword}: {score}")

print("\n중립 리뷰 키워드:")
for keyword, score in neutral_keywords:
    print(f"{keyword}: {score}")


# #### 2.2 특허 데이터 분석

# ##### 1) 토픽모델링

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import re
import nltk
nltk.download('punkt')

patents['abstract_token'] = patents['abstract'].astype('str').apply(lambda x: x.lower())

stop_words = stopwords.words('english')
patents['abstract_token'] = patents['abstract_token'].apply(lambda x: [word for word in x if word not in (stop)])

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
patents['abstract_token'] = patents['abstract_token'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])

patents['abstract_token'] = patents['abstract'].apply(lambda row: nltk.word_tokenize(row))

patents['abstract_token'] = patents['abstract_token'].apply(lambda x: [word for word in x if len(word) > 2])

from string import punctuation
for p in punctuation :
  patents['abstract_token'] = patents['abstract_token'].replace(p, "")


# In[ ]:


words_list = ["\'s", "fe", "car", "hyundai", 'one', 'miles', 'ioniq', 'kona', 'i30n', 'maybe', "n\'t", 'like', 'santa', 'tucson', "get", 'ev', "--", 'would', 'drive', 'vw', "``", '2022', 'two', 'also',
              'even', 'i20n', 'rn22e', 'cars', "\'\'", 'much', 'say', 'go', 'tesla', 'better', '..', 'i30', 'electric', 'take', 'make', 'every', 'around', 'come', 'work', 'lot', 'look', 'vehicle',
              'something', 'new', 'buy', 'new', 'really', 'great', 'love', 'well', 'come', 'need','little', 'give', 'think', 'everything', 'see', 'second', 'first', 'nexo', 'evs', '2024', 'happier',
              'help', 'point', 'feature', 'standard', 'system', 'model', 'i10', 'hybrid', 'still', 'include', 'put', 'rival', '2023', 'feel', 'never', 'rear', 'front', 'many', 'ioniq5', 'use', 'back',
              'part', 'want', 'truck', 'way', 'offer', 'pickup', 'know', 'range', 'issue', 'problem', 'zero', 'rodgers', 'yes', 'seem', 'perfect', 'different', 'unit', 'easy', 'less', 'hear', 'deal',
              'dealer', 'degrees', 'top', 'months', 'suv', '-the', 'assist', 'bring', 'gt', 'base', 'another', 'longer', 'i20', 'nice', 'fiesta', 'bite', 'life', 'cent', 'far', 'hydrogen', 'fcv',
              'tell', 'van', 'staria', 'uk', 'keep', 'kia', 'brand', 'good', 'best', 'estimate', 'per', 'three', 'almost', 'purchase', 'sel', 'hope', 'pony', 'epiq', 'park', 'plus', 'available',
              'ev6', 'long', 'high', 'test', 'full', 'limit', "\'re", 'amaze', 'years', 'combination', 'sport', 'right', 'people', 'year', 'update', '2019', 'former', 'sort', 'maverick', 'order', 'ford', 'city',
              'excellent', 'try', 'epa', 'without', 'things', 'number', 'enough', 'pretty', 'may', 'ionic', 'though', '...', 'indeed', 'level', 'se', 'close', 'least', '2021', 'replace', 'leaf', 'days', 'disappoint',
              'i40', 'versions', '300', 'world', 'via', 'anything', 'free', 'please', 'ride', 'jenni', 'day', 'pay', 'behind', 'motor', 'side', 'decent', 'n-line', '8.0', 'winter', 'expect', 'review', 'us', 'sell',
              'could', 'fix', 'tech', 'support', 'let', 'find', 'communicate', 'effective', 'worry', 'money', 'version', 'claim', 'technology', 'ever', 'pull', 'extra', "\'m", 'learn', 'wrong', 'fill', 'reason',
              'koera', 'status', 'flaw', 'wait', 'rev', 'bayon', 'build', 'mean', 'mode', 'diesel', 'market', 'fact', 'major', 'specs', 'mile', 'show', 'choice', 'average', 'techniq', 'live', 'set', 'kmph', 'spec',
              'sound+', 'low', 'pair', 'hit', 'touch', 'week', "\'ve", 'since', 'dealership', '40a', '1.6-litre', 'road', 'ensure', 'allow', 'mini-mpv', 'start', 'course', 'yet', 'thing', 'inside', 'provide', 'actually',
              'launch', 'previous', 'cold', 'hot', '8.5', 'repair', 'change', 'six-speed', 'seven-speed', 'elite', 'value', 'view', 'lower', 'extend', 'weather', 'concern', 'plenty', 'rid', 'haul', 'follow', 'save', 'total',
              'class', 'audi', 'smaller', 'cargo', 'rat', 'quite', 'leave', 'km', 'although', 'whole', 'ultimate', 'buyers', 'end', 'amperage', 'variants', 'nothing', 'bother', 'compare', 'several', 'contrast', 'able', 'real',
              'computer', 'bmw', 'anxiety', 'pack', 'proper', 'i800', 'plan', 'enjoy', 'complete', 'import', 'away', 'sit', 'four', 'instead', 'track', 'bin', 'underneath', 'firm', 'fit', 'eye', 'always', 'open',
              'except', 'overall', 'bolt', 'chevy', 'wife', 'i-pedal', 'default', 'sure', 'division', 'fall', 'easily', '258', 'become', 'genesis', 'roll', 'current', 'nissan', 'thank', 'call', 'continue', 'public',
              'mondeo', 'visit', 'korea', 'seven', 'discovery', 'finally', 'super', 'multilink', 'clean', '7500', 'rather', 'canada', 'already', 'hours', '8k', 'regret', 'luxury', 'mercedes', 'lifetime', 'msrp',
              'move', 'run', 'add', 'lead', 'lease', 'perk', 'reduce', 'credit', 'pant', 'mi', 'terrific', 'head', 'options', 'seriously', 'con', 'strange', 'require', '12v', 'die', 'bad', 'unlike', 'port', 'either', 'grown-up',
              'fresh', '64kwh', 'might', 'ahead', 'convince', 'recall', '50', 'remain', 'turn', 'the', 'and', 'for', 'more', 'provided', 'input', 'based', 'are', 'from', 'that', 'whether', 'includes', 'provided',
              'third', 'operating', 'when', 'shafts', 'configured', 'determined', 'determining', 'session', 'after-burn', 'control', 'line', 'core', 'connected', 'element', 'hev', 'information', 'selectively', 'soc', 'with',
              'shifting', 'agent', 'line', 'effect', 'virtual', 'which','module', 'inner', 'state', 'cells', 'case', 'cell', 'through', 'command', 'where', 'feeling', 'method', 'deicing', 'amount', 'fixedly', 'output', 'load',
              'target', 'controlling', 'characteristics', 'device', 'learning', 'charging', 'apparatus', 'rotating', 'braking', 'component', 'arm', 'member', 'regenerative', 'coolant', 'relay', 'internal', 'having', 'disposed', 'mounted',
              'condition', 'elements', 'main', 'thereof', 'satisfied', 'user', 'calculating', 'according', 'coaxially', 'fourth', 'recirculation', 'connectable', 'request', 'connecting', 'supply', 'detecting', 'traveling'
              'receiving', 'transmitted', 'fee', 'including', 'other', 'each', 'performing', 'during', 'perform', 'rotational', 'predetermined', 'characteristic', 'variable', 'torques', 'sets', 'intercept', 'determine', 'leisure', 'aps', 'interlock',
              'using', 'packs', 'receiving', 'intermediate', 'pulley', 'server', 'conditioning', 'catalyst', 'starting', 'sixth', 'section', 'zone', 'shifted','outputting', 'calculated', 'transferring', 'collected', 'instruction', 'processor',
              'locking', 'infrared', 'expected', 'required', 'controller', 'producing', 'opening', 'parts', 'object', 'performed', 'frequency', 'key', 'basis', 'contact', 'shift', 'gear', 'higher', 'authentication', 'generated', 'electronic',
              'exhaust', 'male', 'being', 'parallel', 'corresponding', 'fifth', 'fixed', 'corresponding', 'inlet', 'parallel', 'between', 'differential', 'operatively']
patents['abstract_token'] = patents['abstract_token'].apply(lambda x: [words for words in x if words not in (words_list)])


# In[ ]:


#Coherence 및 Perplexity score 계산
import gensim
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel

def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    coherence_values = []
    perplexity_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=patents['abstract_token'], dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        perplexity_values.append(model.log_perplexity(corpus))

    return model_list, coherence_values, perplexity_values

def find_optimal_number_of_topics(dictionary, corpus, processed_data):
    limit = 20;
    start = 2;
    step = 2;

    model_list, coherence_values, perplexity_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=processed_data, start=start, limit=limit, step=step)
    x = range(start, limit, step)

    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    plt.plot(x, perplexity_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Perplexity score")
    plt.legend(("perplexity_values"), loc='best')
    plt.show()


# In[ ]:


from gensim import corpora
dictionary = corpora.Dictionary(patents['abstract_token'])
corpus = [dictionary.doc2bow(txt) for txt in patents['abstract_token']]


# In[ ]:


find_optimal_number_of_topics(dictionary, corpus, patents['abstract_token'])


# In[ ]:


#LDA 모델링
NUM_TOPICS = 6
ldamodel = gensim.models.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
topics = ldamodel.print_topics()
for topic in topics:
  print(topic)


# ##### 2) 키워드 분석

# In[ ]:


review_token = []
for i in range(len(patents['abstract_token'])):
   review_token.extend(patents['abstract_token'][i])

token = list(filter(lambda word: word not in words_list, review_token))

from collections import Counter

token_counter = Counter(token)

new_wordInfo = dict()
print("\n--Token : Freq--")
for tags, counts in token_counter.most_common(50):
    new_wordInfo[tags] = counts
    print("%6s : %d" % (tags, counts))


# In[ ]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(relative_scaling = 0.2,
                      background_color='white',
                      colormap="twilight"
                      ).generate_from_frequencies(new_wordInfo)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud)
plt.axis("off")
#plt.savefig("before_gpt_wordcloud.png")
plt.show()


# ##### 3) 특허 군집화

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(patents['abstract'])

lsa = TruncatedSVD(n_components=100, random_state=42)
tfidf_matrix_lsa = lsa.fit_transform(tfidf_matrix)

sum_of_squared_distances = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(tfidf_matrix_lsa)
    sum_of_squared_distances.append(kmeans.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method for Optimal k')
plt.show()

optimal_num_clusters = 3

kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
kmeans.fit(tfidf_matrix_lsa)

patents['cluster'] = kmeans.labels_
print(patents[['title', 'cluster']])


# In[ ]:


patents['cluster'].nunique()


# In[ ]:


stop_words = stopwords.words('english')
words_list = ["\'s", "fe", "car", "hyundai", 'one', 'miles', 'ioniq', 'kona', 'i30n', 'maybe', "n\'t", 'like', 'santa', 'tucson', "get", 'ev', "--", 'would', 'drive', 'vw', "``", '2022', 'two', 'also',
              'even', 'i20n', 'rn22e', 'cars', "\'\'", 'much', 'say', 'go', 'tesla', 'better', '..', 'i30', 'electric', 'take', 'make', 'every', 'around', 'come', 'work', 'lot', 'look', 'vehicle',
              'something', 'new', 'buy', 'new', 'really', 'great', 'love', 'well', 'come', 'need','little', 'give', 'think', 'everything', 'see', 'second', 'first', 'nexo', 'evs', '2024', 'happier',
              'help', 'point', 'feature', 'standard', 'system', 'model', 'i10', 'hybrid', 'still', 'include', 'put', 'rival', '2023', 'feel', 'never', 'rear', 'front', 'many', 'ioniq5', 'use', 'back',
              'part', 'want', 'truck', 'way', 'offer', 'pickup', 'know', 'range', 'issue', 'problem', 'zero', 'rodgers', 'yes', 'seem', 'perfect', 'different', 'unit', 'easy', 'less', 'hear', 'deal',
              'dealer', 'degrees', 'top', 'months', 'suv', '-the', 'assist', 'bring', 'gt', 'base', 'another', 'longer', 'i20', 'nice', 'fiesta', 'bite', 'life', 'cent', 'far', 'hydrogen', 'fcv',
              'tell', 'van', 'staria', 'uk', 'keep', 'kia', 'brand', 'good', 'best', 'estimate', 'per', 'three', 'almost', 'purchase', 'sel', 'hope', 'pony', 'epiq', 'park', 'plus', 'available',
              'ev6', 'long', 'high', 'test', 'full', 'limit', "\'re", 'amaze', 'years', 'combination', 'sport', 'right', 'people', 'year', 'update', '2019', 'former', 'sort', 'maverick', 'order', 'ford', 'city',
              'excellent', 'try', 'epa', 'without', 'things', 'number', 'enough', 'pretty', 'may', 'ionic', 'though', '...', 'indeed', 'level', 'se', 'close', 'least', '2021', 'replace', 'leaf', 'days', 'disappoint',
              'i40', 'versions', '300', 'world', 'via', 'anything', 'free', 'please', 'ride', 'jenni', 'day', 'pay', 'behind', 'motor', 'side', 'decent', 'n-line', '8.0', 'winter', 'expect', 'review', 'us', 'sell',
              'could', 'fix', 'tech', 'support', 'let', 'find', 'communicate', 'effective', 'worry', 'money', 'version', 'claim', 'technology', 'ever', 'pull', 'extra', "\'m", 'learn', 'wrong', 'fill', 'reason',
              'koera', 'status', 'flaw', 'wait', 'rev', 'bayon', 'build', 'mean', 'mode', 'diesel', 'market', 'fact', 'major', 'specs', 'mile', 'show', 'choice', 'average', 'techniq', 'live', 'set', 'kmph', 'spec',
              'sound+', 'low', 'pair', 'hit', 'touch', 'week', "\'ve", 'since', 'dealership', '40a', '1.6-litre', 'road', 'ensure', 'allow', 'mini-mpv', 'start', 'course', 'yet', 'thing', 'inside', 'provide', 'actually',
              'launch', 'previous', 'cold', 'hot', '8.5', 'repair', 'change', 'six-speed', 'seven-speed', 'elite', 'value', 'view', 'lower', 'extend', 'weather', 'concern', 'plenty', 'rid', 'haul', 'follow', 'save', 'total',
              'class', 'audi', 'smaller', 'cargo', 'rat', 'quite', 'leave', 'km', 'although', 'whole', 'ultimate', 'buyers', 'end', 'amperage', 'variants', 'nothing', 'bother', 'compare', 'several', 'contrast', 'able', 'real',
              'computer', 'bmw', 'anxiety', 'pack', 'proper', 'i800', 'plan', 'enjoy', 'complete', 'import', 'away', 'sit', 'four', 'instead', 'track', 'bin', 'underneath', 'firm', 'fit', 'eye', 'always', 'open',
              'except', 'overall', 'bolt', 'chevy', 'wife', 'i-pedal', 'default', 'sure', 'division', 'fall', 'easily', '258', 'become', 'genesis', 'roll', 'current', 'nissan', 'thank', 'call', 'continue', 'public',
              'mondeo', 'visit', 'korea', 'seven', 'discovery', 'finally', 'super', 'multilink', 'clean', '7500', 'rather', 'canada', 'already', 'hours', '8k', 'regret', 'luxury', 'mercedes', 'lifetime', 'msrp',
              'move', 'run', 'add', 'lead', 'lease', 'perk', 'reduce', 'credit', 'pant', 'mi', 'terrific', 'head', 'options', 'seriously', 'con', 'strange', 'require', '12v', 'die', 'bad', 'unlike', 'port', 'either', 'grown-up',
              'fresh', '64kwh', 'might', 'ahead', 'convince', 'recall', '50', 'remain', 'turn', 'the', 'and', 'for', 'more', 'provided', 'input', 'based', 'are', 'from', 'that', 'whether', 'includes', 'provided',
              'third', 'operating', 'when', 'shafts', 'configured', 'determined', 'determining', 'session', 'after-burn', 'control', 'line', 'core', 'connected', 'element', 'hev', 'information', 'selectively', 'soc', 'with',
              'shifting', 'agent', 'line', 'effect', 'virtual', 'which','module', 'inner', 'state', 'cells', 'case', 'cell', 'through', 'command', 'where', 'feeling', 'method', 'deicing', 'amount', 'fixedly', 'output', 'load',
              'target', 'controlling', 'characteristics', 'device', 'learning', 'charging', 'apparatus', 'rotating', 'braking', 'component', 'arm', 'member', 'regenerative', 'coolant', 'relay', 'internal', 'having', 'disposed', 'mounted',
              'condition', 'elements', 'main', 'thereof', 'satisfied', 'user', 'calculating', 'according', 'coaxially', 'fourth', 'recirculation', 'connectable', 'request', 'connecting', 'supply', 'detecting', 'traveling'
              'receiving', 'transmitted', 'fee', 'including', 'other', 'each', 'performing', 'during', 'perform', 'rotational', 'predetermined', 'characteristic', 'variable', 'torques', 'sets', 'intercept', 'determine', 'leisure', 'aps', 'interlock',
              'using', 'packs', 'receiving', 'intermediate', 'pulley', 'server', 'conditioning', 'catalyst', 'starting', 'sixth', 'section', 'zone', 'shifted','outputting', 'calculated', 'transferring', 'collected', 'instruction', 'processor',
              'locking', 'infrared', 'expected', 'required', 'controller', 'producing', 'opening', 'parts', 'object', 'performed', 'frequency', 'key', 'basis', 'contact', 'shift', 'gear', 'higher', 'authentication', 'generated', 'electronic',
              'exhaust', 'male', 'being', 'parallel', 'corresponding', 'fifth', 'fixed', 'corresponding', 'inlet', 'parallel', 'between', 'differential', 'operatively', 'supplying', 'externally']
stop_words.extend(words_list)

cluster_keywords = {}
for cluster_id in range(optimal_num_clusters):
    cluster_patents = patents[patents['cluster'] == cluster_id]
    cluster_abstracts = cluster_patents['abstract']
    cluster_tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words=stop_words)
    cluster_tfidf_matrix = cluster_tfidf_vectorizer.fit_transform(cluster_abstracts)
    feature_names = cluster_tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = cluster_tfidf_matrix.sum(axis=0).A1
    keywords = [feature_names[i] for i in tfidf_scores.argsort()[-10:][::-1]]
    cluster_keywords[cluster_id] = keywords

for cluster_id, keywords in cluster_keywords.items():
    print(f"클러스터 {cluster_id}의 주요 토픽:")
    print(keywords)


# In[ ]:


patents.info()


# #### 2.3 리뷰 데이터와 특허 데이터 상관분석

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

patents_texts = patents['title'].fillna('') + " " + patents['abstract'].fillna('')
reviews_texts = review['txt'].fillna('')

vectorizer = TfidfVectorizer(stop_words=stop_words)
patents_tfidf = vectorizer.fit_transform(patents_texts)
reviews_tfidf = vectorizer.transform(reviews_texts)

cosine_similarities = cosine_similarity(patents_tfidf, reviews_tfidf)

# 유사도가 0.6 이상인 경우만 필터링하여 출력
threshold = 0.6
filtered_similarities = []

print("Patents and reviews with cosine similarity >= 0.6:")
for i, similarities in enumerate(cosine_similarities):
    for j, similarity in enumerate(similarities):
        if similarity >= threshold:
            filtered_similarities.append((i, j, similarity))
            print(f"Patent {i} and Review {j}: {similarity:.4f}")


# In[ ]:


filtered_df = pd.DataFrame(filtered_similarities, columns=['Patent Index', 'Review Index', 'Cosine Similarity'])
filtered_df.sort_values(by='Cosine Similarity', ascending=False)


# In[ ]:


review.iloc[75]['sentiment']


# In[ ]:


filtered_df['Review Index'].sort_values()


# In[ ]:


for i in filtered_df['Review Index'].sort_values():
  print(i, review.iloc[i]['sentiment'])


# In[ ]:


filtered_df['sentiment'] = ['Negative', 'Positive', 'Negative', 'Negative', 'Positive', 'Negative', 'Negative', 'Positive', 'Negative', 'Negative', 'Negative']
filtered_df.sort_values(by='Cosine Similarity', ascending=False)


# In[ ]:


stop_words = stopwords.words('english')
words_list = ["\'s", "fe", "car", "hyundai", 'one', 'miles', 'ioniq', 'kona', 'i30n', 'maybe', "n\'t", 'like', 'santa', 'tucson', "get", 'ev', "--", 'would', 'drive', 'vw', "``", '2022', 'two', 'also',
              'even', 'i20n', 'rn22e', 'cars', "\'\'", 'much', 'say', 'go', 'tesla', 'better', '..', 'i30', 'electric', 'take', 'make', 'every', 'around', 'come', 'work', 'lot', 'look', 'vehicle',
              'something', 'new', 'buy', 'new', 'really', 'great', 'love', 'well', 'come', 'need','little', 'give', 'think', 'everything', 'see', 'second', 'first', 'nexo', 'evs', '2024', 'happier',
              'help', 'point', 'feature', 'standard', 'system', 'model', 'i10', 'hybrid', 'still', 'include', 'put', 'rival', '2023', 'feel', 'never', 'rear', 'front', 'many', 'ioniq5', 'use', 'back',
              'part', 'want', 'truck', 'way', 'offer', 'pickup', 'know', 'range', 'issue', 'problem', 'zero', 'rodgers', 'yes', 'seem', 'perfect', 'different', 'unit', 'easy', 'less', 'hear', 'deal',
              'dealer', 'degrees', 'top', 'months', 'suv', '-the', 'assist', 'bring', 'gt', 'base', 'another', 'longer', 'i20', 'nice', 'fiesta', 'bite', 'life', 'cent', 'far', 'hydrogen', 'fcv',
              'tell', 'van', 'staria', 'uk', 'keep', 'kia', 'brand', 'good', 'best', 'estimate', 'per', 'three', 'almost', 'purchase', 'sel', 'hope', 'pony', 'epiq', 'park', 'plus', 'available',
              'ev6', 'long', 'high', 'test', 'full', 'limit', "\'re", 'amaze', 'years', 'combination', 'sport', 'right', 'people', 'year', 'update', '2019', 'former', 'sort', 'maverick', 'order', 'ford', 'city',
              'excellent', 'try', 'epa', 'without', 'things', 'number', 'enough', 'pretty', 'may', 'ionic', 'though', '...', 'indeed', 'level', 'se', 'close', 'least', '2021', 'replace', 'leaf', 'days', 'disappoint',
              'i40', 'versions', '300', 'world', 'via', 'anything', 'free', 'please', 'ride', 'jenni', 'day', 'pay', 'behind', 'motor', 'side', 'decent', 'n-line', '8.0', 'winter', 'expect', 'review', 'us', 'sell',
              'could', 'fix', 'tech', 'support', 'let', 'find', 'communicate', 'effective', 'worry', 'money', 'version', 'claim', 'technology', 'ever', 'pull', 'extra', "\'m", 'learn', 'wrong', 'fill', 'reason',
              'koera', 'status', 'flaw', 'wait', 'rev', 'bayon', 'build', 'mean', 'mode', 'diesel', 'market', 'fact', 'major', 'specs', 'mile', 'show', 'choice', 'average', 'techniq', 'live', 'set', 'kmph', 'spec',
              'sound+', 'low', 'pair', 'hit', 'touch', 'week', "\'ve", 'since', 'dealership', '40a', '1.6-litre', 'road', 'ensure', 'allow', 'mini-mpv', 'start', 'course', 'yet', 'thing', 'inside', 'provide', 'actually',
              'launch', 'previous', 'cold', 'hot', '8.5', 'repair', 'change', 'six-speed', 'seven-speed', 'elite', 'value', 'view', 'lower', 'extend', 'weather', 'concern', 'plenty', 'rid', 'haul', 'follow', 'save', 'total',
              'class', 'audi', 'smaller', 'cargo', 'rat', 'quite', 'leave', 'km', 'although', 'whole', 'ultimate', 'buyers', 'end', 'amperage', 'variants', 'nothing', 'bother', 'compare', 'several', 'contrast', 'able', 'real',
              'computer', 'bmw', 'anxiety', 'pack', 'proper', 'i800', 'plan', 'enjoy', 'complete', 'import', 'away', 'sit', 'four', 'instead', 'track', 'bin', 'underneath', 'firm', 'fit', 'eye', 'always', 'open',
              'except', 'overall', 'bolt', 'chevy', 'wife', 'i-pedal', 'default', 'sure', 'division', 'fall', 'easily', '258', 'become', 'genesis', 'roll', 'current', 'nissan', 'thank', 'call', 'continue', 'public',
              'mondeo', 'visit', 'korea', 'seven', 'discovery', 'finally', 'super', 'multilink', 'clean', '7500', 'rather', 'canada', 'already', 'hours', '8k', 'regret', 'luxury', 'mercedes', 'lifetime', 'msrp',
              'move', 'run', 'add', 'lead', 'lease', 'perk', 'reduce', 'credit', 'pant', 'mi', 'terrific', 'head', 'options', 'seriously', 'con', 'strange', 'require', '12v', 'die', 'bad', 'unlike', 'port', 'either', 'grown-up',
              'fresh', '64kwh', 'might', 'ahead', 'convince', 'recall', '50', 'remain', 'turn', 'the', 'and', 'for', 'more', 'provided', 'input', 'based', 'are', 'from', 'that', 'whether', 'includes', 'provided',
              'third', 'operating', 'when', 'shafts', 'configured', 'determined', 'determining', 'session', 'after-burn', 'control', 'line', 'core', 'connected', 'element', 'hev', 'information', 'selectively', 'soc', 'with',
              'shifting', 'agent', 'line', 'effect', 'virtual', 'which','module', 'inner', 'state', 'cells', 'case', 'cell', 'through', 'command', 'where', 'feeling', 'method', 'deicing', 'amount', 'fixedly', 'output', 'load',
              'target', 'controlling', 'characteristics', 'device', 'learning', 'charging', 'apparatus', 'rotating', 'braking', 'component', 'arm', 'member', 'regenerative', 'coolant', 'relay', 'internal', 'having', 'disposed', 'mounted',
              'condition', 'elements', 'main', 'thereof', 'satisfied', 'user', 'calculating', 'according', 'coaxially', 'fourth', 'recirculation', 'connectable', 'request', 'connecting', 'supply', 'detecting', 'traveling', 'disclosure',
              'receiving', 'transmitted', 'fee', 'including', 'other', 'each', 'performing', 'during', 'perform', 'rotational', 'predetermined', 'characteristic', 'variable', 'torques', 'sets', 'intercept', 'determine', 'leisure', 'aps', 'interlock',
              'using', 'packs', 'receiving', 'intermediate', 'pulley', 'server', 'conditioning', 'catalyst', 'starting', 'sixth', 'section', 'zone', 'shifted','outputting', 'calculated', 'transferring', 'collected', 'instruction', 'processor',
              'locking', 'infrared', 'expected', 'required', 'controller', 'producing', 'opening', 'parts', 'object', 'performed', 'frequency', 'key', 'basis', 'contact', 'shift', 'gear', 'higher', 'authentication', 'generated', 'electronic',
              'exhaust', 'male', 'being', 'parallel', 'corresponding', 'fifth', 'fixed', 'corresponding', 'inlet', 'parallel', 'between', 'differential', 'operatively', 'supplying', 'externally', 'demanded', 'drivable', 'kw', '10', '11', 'acquired',
              'ended', 'resolved', 'ago', 'discharged', 'died', 'shop', 'waiting', 'comes', '000', '100', '2020', 'runaway', 'outputs', 'step', '35', 'advise', 'charger', 'respective', 'values', 'original', 'owner', 'especially', 'efficiently',
              'deplete', 'easier', 'comfortable', 'among', 'acquiring', 'stable', 'unstable', 'controls', 'detects', 'happens', 'explain', 'fatal', 'completion', 'conditions','determines','highly','compute', 'equalizing', 'detectors', 'computed']
stop_words.extend(words_list)

def extract_keywords(text, stopwords, num_keywords=5):
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    keywords = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}
    sorted_keywords = sorted(keywords.items(), key=lambda item: item[1], reverse=True)
    return [keyword for keyword, score in sorted_keywords[:num_keywords]]

patents_texts = patents['title'].fillna('') + " " + patents['abstract'].fillna('')
reviews_texts = review['txt'].fillna('')

for index, row in filtered_df.iterrows():
    patent_index = row['Patent Index']
    review_index = row['Review Index']

    patent_text = patents_texts[patent_index]
    review_text = reviews_texts[review_index]

    patent_keywords = extract_keywords(patent_text, stop_words)
    review_keywords = extract_keywords(review_text, stop_words)

    print(f"Patent {patent_index} and Review {review_index} (Cosine Similarity: {row['Cosine Similarity']}, Sentiment: {row['sentiment']})")
    print(f"Patent Keywords: {patent_keywords}")
    print(f"Review Keywords: {review_keywords}")
    print("\n")


# #### 2.5 추천시스템
# - 콘텐츠 기반 필터링 (Content-based Filtering)
# : 1) 특허와 리뷰의 텍스트 내용을 분석하여 각각의 특성 벡터를 생성합니다.
# 2) 이 특성 벡터를 바탕으로 유사한 특허나 리뷰를 추천합니다.
# 3) 예를 들어, 특정 기술 영역의 특허와 유사한 내용을 가진 리뷰를 찾아 소비자 니즈를 파악할 수 있습니다.

# In[ ]:


review.head(3)


# In[ ]:


patents.head(3)


# In[ ]:


patent = patents[['abstract', 'cite']]
patent.head(3)


# In[ ]:


for i in range(len(patent)):
  if type(patent.iloc[i]['abstract']) != type(patent.iloc[0]['abstract']):
    print(i)


# In[ ]:


patent.iloc[127]


# In[ ]:


patent.drop(127, axis=0, inplace=True)
patent.info()


# In[ ]:


import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec

stop_words = list(stopwords.words('english'))
stop_words.extend(['without', 'hyundais', 'would', 'use', 'start', 'state', 'got', 'amount',
                    'really', 'change', 'road', 'feel', 'ride', 'youre', 'system', 'going', 'ev',
                    'use', 'back', 'test', 'people', 'high', 'things', 'cars', 'available', 'little',
                   'dont', 'car', 'around', 'point', 'enough', 'take', 'need', 'something', 'one',
                   'im', 'buy', 'go', 'driver', 'weight', 'lot', 'way', 'doesnt', 'used', 'looking',
                   'fuel', 'great', 'version', 'level', 'screen', 'value', 'body', 'work', 'every',
                   'even', 'signal', 'long', 'air', 'cant', 'thats', 'though', 'make', 'top', 'auto',
                   'bhp', 'turn', 'time', 'digital', 'want', 'case', 'less', 'big', 'better', 'current',
                   'family', 'however', 'looks', 'experience', 'best', 'camera', 'year', 'fast', 'easy',
                   'wheels', 'class', 'home', 'kia', 'still', 'styling', 'gets', 'much', 'many',
                   'could', 'using', 'energy', 'suv', 'kw', 'evs', 'sound', 'get', 'well', 'theres',
                   'charging', 'bit', 'lane', 'km', 'pretty', 'parking', 'years', 'inch', 'new',
                   'makes', 'come', 'tesla', 'look', 'isnt', 'fun', 'inside', 'also', 'like', 'price',
                   'good', 'rear', 'model', 'ioniq', 'wireless', 'love', 'hyundai', 'far', 'full',
                   'quite', 'front', 'trim', 'tech', 'comfortable', 'assist', 'small', 'range', 'drive',
                   'comes', 'youll', 'based', 'kona', 'miles', 'mode', 'plenty', 'models', 'standard',
                   'drivers', 'behind', 'unit', 'seats', 'electric', 'features', 'least',
                   'technology', 'market', 'getting', 'centre', 'virtual', 'base', 'line', 'position',
                   'per', 'provided', 'plus', 'vehicles', 'thing', 'average', 'side', 'hybrid', 'limited',
                   'console', 'premium', 'collision', 'keep', 'kwh', 'right', 'feels', 'litre', 'according',
                   'offers', 'whether', 'warning', 'part', 'electrics', 'crossover', 'easily', 'determining',
                   'excellent', 'information', 'buttons', 'light', 'costs', 'includes', 'especially', 'konas',
                   'gas', 'three', 'said', 'may', 'changes', 'think', 'determined', 'device', 'touchscreen',
                   'controls', 'forward', 'method', 'see', 'pack', 'vehicle', 'sel', 'low', 'regular', 'center',
                   'trip', 'expensive', 'two', 'quick', 'condition', 'hours', 'petrol', 'testing', 'might',
                   'service', 'pedal', 'option', 'voltage', 'hatch', 'sport', 'quality', 'engine', 'end',
                   'smooth', 'another', 'fact', 'torque', 'door', 'charge', 'controller', 'warranty', 'operation',
                   'overall', 'yet', 'dc', 'module', 'headlights', 'almost', 'worth', 'brake', 'driving',
                   'cent', 'comfort', 'cooling', 'equipment', 'including', 'include', 'se', 'feature', 'awd', 'track',
                   'hatchback', 'brakes', 'audio', 'actually', 'spot', 'controlling', 'automatic', 'systems', 'floor',
                   'santa', 'suspension', 'single', 'wont', 'ford', 'along', 'powertrain', 'turns', 'practical', 'plurality',
                   'found', 'regenerative', 'blind', 'led', 'slightly', 'station', 'monitoring', 'traffic', 'didnt',
                   'alloy', 'mph', 'four', 'larger', 'months', 'say', 'extra', 'mache', 'leather', 'choice',
                   'drives', 'distance', 'configured', 'apparatus', 'real', 'smaller', 'minutes', 'predetermined', 'times',
                   'heat', 'room', 'find', 'key', 'day', 'course', 'latest', 'world', 'heater', 'expect', 'across',
                   'different', 'instrument', 'shift', 'generation', 'performing', 'money', 'rating', 'motor',
                   'size', 'thanks', 'combustion', 'priced', 'via', 'panel', 'roads', 'flat', 'modes', 'required',
                   'allows', 'instead', 'longer', 'help', 'climate', 'android', 'order', 'fe', 'levels', 'main',
                   'tyres', 'since', 'hot', 'realworld', 'among', 'tight', 'update', 'port', 'given', 'temperature',
                   'corners', 'starting', 'know', 'route', 'us', 'power', 'safe', 'manual', 'impressive', 'sense',
                   'cheaper', 'anything', 'target', 'highlander', 'definitely', 'rne', 'points', 'eco', 'communication',
                   'city', 'remains', 'techniq', 'large', 'stop', 'seconds', 'cost', 'give', 'everything', 'kmh',
                   'cell', 'parts', 'close', 'previous', 'variable', 'offer', 'sure', 'bigger', 'last', 'carplay',
                   'kwnm', 'problem', 'calculating', 'ultimate', 'nissan', 'apple', 'rest', 'rivals', 'claimed'])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words and len(token) >= 2]
    return tokens

patent_tokens = [preprocess_text(text) for text in patent['abstract']]
review_tokens = [preprocess_text(text) for text in review['txt']]

patent_tokens = [tokens for tokens in patent_tokens if tokens]
review_tokens = [tokens for tokens in review_tokens if tokens]

model = Word2Vec(patent_tokens + review_tokens, vector_size=100, window=5, min_count=1, workers=4)

# 'display'와 유사한 단어 찾기
if 'display' in model.wv:
    similar_words = model.wv.most_similar('display', topn=20)
    similar_words
else:
    print("'display' is not in the vocabulary.")


# In[ ]:


similar_words


# In[ ]:


similarity_matrix = cosine_similarity(all_vectors)
print("\nSimilarity Matrix:")
print(similarity_matrix)

doc_id = 0
similarities = similarity_matrix[doc_id]
most_similar = similarities.argsort()[-2]
print(f"\nMost similar document to document {doc_id}:")
print(result_df.iloc[most_similar])


# In[ ]:


patent.info()


# In[ ]:


print("Patent vectors shape:", patent_vectors.shape)
print("Review vectors shape:", review_vectors.shape)

patent_features = common_keywords + ['cite', 'doc_type']
review_features = common_keywords + ['compound'] + list(review_category.columns) + ['doc_type']

print("Patent features:", patent_features)
print("Review features:", review_features)


# #### 2.6 협업필터링

# In[ ]:


get_ipython().system('pip install implicit')


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


# In[ ]:


patents


# In[ ]:


review


# In[ ]:


def content_based_filtering(text1, text2):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(text1 + text2)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

all_texts = patents['abstract'].tolist() + review['txt'].tolist()
similarity_matrix = content_based_filtering(all_texts, all_texts)


# ##### 특허 초록의 nan값 제거

# In[ ]:


for i in range(len(patents['abstract'])):
  if type(patents['abstract'].iloc[0]) != type(patents['abstract'].iloc[i]):
    print(i)


# In[ ]:


patents['abstract'].iloc[127]


# In[ ]:


patents.drop(127, axis=0, inplace=True)


# In[ ]:


for i in range(len(patents['abstract'])):
  if type(patents['abstract'].iloc[0]) != type(patents['abstract'].iloc[i]):
    print(i)


# ##### 키워드 매칭 사용 방법 2

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

review['review_id'] = review.index

combined_texts = list(patents['abstract']) + list(review['txt'])

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(combined_texts)

num_patents = len(patents)
num_reviews = len(review)
patent_tfidf = tfidf_matrix[:num_patents]
review_tfidf = tfidf_matrix[num_patents:]

cosine_similarities = cosine_similarity(patent_tfidf, review_tfidf)

def recommend_patents_for_review(review_id, top_n=5):
    try:
        review_index = review.index[review['review_id'] == review_id].tolist()[0]
    except IndexError:
        return pd.DataFrame()

    similarity_scores = cosine_similarities[:, review_index]
    top_patent_indices = similarity_scores.argsort()[-top_n:][::-1]
    return patents.iloc[top_patent_indices]

top_positive_reviews = review.nlargest(5, 'compound')
top_negative_reviews = review.nsmallest(5, 'compound')

print("Top patents for positive reviews:")
for idx, review_data in top_positive_reviews.iterrows():
    print(f"\nReview ID {review_data['review_id']} with sentiment score {review_data['compound']}:")
    recommended_patents = recommend_patents_for_review(review_data['review_id'])['publication number']
    print(recommended_patents)

print("\nTop patents for negative reviews:")
for idx, review_data in top_negative_reviews.iterrows():
    print(f"\nReview ID {review_data['review_id']} with sentiment score {review_data['compound']}:")
    recommended_patents = recommend_patents_for_review(review_data['review_id'])['publication number']
    print(recommended_patents)


# In[ ]:


print(patents[patents['publication number']=='US11161423']['title'].values)
print(patents[patents['publication number']=='US11161426']['title'].values)
print(patents[patents['publication number']=='US11613153']['title'].values)
print(patents[patents['publication number']=='US11820374']['title'].values)
print(patents[patents['publication number']=='US11787302']['title'].values)
print(patents[patents['publication number']=='US11762354']['title'].values)


# #### 2.7 이상치 탐지

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from keras.models import Model, Sequential
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

patents.dropna(subset=['abstract', 'cite', 'cpc'], inplace=True)
review.dropna(subset=['txt'], inplace=True)

vectorizer = TfidfVectorizer(stop_words='english')
patent_tfidf = vectorizer.fit_transform(patents['abstract'])
review_tfidf = vectorizer.fit_transform(review['txt'])

patent_tfidf_df = pd.DataFrame(patent_tfidf.toarray(), index=patents.index)
review_tfidf_df = pd.DataFrame(review_tfidf.toarray(), index=review.index)

patent_tfidf_df['citations'] = patents['cite'].values

review_tfidf_df['sentiment_score'] = review['compound'].values

def detect_outliers_z_score(data):
    z_scores = np.abs((data - data.mean()) / data.std())
    return data[z_scores > 3]

def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]

# 특허의 기술적 특성과 리뷰 감성 점수를 결합하여 Z-점수 및 IQR 계산
combined_data = pd.concat([patent_tfidf_df['citations'], review_tfidf_df['sentiment_score']], axis=1, join='inner')

outliers_z_score = detect_outliers_z_score(combined_data)
outliers_iqr = detect_outliers_iqr(combined_data)

print("Z-Score Outliers:\n", outliers_z_score)
print("\nIQR Outliers:\n", outliers_iqr)


# In[ ]:


iso_forest = IsolationForest(contamination=0.1)
iso_forest.fit(combined_data)
outliers_iso_forest = combined_data[iso_forest.predict(combined_data) == -1]

oc_svm = OneClassSVM(nu=0.1)
oc_svm.fit(combined_data)
outliers_oc_svm = combined_data[oc_svm.predict(combined_data) == -1]


# In[ ]:


input_dim = combined_data.shape[1]
encoding_dim = 14
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="tanh")(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

scaler = StandardScaler()
combined_data_scaled = scaler.fit_transform(combined_data)
autoencoder.fit(combined_data_scaled, combined_data_scaled, epochs=50, batch_size=32, shuffle=True, validation_split=0.1, verbose=0)

reconstructions = autoencoder.predict(combined_data_scaled)
reconstruction_errors = np.mean(np.abs(reconstructions - combined_data_scaled), axis=1)

threshold = np.percentile(reconstruction_errors, 95)
outliers_autoencoder = combined_data[reconstruction_errors > threshold]


# 기술적 가치를 가진 특허(인용수가 높고,  cpc 코드가 다양한)임에도 소비자 반응이 부정적인 경우나 단순한 특허임에도 긍정적인 반응을 보이는 특허를 선별

# In[ ]:


patents.dropna(subset=['abstract', 'cite', 'cpc'], inplace=True)
review.dropna(subset=['txt'], inplace=True)

patents['cpc_code_count'] = patents['cpc'].apply(lambda x: len(x.split(',')))
patents['technical_value'] = patents['cite'] + patents['cpc_code_count']
patent_tfidf_df['technical_value'] = patents.set_index(patent_tfidf_df.index)['technical_value']

review_tfidf_df['sentiment_score'] = review['compound'].values

combined_data = pd.concat([patent_tfidf_df['technical_value'], review_tfidf_df['sentiment_score']], axis=1, join='inner')

def detect_outliers_z_score(data):
    z_scores = np.abs((data - data.mean()) / data.std())
    return data[z_scores > 3]

def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]

outliers_z_score = detect_outliers_z_score(combined_data)
outliers_iqr = detect_outliers_iqr(combined_data)

# 기술적 가치가 높은 특허 중 소비자 반응이 부정적인 경우
high_value_negative_reviews = combined_data[(combined_data['technical_value'] > combined_data['technical_value'].mean()) & (combined_data['sentiment_score'] < 0)]

# 기술적 가치가 낮은 특허 중 소비자 반응이 긍정적인 경우
low_value_positive_reviews = combined_data[(combined_data['technical_value'] <= combined_data['technical_value'].mean()) & (combined_data['sentiment_score'] > 0)]

print("High Value Patents with Negative Reviews:")
print(high_value_negative_reviews)

print("\nLow Value Patents with Positive Reviews:")
print(low_value_positive_reviews)


# In[ ]:


patents.dropna(subset=['abstract', 'cite', 'cpc'], inplace=True)
review.dropna(subset=['txt'], inplace=True)

patents['cpc_code_count'] = patents['cpc'].apply(lambda x: len(x.split(',')))
patents['technical_value'] = patents['cite'] + patents['cpc_code_count']
patent_tfidf_df['technical_value'] = patents.set_index(patent_tfidf_df.index)['technical_value']

review_tfidf_df['sentiment_score'] = review['compound'].values

combined_data = pd.concat([patent_tfidf_df['technical_value'], review_tfidf_df['sentiment_score']], axis=1, join='inner')

def detect_outliers_z_score(data):
    z_scores = np.abs((data - data.mean()) / data.std())
    return data[z_scores > 3]

def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]

outliers_z_score = detect_outliers_z_score(combined_data)
outliers_iqr = detect_outliers_iqr(combined_data)

# 기술적 가치가 높은 특허 중 소비자 반응이 부정적인 경우
high_value_negative_reviews = combined_data[(combined_data['technical_value'] > combined_data['technical_value'].mean()) & (combined_data['sentiment_score'] < 0)]

# 기술적 가치가 낮은 특허 중 소비자 반응이 긍정적인 경우
low_value_positive_reviews = combined_data[(combined_data['technical_value'] <= combined_data['technical_value'].mean()) & (combined_data['sentiment_score'] > 0)]

print("High Value Patents with Negative Reviews:")
print(high_value_negative_reviews)

print("\nLow Value Patents with Positive Reviews:")
print(low_value_positive_reviews)


# In[ ]:


# 수정된 코드
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

patents.dropna(subset=['abstract', 'cite', 'cpc'], inplace=True)
review.dropna(subset=['txt'], inplace=True)

patents['cpc_code_count'] = patents['cpc'].apply(lambda x: len(x.split(',')))

scaler = MinMaxScaler()

patents['cite_scaled'] = scaler.fit_transform(patents[['cite']])
patents['cpc_code_count_scaled'] = scaler.fit_transform(patents[['cpc_code_count']])

patents['technical_value'] = patents['cite_scaled'] + patents['cpc_code_count_scaled']

patent_tfidf_df['technical_value'] = patents.set_index(patent_tfidf_df.index)['technical_value']

review_tfidf_df['sentiment_score'] = review['compound'].values

combined_data = pd.concat([patent_tfidf_df['technical_value'], review_tfidf_df['sentiment_score']], axis=1, join='inner')

def detect_outliers_z_score(data):
    z_scores = np.abs((data - data.mean()) / data.std())
    return data[z_scores > 3]

def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]

outliers_z_score = detect_outliers_z_score(combined_data)
outliers_iqr = detect_outliers_iqr(combined_data)

# 기술적 가치가 높은 특허 중 소비자 반응이 부정적인 경우
high_value_negative_reviews = combined_data[(combined_data['technical_value'] > combined_data['technical_value'].mean()) & (combined_data['sentiment_score'] < -0.5)]

# 기술적 가치가 낮은 특허 중 소비자 반응이 긍정적인 경우
low_value_positive_reviews = combined_data[(combined_data['technical_value'] <= combined_data['technical_value'].mean()) & (combined_data['sentiment_score'] > 0.5)]

print("High Value Patents with Negative Reviews:")
print(high_value_negative_reviews)

print("\nLow Value Patents with Positive Reviews:")
print(low_value_positive_reviews)


# In[ ]:


# 수정된 코드2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

patents.dropna(subset=['abstract', 'cite', 'cpc'], inplace=True)
review.dropna(subset=['txt'], inplace=True)

patents['cpc_code_count'] = patents['cpc'].apply(lambda x: len(x.split(',')))

scaler = MinMaxScaler()

patents['cite_scaled'] = scaler.fit_transform(patents[['cite']])
patents['cpc_code_count_scaled'] = scaler.fit_transform(patents[['cpc_code_count']])

patents['technical_value'] = patents['cite_scaled'] + patents['cpc_code_count_scaled']

patent_tfidf_df['technical_value'] = patents.set_index(patent_tfidf_df.index)['technical_value']

review_tfidf_df['sentiment_score'] = review['compound'].values

combined_data = pd.concat([patent_tfidf_df['technical_value'], review_tfidf_df['sentiment_score']], axis=1, join='inner')

def detect_outliers_z_score(data):
    z_scores = np.abs((data - data.mean()) / data.std())
    return data[z_scores > 3]

def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]

outliers_z_score = detect_outliers_z_score(combined_data)
outliers_iqr = detect_outliers_iqr(combined_data)

# 기술적 가치가 높은 특허 중 소비자 반응이 부정적인 경우
high_value_negative_reviews = combined_data[(combined_data['technical_value'] > combined_data['technical_value'].mean()) & (combined_data['sentiment_score'] < -0.5)]

# 기술적 가치가 낮은 특허 중 소비자 반응이 긍정적인 경우
low_value_positive_reviews = combined_data[(combined_data['technical_value'] <= combined_data['technical_value'].mean()) & (combined_data['sentiment_score'] > 0.5)]

print("High Value Patents with Negative Reviews:")
print(high_value_negative_reviews)

print("\nLow Value Patents with Positive Reviews:")
print(low_value_positive_reviews)


# 수정

# In[ ]:


pip install sentence_transformers


# In[ ]:


from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
patent_embeddings = model.encode(patents['abstract'].tolist())
review_embeddings = model.encode(review['txt'].tolist())

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarities = cosine_similarity(patent_embeddings, review_embeddings)

most_similar_patents_idx = cosine_similarities.argmax(axis=0)

matched_patents = patents.iloc[most_similar_patents_idx]
matched_patents.reset_index(drop=True, inplace=True)

matched_reviews = review.copy()
matched_reviews['matched_patent_index'] = most_similar_patents_idx

combined_data = pd.DataFrame({
    'technical_value': matched_patents['technical_value'],
    'sentiment_score': matched_reviews['compound']
})


# In[ ]:


def detect_outliers_z_score(data):
    z_scores = np.abs((data - data.mean()) / data.std())
    return data[z_scores > 3]

def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]

outliers_z_score = detect_outliers_z_score(combined_data['technical_value'])
outliers_iqr = detect_outliers_iqr(combined_data['technical_value'])

# 기술적 가치가 높은 특허 중 소비자 반응이 부정적인 경우
high_value_negative_reviews = combined_data[(combined_data['technical_value'] > combined_data['technical_value'].mean()) & (combined_data['sentiment_score'] < -0.8) & (~combined_data.index.isin(outliers_z_score.index)) & (~combined_data.index.isin(outliers_iqr.index))]

# 기술적 가치가 낮은 특허 중 소비자 반응이 긍정적인 경우
low_value_positive_reviews = combined_data[(combined_data['technical_value'] <= combined_data['technical_value'].mean()) & (combined_data['sentiment_score'] > 0.8) & (~combined_data.index.isin(outliers_z_score.index)) & (~combined_data.index.isin(outliers_iqr.index))]

print("\nHigh Value Patents with Negative Reviews:")
print(high_value_negative_reviews)

print("\nLow Value Patents with Positive Reviews:")
print(low_value_positive_reviews)


# In[ ]:


high_value_negative_reviews


# In[ ]:


low_value_positive_reviews


# #### 2.8 시각화(기술의 혁신성, 소비자 만족도)

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

vectorizer = TfidfVectorizer(stop_words='english')
patent_abstracts = patents['abstract'].tolist()
review_texts = review['txt'].tolist()

all_texts = patent_abstracts + review_texts
tfidf_matrix = vectorizer.fit_transform(all_texts)

cosine_similarities = cosine_similarity(tfidf_matrix[:len(patent_abstracts)], tfidf_matrix[len(patent_abstracts):])

patent_ids = []
for review_idx in range(len(review_texts)):
    most_similar_patent_idx = cosine_similarities[:, review_idx].argmax()
    patent_ids.append(patents.iloc[most_similar_patent_idx]['publication number'])

merged_data = pd.DataFrame({
    'review_id': review.index,
    'review_text': review['txt'],
    'sentiment_score': review['compound'],
    'publication number': patent_ids
})

merged_data = merged_data.merge(patents, on='publication number')
merged_data['innovation_score'] = merged_data.apply(lambda row: row['cite'] + len(row['cpc']), axis=1)

plt.figure(figsize=(10, 6))
plt.scatter(merged_data['innovation_score'], merged_data['sentiment_score'], color='blue', alpha=0.6)
plt.title('Innovation vs. Consumer Satisfaction')
plt.xlabel('Innovation Score')
plt.ylabel('Consumer Satisfaction Score')
plt.grid(True)

for i, row in merged_data.iterrows():
    plt.text(row['innovation_score'], row['sentiment_score'], row['review_id'], fontsize=9)

plt.show()


# In[ ]:


merged_data[merged_data['review_id']==134]


# In[ ]:


pip install sentence_transformers


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
patent_embeddings = model.encode(patents['abstract'].tolist())
review_embeddings = model.encode(review['txt'].tolist())

cosine_similarities = cosine_similarity(patent_embeddings, review_embeddings)

patent_ids = []
for review_idx in range(len(review)):
    most_similar_patent_idx = cosine_similarities[:, review_idx].argmax()
    patent_ids.append(patents.iloc[most_similar_patent_idx]['publication number'])

merged_data = pd.DataFrame({
    'review_id': review.index,
    'review_text': review['txt'],
    'sentiment_score': review['compound'],
    'publication number': patent_ids
})

merged_data = merged_data.merge(patents, on='publication number')
merged_data['innovation_score'] = merged_data.apply(lambda row: row['cite'] + len(row['cpc']), axis=1)

scaler = MinMaxScaler()
merged_data[['innovation_score_scaled', 'sentiment_score_scaled']] = scaler.fit_transform(
    merged_data[['innovation_score', 'sentiment_score']]
)

plt.figure(figsize=(10, 6))
plt.scatter(merged_data['innovation_score_scaled'], merged_data['sentiment_score_scaled'], color='blue', alpha=0.6)
plt.title('Innovation vs. Consumer Satisfaction')
plt.xlabel('Innovation Score (Scaled)')
plt.ylabel('Consumer Satisfaction Score (Scaled)')
plt.grid(True)

for i, row in merged_data.iterrows():
    plt.text(row['innovation_score_scaled'], row['sentiment_score_scaled'], row['review_id'], fontsize=9)

plt.show()


# In[ ]:


merged_data[merged_data['review_id']==85]


# In[ ]:


merged_data[merged_data['review_id']==85]['title']


# In[ ]:


merged_data.info()


# In[ ]:


from keyphrasetransformer import KeyPhraseTransformer

kp = KeyPhraseTransformer()

kp.get_key_phrases(merged_data.iloc[152]['abstract'])


# In[ ]:


kp.get_key_phrases(merged_data.iloc[152]['review_text'])


# In[ ]:


pip install pytextrank


# In[ ]:


import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
import pytextrank

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

doc = nlp(merged_data.iloc[152]['abstract'])

for phrase in doc._.phrases:
    print(phrase.text)


# In[ ]:




