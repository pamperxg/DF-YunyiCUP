import pandas as pd
import jieba
import re
from gensim.models import Word2Vec
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import scipy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
from scipy.sparse import csr_matrix,hstack
import gc

def score(predict,label):
    error = []
    for i in range(len(label)):
        error.append(label[i]-predict[i])
    squared_error = []
    for val in error:
        squared_error.append(val*val)
    RMSE = np.sqrt(np.sum(squared_error)/len(squared_error))
    score = 1/(1+RMSE)       
    return score

with open('../input/positiveWords_.txt',encoding='UTF-8') as words:
    positiveWords = [i.strip() for i in words.readlines()]
with open('../input/negativeWords_.txt',encoding='UTF-8') as words:
    negativeWords = [i.strip() for i in words.readlines()]
stop_words = ['的','是']
positiveWords = set(positiveWords)
negativeWords = set(negativeWords)

train_df = pd.read_csv('../input/train_first.csv')
test_df = pd.read_csv('../input/predict_first.csv')
all_df = pd.concat([train_df,test_df],axis=0)

all_words = []
for discuss in all_df['Discuss']:
    words = []
    discuss = re.findall('[a-zA-Z0-9\u4e00-\u9fa5]+',discuss)  #只留下数字字母和文字a-zA-Z0-9
    discuss = ''.join(discuss)
    for word in jieba.cut(discuss):              #分词
        word = word.strip()
        if word not in stop_words:                   #去停词
            words.append(word)
    all_words.append(words)
all_df['Dis_cut'] = all_words

model = Word2Vec(all_words,min_count=5,size=256,workers=4)
# model.save(fname)
# model = Word2Vec.load(fname)
l_dis_cut = list(all_df['Dis_cut'])[0]
for word in l_dis_cut:
    if word in model:
        r = model[word].shape
        break
        
def wordcut2sumW2V(word_cut):        #词向量值累加
    sumW2V = np.zeros(r)
    for word in word_cut:
        if word in model:
            sumW2V += model[word]
    return sumW2V

arr_sumW2V = all_df['Dis_cut'].map(wordcut2sumW2V)
arr_sumW2V = np.array(arr_sumW2V)
arr_sumW2V = np.array([arr for arr in arr_sumW2V])#维度
for i in range(arr_sumW2V.shape[1]):
    all_df['sumW2V%d' % i] = arr_sumW2V[:,i]
del arr_sumW2V,l_dis_cut,model
gc.collect()

l_wordcut = list(all_df['Dis_cut'].map(lambda s:' '.join(str(i) for i in s)))
cv = CountVectorizer(ngram_range=(1,2))           #词频
discuss = cv.fit_transform(l_wordcut)
tf = TfidfVectorizer(max_df=1000000,ngram_range=(1,2))  #tfidf
discuss_df = tf.fit_transform(l_wordcut)
data = hstack((discuss,discuss_df)).tocsr()

def get_positive(dis_cut):
    count = 0
    for word in dis_cut:
        if word in positiveWords:
            count += 1
    return count
def get_negative(dis_cut):
    count = 0
    for word in dis_cut:
        if word in negativeWords:  
            count += 1
    return count
all_df['dis_len'] = all_df['Dis_cut'].apply(lambda s:len(s))
all_df['pos'] = all_df['Dis_cut'].apply(get_positive)    #正负向词个数
all_df['neg'] = all_df['Dis_cut'].apply(get_negative)
all_df['sub_pos_neg'] = all_df['pos'] - all_df['neg']
all_df['punc_num'] = all_df['Discuss'].apply(lambda s:len(re.findall('[^a-zA-Z0-9\u4e00-\u9fa5]+',s)))
all_df['comma'] = all_df['Discuss'].apply(lambda s:len(re.findall('[,，]+',s)))
all_df['fullstop'] = all_df['Discuss'].apply(lambda s:len(re.findall('[。]+',s)))
all_df['authpunc'] = all_df['Discuss'].apply(lambda s:len(re.findall('[~]+',s)))

features = list(all_df.columns)[4:]
data_ = scipy.sparse.csr_matrix(all_df[features].values)
data_a = hstack((data_,data)).tocsr()

X = data_a[:100000]       #训练集
y = train_df[['Score']]
test = data_a[100000:]    #测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=66)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=110,
                              learning_rate=0.05, n_estimators=1000,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(X_train,list(y_train.Score))
predict = model_lgb.predict(X_test)
print(predict)
label = y_test.Score.values.astype(np.float32)
print(score(predict,label)) #线下score
#生成线上提交结果
model_lgb.fit(X,list(y.values))
predict = model_lgb.predict(test)
sub = test_df[['Id']]
sub['1'] = predict
sub.to_csv('../sub/result.csv',header=None,index=None)




