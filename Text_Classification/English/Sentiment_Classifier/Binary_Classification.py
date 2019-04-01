# =============================================================================
# # 0. Libraries
# =============================================================================
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


import nltk
from nltk import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


from sklearn.feature_extraction import  DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.extmath import density


import sys
sys.path.append("C:/Users/me/Desktop/NLP_Learning")
sys.path.append('C:/Users/me/Anaconda3/envs/nlpLearning/lib/site-packages')
    
from textproc import summary_corpus, TextPreProcess

# =============================================================================
# # 1. Read the Data
# =============================================================================

cats= movie_reviews.categories()
cats

reviews= []
for cat in cats:
	for fid in movie_reviews.fileids(cat):
		review= (list(movie_reviews.words(fid)), cat)
		reviews.append(review)
	random.shuffle(reviews)



reviews_list, sentiments_list= [], []
for key, value in reviews:
    reviews_list.append(" ".join(key))
    sentiments_list.append(value)
    

df= pd.DataFrame.from_dict({'review': reviews_list, 'sentiment': sentiments_list})

# =============================================================================
# # 2. Explore the Data
# =============================================================================
print("Total Number of reviews    : ", len(reviews))
print("Number of Positive reviews : ", len([(key, val) for key , val in reviews if val =="pos"]))
print("Number of Negative reviews : ", len([(key, val) for key , val in reviews if val =="neg"]))

tokens, vocab, frequency_dist= summary_corpus(data= df, column='review', language="english")


# Distribution of the target variable
ax = sns.countplot(x= "sentiment", data= df) # data is balanced

# Plot the top 30 words
fig, ax = plt.subplots(figsize=(7,10))
sns.barplot(ax= ax, x= "frequency", y="word", data= frequency_dist[:30] , orient= "h")


mask = df["sentiment"] == "pos"
tokens_pos, vocab_pos, frequency_dist_pos= summary_corpus(data= df[mask], column='review', language="english")
tokens_neg, vocab_neg, frequency_dist_neg= summary_corpus(data= df[~mask], column='review', language="english")


fig, ax = plt.subplots(figsize=(7,10))
sns.barplot(ax= ax, x= "frequency", y="word", data= frequency_dist_pos[:30] , orient= "h")
fig, ax = plt.subplots(figsize=(7,10))
sns.barplot(ax= ax, x= "frequency", y="word", data= frequency_dist_neg[:30] , orient= "h")


# =============================================================================
# # 3. Preprocess the text
# =============================================================================
preprocess = TextPreProcess()

stopwords = set(stopwords.words('english'))

df["review_processed"] = df["review"].apply(preprocess.remove_punct_digits)
df["review_processed"] = df["review_processed"].apply(preprocess.remove_double_spaces)
df["review_processed"] = df["review_processed"].apply(preprocess.expand_contractions)
df["review_processed"] = df["review_processed"].apply(lambda x: preprocess.remove_stops(x , stopwords))
df["review_processed"] = df["review_processed"].apply(preprocess.remove_outside_spaces)


# stemming and lemmatization
# SpellCorrection
# Language Detection


mask = df["sentiment"] == "pos"
tokens_pos, vocab_pos, frequency_dist_pos= summary_corpus(data= df[mask], column='review_processed', language="english")
tokens_neg, vocab_neg, frequency_dist_neg= summary_corpus(data= df[~mask], column='review_processed', language="english")

fig, ax = plt.subplots(figsize=(7,10))
sns.barplot(ax= ax, x= "frequency", y="word", data= frequency_dist_pos[:30] , orient= "h")
fig, ax = plt.subplots(figsize=(7,10))
sns.barplot(ax= ax, x= "frequency", y="word", data= frequency_dist_neg[:30] , orient= "h")     


# =============================================================================
# # 4. Feature Extraction
# =============================================================================

from scipy import sparse
class Vectorizers():
    def __init__(self, target, verbatim , covariates):
        self.target= target
        self.verbatim= verbatim
        self.covariates= covariates
    
    def train_test_split(self, data, stratify=None):
        self.train , self.test = train_test_split(data,
                                        test_size= .25,
                                        stratify= stratify,
                                        random_state= 1953)
        
        
    def feature_extraction(self, vectorizer):
        
        self.X_verbatim_train= vectorizer.fit_transform(self.train[self.verbatim])
        self.X_verbatim_test = vectorizer.transform(self.test[self.verbatim])
        self.vect_feature_names = vectorizer.get_feature_names()
        
        self.X_covariates_train = sparse.csr_matrix(self.train[self.covariates])
        self.X_covariates_test  = sparse.csr_matrix(self.test[self.covariates])


        self.X_train = sparse.hstack([self.X_covariates_train, self.X_verbatim_train])
        self.X_test  = sparse.hstack([self.X_covariates_test, self.X_verbatim_test])
    
    def processed_data(self, vectorizer ):
        self.feature_extraction(vectorizer= vectorizer)
        return self.X_train , self.train[self.target] , self.X_test, self.test[self.target]


verbatim = ["review", "review_processed"]

vect = Vectorizers(target= "sentiment", verbatim= "review_processed", covariates= [])
vect.train_test_split(data= df, stratify = df["sentiment"])


countVect= CountVectorizer(tokenizer= word_tokenize,
                           stop_words= None,
                           ngram_range= (1,1),
                           analyzer= "word",
                           max_df= 0.9,
                           min_df= 0.02,
                           max_features= 2000,
                           strip_accents=None)

countVectBinary= CountVectorizer(tokenizer= word_tokenize,
                                 stop_words= None,
                                 ngram_range= (1,1),
                                 analyzer= "word",
                                 max_df= 0.9,
                                 min_df= 0.02,
                                 max_features= 2000,
                                 binary= True,
                                 strip_accents=None)


tfidfVect = TfidfVectorizer(tokenizer= word_tokenize,
                            stop_words= None,
                            ngram_range= (1,3),
                            analyzer= "word",
                            max_df= 0.9,
                            min_df= 0.02,
                            max_features= 7000,
                            norm="l2", 
                            use_idf=True, 
                            smooth_idf=True,
                            sublinear_tf=True,
                            strip_accents=None)

#X_train_countVect , y_train, X_test_countVect, y_test = vect.processed_data(vectorizer= countVect)
#feature_names_CountVect = vect.vect_feature_names

vetorizer = tfidfVect
print("Extracting features from the training data using a sparse vectorizer : ")
print(vetorizer)
t0 = time.time()
X_train , y_train, X_test, y_test = vect.processed_data(vectorizer= vetorizer)
feature_names = vect.vect_feature_names
feature_names = np.asarray(feature_names)
duration = time.time() - t0
print("done in %fs " % (duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()




# =============================================================================
# # 5. Models
# =============================================================================
chi2_selection= False
print_top10= True
print_report = True
print_cm = True


if chi2_selection:
    k = 100
    print("Extracting %d best features by a chi-squared test" % k)
    t0 = time.time()
    chi_square = SelectKBest(chi2, k= k)
    X_train = chi_square.fit_transform(X_train, y_train)
    X_test  = chi_square.transform(X_test)
    
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i in chi_square.get_support(indices=True)]
        feature_names = np.asarray(feature_names)
    
    print("done in %fs" % (time.time() - t0))
    print()


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time.time()
    pred = clf.predict(X_test)
    test_time = time.time() - t0
    print("test time:  %0.3fs" % test_time)

    score = accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if print_top10 and feature_names is not None:
            print("top 10 keywords :")
            top10 = np.argsort(clf.coef_)[-10:]
            print(trim("{}".format(" ".join(feature_names[top10][0]))))
#            for i, label in enumerate(target_names):
#                top10 = np.argsort(clf.coef_[i])[-10:]
#                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    if print_report:
        print("classification report:")
        print(classification_report(y_test, pred))

    if print_cm:
        print("confusion matrix:")
        print(confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50, tol=1e-3), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50, tol=1e-3),"Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100, 
                                n_jobs=-1,
                                max_depth= None,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                max_features= "auto",
                                criterion= "gini",
                                oob_score=True,
                                random_state=1953,
                                verbose=1) , "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))
results.append(benchmark(ComplementNB(alpha=.1)))

#print('=' * 80)
#print("LinearSVC with L1-based feature selection")
## The smaller C, the stronger the regularization.
## The more regularization, the more sparsity.
#results.append(benchmark(Pipeline([
#  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
#                                                  tol=1e-3))),
#  ('classification', LinearSVC(penalty="l2"))])))


# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()


# =============================================================================
# # 6. Fine Tune Hyperparameters
# =============================================================================



# =============================================================================
# # 7. Code for Deployment
# =============================================================================

# Create Matrix with counts
# Create Matrix with binaries (presence or abscence)
# Create Matrix with TFIDF

