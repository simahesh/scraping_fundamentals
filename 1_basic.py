import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np
import matplotlib.pyplot as plt

### 1. Web Scraping
url = 'https://www.amazon.in/OnePlus-Mirror-Black-128GB-Storage/product-reviews/B07DJHV6VZ/ref=dpx_acr_txt?showViewpoints=1'
review_html = requests.get(url).text # not understood by BS
soup = BeautifulSoup(review_html, 'lxml') # this is a structured BS object.

#soup.findAll -takes a dict of tags as input and gets the text in that tag
review_titles = soup.findAll('a', {'class': 'a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold'})
review_texts = soup.findAll('span', {'class': 'a-size-base review-text review-text-content'})
profile_names = soup.findAll('span', {'class': 'a-profile-name'})

#constructing DF for better understanding
df=pd.DataFrame({'cust':pd.Series([x.text for x in profile_names]),
	             'title':pd.Series([x.text for x in review_titles]),
	             'text':pd.Series([x.text for x in review_texts])})

### 2. Text Mining
soup = BeautifulSoup(requests.get("http://ptucha.com/").text)
l1 = soup.find_all(type='disc')
data = [listItem.li.text for listItem in l1]
pubRe = re.compile(r"R.W. Ptucha")
patentRe = re.compile(r"U.S. Patent")
#grab matching data in substitute unicode char with ascii
goodData = [re.sub("\x93|\x94", '""', d).strip() for d in data if pubRe.search(d) or patentRe.search(d)]

#plotting 
yearRe = re.compile("19\d{2} | 20\d{2}")
years = np.array([int(yearRe.search(d).group(0)) for d in data if yearRe.search(d) is not None])
numBins = np.max(years) - np.min(years) + 1
plt.figure(figsize=(12, 6))
plt.hist(years, numBins)
plt.title("# of Dr. Ptucha's Publications, Patents, and Presentations per Year")
plt.annotate("Earned M.S.", xy=(2002, 1), xytext=(2000, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

##############################################################
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']
vect = CountVectorizer()
vect.fit(simple_train)
vect.get_feature_names()
simple_train_dtm = vect.transform(simple_train) #transform training data into a 'document-term matrix' #sparse matrix
simple_train_dtm.toarray()                      #convert sparse matrix to a dense matrix
pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names()) #examine the vocabulary and document-term matrix together


sms = pd.read_csv('C:/Users/USER/Desktop/DSP Material/4. ML2 (MBA-CF-TextMining)/14. Text Mining/sms case study/sms.csv')
sms.label.value_counts()
sms['label'] = sms.label.map({'ham':0, 'spam':1})  #convert label to a numeric variable
X = sms.message
y = sms.label
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)  #split into training and testing sets
vect = CountVectorizer() #instantiate the vectorizer
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_train_dtm = vect.fit_transform(X_train) #alternative:combine fit and transform into a single step
X_test_dtm = vect.transform(X_test)
X_train_tokens = vect.get_feature_names()
X_train_counts = np.sum(X_train_dtm.toarray(), axis=0)
pd.DataFrame({'token':X_train_tokens, 'count':X_train_counts})

#Building a Naive Bayes model
# We will use [Multinomial Naive Bayes](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html):
# > The multinomial Naive Bayes classifier is suitable for classification with **discrete features** (e.g., word counts for text classification).
# The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)            #class predictions
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1] #predicted probabilities
print(metrics.accuracy_score(y_test, y_pred_class))
print(metrics.roc_auc_score(y_test, y_pred_prob))
print(metrics.confusion_matrix(y_test, y_pred_class))
print(X_test[y_test < y_pred_class]) #false positives
print(X_test[y_test > y_pred_class]) #false negatives
pritn(X_test[3132])   # what do you notice about the false negatives?

#Comparing Naive Bayes with logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train_dtm, y_train)
y_pred_class = logreg.predict(X_test_dtm)            #class predictions
y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1] #predicted probabilities
print(metrics.accuracy_score(y_test, y_pred_class))
print(metrics.roc_auc_score(y_test, y_pred_prob))

# ## Bonus: Calculating the "spamminess" of each token
sms_ham = sms[sms.label==0] # create separate DataFrames for ham and spam
sms_spam = sms[sms.label==1]
vect.fit(sms.message) # learn the vocabulary of ALL messages and save it
all_tokens = vect.get_feature_names()
ham_dtm = vect.transform(sms_ham.message)  #DTM (document-term matrices) for ham and spam
spam_dtm = vect.transform(sms_spam.message)
ham_counts = np.sum(ham_dtm.toarray(), axis=0) # count how many times EACH token appears across ALL ham/spam messages
spam_counts = np.sum(spam_dtm.toarray(), axis=0)
token_counts = pd.DataFrame({'token':all_tokens, 'ham':ham_counts, 'spam':spam_counts})
token_counts['ham'] = token_counts.ham + 1   # +1 to avoid dividing by zero
token_counts['spam'] = token_counts.spam + 1
token_counts['spam_ratio'] = token_counts.spam / token_counts.ham #ratio of spam-to-ham for each token
token_counts.sort_values('spam_ratio')
#############################################################
