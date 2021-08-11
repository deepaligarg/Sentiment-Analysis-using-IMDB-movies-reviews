from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib
import pandas as pd
import numpy as np
import nltk
from matplotlib import pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import joblib


app = Flask(__name__)

@app.route('/') #if 127.0.0.1:5000/ --> it will open home.html as home function follows just after this route decorator.
def home():
        return render_template('home.html')




#PUNCT_TO_REMOVE = string.punctuation

mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have",
           "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not",
           "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is",
           "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", 
           "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have",
           "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am",
           "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", 
           "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
           "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
           "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
           "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", 
           "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", 
           "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is",
           "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",
           "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", 
           "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", 
           "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", 
           "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
           "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
           "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
           "won't": "will not", "won't've": "will not have", "would've": "would have","wouldn't": "would not",
           "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have",
           "y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", 
           "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))

def clean_contractions(text, mapping):
        specials = ["’", "‘", "´", "`"]
        for s in specials:
                text = text.replace(s, "'")
        text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
        return text

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def word_replace(text):
        return text.replace('<br />','')


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def stem_words(text):
        return " ".join([stemmer.stem(word) for word in text.split()])



from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])



def remove_urls(text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)


def remove_html(text):
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', text)

def preprocess(text):
        text=clean_contractions(text,mapping)
        text=text.lower()
        text=word_replace(text)
        text=remove_urls(text)
        text=remove_html(text)
        text=remove_stopwords(text)
        text=remove_punctuation(text)
        text=lemmatize_words(text)
        
        return text

@app.route('/predict',methods=['POST'])
def predict():
	#df= pd.read_csv("spam.csv", encoding="latin-1")
	#df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	# Features and Labels
	#df['label'] = df['class'].map({'ham': 0, 'spam': 1})
	#X = df['message']
	#y = df['label']


	movies = pd.read_csv('IMDB Dataset.csv')
	movies.head()
	movies.sentiment=movies.sentiment.apply(lambda x : 0 if x=='negative' else 1)
	# Mapping for all the abbreviated words and slangs commonly used in chats/reviews/comments of people
	

	movies["reviews_processed"] = movies["review"].apply(lambda text: preprocess(text))

	x=movies['reviews_processed']
	y=movies['sentiment']
	x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)
	tfidf= TfidfVectorizer(max_features=10000,ngram_range=(1,3))
	tfidf.fit(x)
	tfidf_train=tfidf.transform(x_train)
	tfidf_test= tfidf.transform(x_test)

	lr1= LogisticRegression(penalty='l2',solver='saga')
	lr1.fit(tfidf_train,y_train)
	



	

	#clf = MultinomialNB()
	#clf.fit(X_train,y_train)
	#clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

	#lr1 = joblib.load(LR_sentiment_model.pkl)

	if request.method == 'POST':
                message = request.form['message']
                data = [message]
                #vect = cv.transform(data).toarray()
                vect= tfidf.transform(data).toarray()
                #my_prediction = clf.predict(vect)
                y_pred_lr1_tfidf= lr1.predict(vect)
		
	return render_template('result.html',prediction = y_pred_lr1_tfidf)



if __name__ == '__main__':
        app.run(debug=True) #server will reload itself if any code changes and also track useful errors if got while running app.
