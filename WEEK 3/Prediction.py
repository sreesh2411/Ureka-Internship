#Import the required libraries

import numpy as np 
import pandas as pd 
import re  

import nltk 
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

validation = pd.read_csv("Datasets/validation.csv")

#DATA PREPROCESSING

validation.dropna(subset=['text']) #Dropping null values (if any are present)

#Removing @users since they do not contribute to sentiment analysis
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt    

for x in range(validation.shape[0]):
    validation['text'][x] = np.vectorize(remove_pattern)(str(validation['text'][x]), "@[\w]*")


X = validation.iloc[:, 0].values  

processed_validations = []

for text in range(0, len(X)):  
    # Remove all the special characters
    processed_validation = re.sub(r'\W', ' ', str(X[text]))
 
    # remove all single characters
    processed_validation = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_validation)
 
    # Remove single characters from the start
    processed_validation = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_validation) 
 
    # Substituting multiple spaces with single space
    processed_validation = re.sub(r'\s+', ' ', processed_validation, flags=re.I)
 
    #Removing numbers from the validation
    processed_validation = re.sub(r'\d', ' ', processed_validation)
 
    # Converting to Lowercase
    processed_validation = processed_validation.lower()
 
    processed_validations.append(processed_validation)   

processed_tweets = np.array(processed_validations)

processed_tweets = pd.DataFrame(data=processed_tweets, columns=["tidy_text"]) #Creating a new dataframe which consists of tidy tweets.

#Removing words which have length lesser than 3 since words like 'oh', 'at' etc. do not contribute to the analysis.
processed_tweets['tidy_text'] = processed_tweets['tidy_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>=3]))


#Tokenizing the clean tweets. Tokens are individual terms or words, and tokenization is the process of splitting a string of text into tokens.
tokenized_tweet = processed_tweets['tidy_text'].apply(lambda x: x.split())


#Stemming is a process of obtaining root words and removing unneessary prefixes like 'ing' etc.
from nltk.stem import WordNetLemmatizer, PorterStemmer
stemmer = PorterStemmer()
wnl = WordNetLemmatizer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [(wnl.lemmatize(i) if wnl.lemmatize(i).endswith('e') else stemmer.stem(i)) for i in x]) # stemming


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    processed_tweets['tidy_text'] = tokenized_tweet


validation['tidy_text']= processed_tweets['tidy_text']

validation.head()

import pickle 
  
# Load the Model back from file
filename=open("Sreesh_SVC.pkl","rb")
svc_pickle_model = pickle.load(filename)


from sklearn.pipeline import Pipeline
vectorizer = CountVectorizer(min_df=0.01,max_df = 2.5, lowercase = False, stop_words = 'english')

X = vectorizer.fit_transform(validation['tidy_text'])
transformer = TfidfTransformer()
X = transformer.fit_transform(X)

X

validation['sentiment']=svc_pickle_model.predict(X)

di = {'2': "positive", '1': "neutral", '0' : "negative"}

validation['sentiment'] = validation['sentiment'].map(di)

