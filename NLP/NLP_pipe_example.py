import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
# train test split
from sklearn.model_selection import train_test_split

nltk.download('wordnet')

posts = fetch_20newsgroups(subset='all', categories=['sci.electronics', 'sci.space'],
                           remove=('headers', 'footers', 'quotes'))

df = pd.DataFrame({'text': posts.data, 'label': [posts.target_names[i] for i in posts.target]})

stop_words = set(stopwords.words('english'))


def clean_text(text: str, stop_words: set) -> str:
    # tokenize the text
    tokens = word_tokenize(text)
    # remove punctuations
    tokens = [word for word in tokens if word.isalpha()]
    # lowercase all words
    tokens = [word.lower() for word in tokens]
    # remove stopwords

    tokens = [word for word in tokens if not word in stop_words]
    # lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


df['cleaned_text'] = df['text'].apply(lambda x: clean_text(x, stop_words))

X = df['cleaned_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# vectorize the data
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(min_df=10, ngram_range=(2, 2))
X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.transform(X_test)

count_df = pd.DataFrame(X_train_counts.todense(), columns=count_vect.get_feature_names_out())
count_df.head()

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(max_df=0.7, min_df=0.01)

tfidf_train = tfidf_vect.fit_transform(X_train)
tfidf_test = tfidf_vect.transform(X_test)

# Display feature names
tfidf_df = pd.DataFrame(tfidf_train.toarray(), columns=tfidf_vect.get_feature_names_out())

# train the model classfier and predict for test data
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

nb = MultinomialNB()
nb.fit(tfidf_train, y_train)
y_pred = nb.predict(tfidf_test)
print(classification_report(y_test, y_pred))
print(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=posts.target_names, index=posts.target_names))
