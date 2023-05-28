import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

# printing the stopwords in English
print(stopwords.words('english'))

# loading the dataset to a pandas DataFrame
true_dataset = pd.read_csv('/content/true.csv')  
fake_dataset = pd.read_csv('/content/fake.csv')

true_content = true_dataset['Title'] + ' ' + true_dataset['Content']
fake_content = fake_dataset['Title'] + ' ' + fake_dataset['Content']

# Creating the merged dataset
news_dataset = pd.DataFrame({
    'content': pd.concat([true_content, fake_content], ignore_index=True),
    'label': [1] * len(true_content) + [0] * len(fake_content)
})

news_dataset.shape

# print the first 5 rows of the dataframe
news_dataset.head()

# counting the number of missing values in the dataset
news_dataset.isnull().sum()

# replacing the null values with empty string
news_dataset = news_dataset.fillna('')

# separating the data & label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

print(X)
print(Y)

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'].head(500).apply(stemming)

print(news_dataset['content'])

#separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

print(X)

print(Y)

Y.shape

# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

model = LogisticRegression()

model.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

# User-defined test case
while True:
  test_headline = input("Enter the headline: ")

  # Preprocess the test headline
  test_input = stemming(test_headline)
  test_input_transformed = vectorizer.transform([test_input])

  # Make prediction
  prediction = model.predict(test_input_transformed)

  if prediction[0] == 0:
    print("The article is Fake")
  else:
    print("The article is Real")

  print("Do you want to test another headline? (Y/N)")