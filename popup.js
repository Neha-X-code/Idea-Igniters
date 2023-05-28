/*function detectArticles() {
    alert('Detecting misleading medicine articles...');
  
    // Your machine learning code
    // Replace this code with your actual implementation
  
    // Example code: Simulating a delay of 3 seconds
    setTimeout(function() {
      alert('Detection complete!');
    }, 3000);
  }
  */
 // Import statements
const { stopwords } = require('nltk.corpus');
const { PorterStemmer } = require('nltk-stemming');
const { TfidfVectorizer } = require('sklearn');
const { train_test_split } = require('sklearn.model_selection');
const { LogisticRegression } = require('sklearn.linear_model');
const { accuracy_score } = require('sklearn.metrics');
const pandas as pd = require('pandas');
const numpy as np = require('numpy');
const re = require('re');

// Download stopwords
nltk.download('stopwords');

// Function to preprocess the headline and make prediction
function detectFakeNews(headline) {
  // Loading the dataset to a pandas DataFrame
  const true_dataset = pd.read_csv('/content/true.csv');
  const fake_dataset = pd.read_csv('/content/fake.csv');

  const true_content = true_dataset['Title'] + ' ' + true_dataset['Content'];
  const fake_content = fake_dataset['Title'] + ' ' + fake_dataset['Content'];

  // Creating the merged dataset
  const news_dataset = pd.DataFrame({
    'content': pd.concat([true_content, fake_content], { 'ignore_index': true }),
    'label': Array(true_content.length).fill(1).concat(Array(fake_content.length).fill(0))
  });

  // Replace the null values with empty string
  news_dataset.fillna('', { inplace: true });

  // Separating the data & label
  const X = news_dataset.drop('label', { axis: 1 });
  const Y = news_dataset['label'];

  // Porter Stemmer
  const port_stem = new PorterStemmer();

  function stemming(content) {
    let stemmed_content = content.replace(/[^a-zA-Z]/g, ' ');
    stemmed_content = stemmed_content.toLowerCase().split(' ');
    stemmed_content = stemmed_content.filter((word) => !stopwords.words('english').includes(word));
    stemmed_content = stemmed_content.map((word) => port_stem.stem(word));
    stemmed_content = stemmed_content.join(' ');
    return stemmed_content;
  }

  news_dataset['content'].head(500).apply(stemming);

  // Separating the data and label
  const X = news_dataset['content'].values;
  const Y = news_dataset['label'].values;

  // Converting the textual data to numerical data
  const vectorizer = new TfidfVectorizer();
  vectorizer.fit(X);

  const X_transformed = vectorizer.transform(X);

  const [X_train, X_test, Y_train, Y_test] = train_test_split(X_transformed, Y, { 'test_size': 0.2, 'stratify': Y, 'random_state': 2 });

  const model = new LogisticRegression();
  model.fit(X_train, Y_train);

  // Preprocess the test headline
  const test_input = stemming(headline);
  const test_input_transformed = vectorizer.transform([test_input]);

  // Make prediction
  const prediction = model.predict(test_input_transformed);

  if (prediction[0] == 0) {
    return "The article is Fake";
  } else {
    return "The article is Real";
  }
}

// Handle button click event
document.getElementById('detect-button').addEventListener('click', () => {
  const headlineInput = document.getElementById('headline-input');
  const headline = headlineInput.value;

  if (headline) {
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = detectFakeNews(headline);
  }
});
