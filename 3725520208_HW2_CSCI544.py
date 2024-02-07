#!/usr/bin/env python
# coding: utf-8

# Name: Anne Sai Venkata Naga Saketh <br>
# USC ID: 3725520208 <br>
# USC Email: annes@usc.edu <br>
# <b> NLP HW2 </b>

# ## Import Statements

# In[21]:


# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical operations
import re            # Regular expressions for text processing
from bs4 import BeautifulSoup  # For HTML parsing
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import nltk          # Natural Language Toolkit for text processing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')  # Download WordNet data
nltk.download('stopwords')   # Download StopWords data

import warnings      # To handle warnings
warnings.filterwarnings("ignore")  # Ignore warnings for the remainder of the code
warnings.filterwarnings("default")  # Set warnings back to default behavior

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec

from nltk.tokenize import word_tokenize
from tqdm import tqdm
# from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC

import torch.nn as nn
import torch.optim as optim


# ## 1. Data Set Generation

# ## Read Data from the file

# Ignoring the bad lines or the rows in the TSV file that contain in-correct data

# In[22]:


# Reading the data from the tsv (Amazon Kitchen dataset) file as a Pandas frame
full_data = pd.read_csv("./amazon_reviews_us_Office_Products_v1_00.tsv", delimiter='\t', encoding='utf-8', error_bad_lines=False)


# ## Extract only Review and Ratings

# In[23]:


# Printing the data frame that contains the entire dataset from the tsv file
print(full_data)

# Keep only the Reviews and Ratings fields from the full data
df = full_data[['review_body', 'star_rating']]

# Converting 'star_rating' to numeric values
df['star_rating'] = pd.to_numeric(full_data['star_rating'], errors='coerce')


# ## Prepare a Balanced Data Set and prepare a Testing and Training Set

# From the dataset we have extracted in the previous step, creating a balanced data set for each of the rating that we have from 1 to 5. Then I had added the sentiment column based on the ratings that we have and then I have split it into training and testing datasets.

# In[24]:


# Check unique values in 'star_rating' column
unique_ratings = df['star_rating'].unique()
# print("Unique ratings:", unique_ratings)

# Convert 'star_rating' column to integer, handling errors
df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce')

# Drop rows with NaN values in 'star_rating' column
df = df.dropna(subset=['star_rating'])

# Convert 'star_rating' column to integer
df['star_rating'] = df['star_rating'].astype(int)

# Define a custom dataset class
class AmazonReviewsDataset(Dataset):
    def __init__(self, reviews, ratings):
        self.reviews = reviews
        self.ratings = ratings

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        rating = self.ratings[idx]
        return review, rating

# Build balanced dataset
ratings = df['star_rating'].unique()
balanced_data = pd.DataFrame(columns=df.columns)
for rating in ratings:
    subset = df[df['star_rating'] == rating]
    if len(subset) >= 50000:
        subset = subset.sample(n=50000, random_state=42)
    balanced_data = pd.concat([balanced_data, subset])

# Create ternary labels
balanced_data['sentiment'] = np.where(balanced_data['star_rating'] > 3, 1, 
                                      np.where(balanced_data['star_rating'] < 3, 2, 3))

print("Checking if the Dataset has been balanced out:\n")
print("Star_rating  Count")
print(balanced_data['star_rating'].value_counts())

print("Checking if the Sentiments has been balanced out:\n")
print("Sentiment  Count")
print(balanced_data['sentiment'].value_counts())

# Perform train-test split
train_data, test_data = train_test_split(balanced_data, test_size=0.2, random_state=42)

# Define train and test datasets
train_dataset = AmazonReviewsDataset(train_data['review_body'].values, train_data['sentiment'].values)
test_dataset = AmazonReviewsDataset(test_data['review_body'].values, test_data['sentiment'].values)

# Define DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print sizes of train and test datasets
print("\nPrinting the Test and the Training set data sizes")
print("Train dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))


# ## Split the Training and Testing Data Set

# In[25]:


# Splitting the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(balanced_data['review_body'],
                                                    balanced_data['sentiment'],
                                                    test_size=0.2,
                                                    random_state=42)


# ## Clean the Data

# Cleaning the data we have split.
# 
# 1. Removing Contractions
# 2. Removing the unnecessary characters
# 3. Remving any HTML links
# 4. Converting to lower cases

# In[26]:


# Define a contraction map
CONTRACTION_MAP = {
    "won't": "will not",
    "can't": "cannot",
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "that's": "that is",
    "we're": "we are",
    "they're": "they are",
    "isn't": "is not",
    "aren't": "are not",
    "haven't": "have not",
    "hasn't": "has not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "won't've": "will not have",
    "can't've": "cannot have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "that'll": "that will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "it'd": "it would",
    "that'd": "that would",
    "we'd": "we would",
    "they'd": "they would",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "shouldn't": "should not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "oughtn't": "ought not",
    "who's": "who is",
    "what's": "what is",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",
    "it's": "it is",
    "let's": "let us"
}

# Function to expand contractions
def expand_contractions(text):
    for contraction, expansion in CONTRACTION_MAP.items():
        text = re.sub(contraction, expansion, text)
    return text

# Preprocess the reviews
def preprocess_reviews(reviews):
    # Convert to lowercase and handle NaN values
    reviews = reviews.apply(lambda x: str(x).lower() if pd.notna(x) else '')
    
    # Remove HTML and URLs
    reviews = reviews.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
    reviews = reviews.apply(lambda x: re.sub(r'http\S+', '', x))

    # Remove non-alphabetical characters (excluding single quote)
    reviews = reviews.apply(lambda x: re.sub(r'[^a-zA-Z\s\']', '', x))

    # Remove extra spaces
    reviews = reviews.apply(lambda x: re.sub(' +', ' ', x))

    # Perform contractions
    reviews = reviews.apply(expand_contractions)

    # Return the processed text of the review
    return reviews

# Preprocess the training set
X_train_preprocessed = preprocess_reviews(X_train)

# Print average length of reviews before and after cleaning
avg_length_before = X_train.apply(lambda x: len(str(x))).mean()
avg_length_after = X_train_preprocessed.apply(len).mean()
print("===================Printing the Average lenght of Reviews Before and After Cleaning====================")
print(f"\nAverage Length of Reviews (Before Cleaning): {int(avg_length_before)} characters")
print(f"Average Length of Reviews (After Cleaning): {int(avg_length_after)} characters")


# ## Pre-Process the Data

# Using NLTK to remove the stop words that are unnecessary for sentiment classification.

# In[27]:


# Initialize NLTK's stopwords and WordNet lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to remove stop words and perform lemmatization
def preprocess_nltk(review):
    if pd.notna(review):
        words = nltk.word_tokenize(str(review).lower())  # Convert to lowercase
        words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
        return ' '.join(words)
    else:
        return ''

# Preprocess the training set using NLTK
X_train_nltk_preprocessed = X_train_preprocessed.apply(preprocess_nltk)

# Print three sample reviews before and after NLTK preprocessing
sample_reviews_indices = X_train_preprocessed.sample(3).index

print("============ Printing Sample Reviews Before and After Pre-processing =============")
for index in sample_reviews_indices:
    print(f"\nSample Review {index} Before Pre-processing:")
    print(X_train_preprocessed.loc[index])

    print(f"\nSample Review {index} After NLTK Pre-processing:")
    print(X_train_nltk_preprocessed.loc[index])

# Print average length of reviews before and after NLTK processing
avg_length_before_nltk = X_train_preprocessed.apply(len).mean()
avg_length_after_nltk = X_train_nltk_preprocessed.apply(len).mean()
print("\n=================Printing the Average lenght of Reviews Before and After Pre-processing==================")
print(f"\nAverage Length of Reviews (Before NLTK Processing): {int(avg_length_before_nltk)} characters")
print(f"Average Length of Reviews (After NLTK Processing): {int(avg_length_after_nltk)} characters")


#  ##  2. Word Embedding

# In[28]:


get_ipython().system('pip install gensim')


# ## Similarity Score using the Word2Vec Pretrained model

# Since we have to use binary data for Simple Models(Perceptron and SVM), so we need to remove the neutral reviews with class 3 and keep only Class 1 and Class 2.
# 
# For Simple models classification, we need to change the class labels from 1 and 2 to 0 and 1, as the class labels should always start from '0'

# In[29]:


# Joining features and targets for training dataset
train_data = pd.concat([X_train_nltk_preprocessed, y_train], axis=1)
train_data_filtered = train_data[train_data['sentiment'] != 3]
X_train_binary = train_data_filtered['review_body']
y_train_binary = train_data_filtered['sentiment']

# Joining features and targets for testing dataset
test_data = pd.concat([X_test, y_test], axis=1)
test_data_filtered = test_data[test_data['sentiment'] != 3]
X_test_binary = test_data_filtered['review_body']
y_test_binary = test_data_filtered['sentiment']


# Download the Pre-trained Model from Google.

# In[59]:


import gensim.downloader as api
wv = api.load('word2vec-google-news-300')


# In[60]:


# Function to extract Word2Vec features for a given sentence
def extract_word2vec_features(sentence, model, vector_size):
    word_vectors = []
    for word in sentence:
        if word in model.key_to_index:
            word_vectors.append(model.get_vector(word))
    if len(word_vectors) == 0:
        return np.zeros(vector_size)  # Return zero vector if no word vectors found
    else:
        return np.mean(word_vectors, axis=0)  # Return average word vector

# Examples to check semantic similarities
example1 = wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print("King - Man + Woman =", example1)

example2 = wv.most_similar(positive=['excellent', 'outstanding'], topn=1)
print("Excellent ~ Outstanding =", example2)

example3 = wv.most_similar(positive=['marriage', 'woman'], negative=['man'], topn=1)
print("Marriage - Man + Woman =", example3)

example4 = wv.most_similar(positive=['good'], negative=['bad'], topn=1)
print("Good ~ Bad =", example4)


# ## Similarity Score using the Word2Vec model with the custom model

# In[32]:


X_train_nltk_preprocessed_new = []

for e, k in tqdm(enumerate(X_train_nltk_preprocessed.to_list())):
    try:
        X_train_nltk_preprocessed_new.append(word_tokenize(k))
    except:
        pass    
def get_doc_embedding(doc):
    words = doc.lower().split()
    return wv.get_mean_vector(words)


# Customizing the Pre-trained model and making it custom trained with the data we have from the Reviews

# In[61]:


from gensim.models import Word2Vec
# Train the Word2Vec model
model_own = Word2Vec(X_train_nltk_preprocessed_new, vector_size=300, window=11, min_count=10)


# In[62]:


# Function to extract Word2Vec features for a given sentence using the trained model
def extract_word2vec_features_own(sentence, model, vector_size):
    word_vectors = []
    for word in sentence:
        if word in model.wv.key_to_index:
            word_vectors.append(model.wv.get_vector(word))
    if len(word_vectors) == 0:
        return np.zeros(vector_size)  # Return zero vector if no word vectors found
    else:
        return np.mean(word_vectors, axis=0)  # Return average word vector

# Examples to check semantic similarities using your own model
if 'king' in model_own.wv.key_to_index and 'woman' in model_own.wv.key_to_index:
    example1_own = model_own.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
    print("King - Man + Woman (Own Model) =", example1_own)
else:
    print("'good' is not present in the vocabulary.")

if 'excellent' in model_own.wv.key_to_index and 'outstanding' in model_own.wv.key_to_index:
    example2_own = model_own.wv.most_similar(positive=['excellent', 'outstanding'], topn=1)
    print("Excellent ~ Outstanding (Own Model) =", example2_own)
else:
    print("'excellent' and/or 'outstanding' are not present in the vocabulary.")
    
if 'marriage' in model_own.wv.key_to_index and 'woman' in model_own.wv.key_to_index:
    example3_own = model_own.wv.most_similar(positive=['marriage', 'woman'], negative=['man'], topn=1)
    print("Marriage - Man + Woman (Own Model) =", example3_own)
else:
    print("'marriage' and/or 'woman' are not present in the vocabulary.")
    
if 'good' in model_own.wv.key_to_index:
    example4_own = model_own.wv.most_similar(positive=['good'], negative=['bad'], topn=1)
    print("Good ~ Bad (Own Model) =", example4_own)
else:
    print("'marriage' and/or 'woman' are not present in the vocabulary.")

# Compare semantic similarities between pretrained and own models
print("\nSemantic similarity (Pretrained Model):", example1)
print("Semantic similarity (Pretrained Model):", example2)
print("Semantic similarity (Pretrained Model):", example3)
print("Semantic similarity (Pretrained Model):", example4)


# ## What do you conclude from comparing vectors generated by yourself and the pretrained model? Which of the Word2Vec models seems to encode semantic similarities between words better?

# Comparing vectors generated by pre-trained Word2Vec models and those trained on specific datasets reveals nuanced trade-offs. Pre-trained models, leveraging vast corpora, excel in capturing broad semantic similarities but may lack fine-grained domain specificity. Conversely, self-generated vectors, tailored to specific contexts, offer potential for domain-specific insights but require representative data and entail computational costs. Evaluating both models on relevant tasks like word similarity or downstream applications is crucial to determining which better encodes semantic relationships for specific use cases.

# ## 3. Simple Models

# ## Extract the TF-IDF Features from Dataset and train Perceptron and SVM

# Extracting the TF-IDF feature vectors from the initial dataset, and using that data set to train the Perceptron and SVM, and calculate the accuracy on the testing dataset

# In[35]:


# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=2000000)

# Fit and transform the training set
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_binary)

# Transform the test set
X_test_tfidf = tfidf_vectorizer.transform(X_test_binary.apply(preprocess_nltk))

# Print the shape of the TF-IDF matrices
print(f"\nShape of X_train_tfidf: {X_train_tfidf.shape}")
print(f"Shape of X_test_tfidf: {X_test_tfidf.shape}")


# In[36]:


perceptron = Perceptron(max_iter=1000)

perceptron.fit(X_train_tfidf, y_train_binary)
y_pred = perceptron.predict(X_test_tfidf)
accuracy_perceptron_tf_idf = accuracy_score(y_test_binary, y_pred)
print(f"Accuracy of the Single Layer Perceptron using TF-IDF Features: {accuracy_perceptron_tf_idf}")


# In[37]:


svm = LinearSVC()

svm.fit(X_train_tfidf, y_train_binary)
y_pred = svm.predict(X_test_tfidf)
accuracy_svm_tf_idf = accuracy_score(y_test_binary, y_pred)
print(f"Accuracy of the SVM using TF-IDF Features: {accuracy_svm_tf_idf}")


# ## Extract the Word2Vec Pre-trained Features from Dataset and train Perceptron and SVM

# Extracting the Word2Vec feature vectors on the Pre-trained model from the initial dataset, and using that data set to train the Perceptron and SVM, and calculate the accuracy on the testing dataset

# In[38]:


def get_doc_embedding(doc):
    doc_str = ' '.join(doc)  # Join tokens within the document list to form a single string
    words = doc_str.lower().split()  # Split the string into words
    if not words:
        return np.zeros(300)  # Return a zero vector if no words are present
    else:
        return wv.get_mean_vector(words)  # Compute the mean vector using Word2Vec model

X_train_w2v_binary = []
X_test_w2v_binary = []

# Extract Word2Vec features for training data
for e in tqdm(X_train_binary):
    X_train_w2v_binary.append(get_doc_embedding(e))

# Extract Word2Vec features for testing data
for e in tqdm(X_test_binary):
    X_test_w2v_binary.append(get_doc_embedding(e))
    
X_train_w2v_binary = np.array(X_train_w2v_binary)
X_test_w2v_binary = np.array(X_test_w2v_binary)

X_train_w2v_binary.shape, X_test_w2v_binary.shape


# In[39]:


# y_train_one_hot = to_categorical(y_train_binary)
# y_test_one_hot = to_categorical(y_test_binary)


# In[40]:


perceptron = Perceptron(max_iter=1000)

perceptron.fit(X_train_w2v_binary, y_train_binary)
y_pred = perceptron.predict(X_test_w2v_binary)
accuracy_perceptron_w2v_pre = accuracy_score(y_test_binary, y_pred)
print(f"Accuracy of the Single Layer Perceptron using Word2Vec Pretrained Features: {accuracy_perceptron_w2v_pre}")


# In[41]:


svm = LinearSVC()

svm.fit(X_train_w2v_binary, y_train_binary)
y_pred = svm.predict(X_test_w2v_binary)
accuracy_svm_w2v_pre = accuracy_score(y_test_binary, y_pred)
print(f"Accuracy of the SVM using Word2Vec Pretrained Features: {accuracy_svm_w2v_pre}")


# ## Extract the Word2Vec Custom Features from Dataset and train Perceptron and SVM

# Extracting the Word2Vec feature vectors on the Custom trained model from the initial dataset, and using that data set to train the Perceptron and SVM, and calculate the accuracy on the testing dataset

# In[42]:


def get_doc_embedding(doc):
    doc_str = ' '.join(doc)  # Join tokens within the document list to form a single string
    words = doc_str.lower().split()  # Split the string into words
    if not words:
        return np.zeros(300)  # Return a zero vector if no words are present
    else:
        return model_own.wv.get_mean_vector(words)  # Compute the mean vector using Word2Vec model

X_train_w2v_own_binary = []
X_test_w2v_own_binary = []

# Extract Word2Vec features for training data
for e in tqdm(X_train_binary):
    X_train_w2v_own_binary.append(get_doc_embedding(e))

# Extract Word2Vec features for testing data
for e in tqdm(X_test_binary):
    X_test_w2v_own_binary.append(get_doc_embedding(e))
    
X_train_w2v_own_binary = np.array(X_train_w2v_own_binary)
X_test_w2v_own_binary = np.array(X_test_w2v_own_binary)

X_train_w2v_own_binary.shape, X_test_w2v_own_binary.shape


# In[43]:


perceptron = Perceptron(max_iter=1000)

perceptron.fit(X_train_w2v_own_binary, y_train_binary)
y_pred = perceptron.predict(X_test_w2v_own_binary)
accuracy_perceptron_w2v_own = accuracy_score(y_test_binary, y_pred)
print(f"Accuracy of the Single Layer Perceptron using Word2Vec Custom Features: {accuracy_perceptron_w2v_own}")


# In[44]:


svm = LinearSVC()

svm.fit(X_train_w2v_own_binary, y_train_binary)
y_pred = svm.predict(X_test_w2v_own_binary)
accuracy_svm_w2v_own = accuracy_score(y_test_binary, y_pred)
print(f"Accuracy of the SVM using Word2Vec Custom Features: {accuracy_svm_w2v_own}")


# ## 4. Feed Forward Neural Networks

# ## Multi Layer Perceptron with Binary Classification

# Using the same binary dataset vectors that we have generated from the Word2Vec Pre-trained and Custom models that we have done in Question 3 to train the Multi-layer perceptron and evaluate the accuracy on the testing dataset.
# 
# Here as well we need to encode the classes to 0 and 1, as the class labels shall start from '0'

# In[45]:


# Convert labels to NumPy arrays
y_train_np = np.array(y_train_binary)
y_test_np = np.array(y_test_binary)

# Convert labels to binary format (0 for class 1, 1 for class 2)
y_train_binary = np.where(y_train_np == 1, 0, 1)
y_test_binary = np.where(y_test_np == 1, 0, 1)

# Define the MLP Model for binary classification
class BinaryMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(BinaryMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

# Training Loop
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")

# Evaluation
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {correct/total}")


# ## Extract the Word2Vec Custom Features and train a Multi Layer Perceptron with Binary Classification

# In[46]:


# Convert data to PyTorch tensors
train_dataset_own_binary = TensorDataset(torch.tensor(X_train_w2v_own_binary, dtype=torch.float32), torch.tensor(y_train_binary, dtype=torch.long))
test_dataset_own_binary = TensorDataset(torch.tensor(X_test_w2v_own_binary, dtype=torch.float32), torch.tensor(y_test_binary, dtype=torch.long))

# Create DataLoader
train_loader_own_binary = DataLoader(train_dataset_own_binary, batch_size=32, shuffle=True)
test_loader_own_binary = DataLoader(test_dataset_own_binary, batch_size=32)

# Initialize the model
input_size = X_train_w2v_own_binary.shape[1]  # Size of input features
hidden_size1 = 50
hidden_size2 = 10
output_size = 2  # Binary classification (classes 1 and 2)
binary_model_own_binary = BinaryMLP(input_size, hidden_size1, hidden_size2, output_size)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(binary_model_own_binary.parameters(), lr=0.001)

# Train the binary classification model
num_epochs = 50
train(binary_model_own_binary, train_loader_own_binary, criterion, optimizer, num_epochs)

# Evaluate the binary classification model
evaluate(binary_model_own_binary, test_loader_own_binary)


# ## Extract the Word2Vec Pretrained Features and train a Multi Layer Perceptron with Binary Classification

# In[47]:


# Convert data to PyTorch tensors
train_dataset_pre_binary = TensorDataset(torch.tensor(X_train_w2v_binary, dtype=torch.float32), torch.tensor(y_train_binary, dtype=torch.long))
test_dataset_pre_binary = TensorDataset(torch.tensor(X_test_w2v_binary, dtype=torch.float32), torch.tensor(y_test_binary, dtype=torch.long))

# Create DataLoader
train_loader_pre_binary = DataLoader(train_dataset_pre_binary, batch_size=32, shuffle=True)
test_loader_pre_binary = DataLoader(test_dataset_pre_binary, batch_size=32)

# Initialize the model
input_size = X_train_w2v_binary.shape[1]  # Size of input features
hidden_size1 = 50
hidden_size2 = 10
output_size = 2  # Binary classification (classes 1 and 2)
binary_model_pre_binary = BinaryMLP(input_size, hidden_size1, hidden_size2, output_size)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(binary_model_pre_binary.parameters(), lr=0.001)

# Train the binary classification model
num_epochs = 50
train(binary_model_pre_binary, train_loader_pre_binary, criterion, optimizer, num_epochs)

# Evaluate the binary classification model
evaluate(binary_model_pre_binary, test_loader_pre_binary)


# ## Multi Layer Perceptron with Ternary Classification

# Here we shall be using the original dataset, that has 3 classes 1, 2 and 3. then calculating the Word2Vec vectors for both the pre-trained and Custom model and using them to train the multi-layer perceptron and evaluating on the testing set.
# 
# Here in this case as well we need to encode the class labels from 1 to 3 to 0 to 2, as the class labels shall start from '0' only.

# In[48]:


def get_doc_embedding(doc):
    doc_str = ' '.join(doc)  # Join tokens within the document list to form a single string
    words = doc_str.lower().split()  # Split the string into words
    if not words:
        return np.zeros(300)  # Return a zero vector if no words are present
    else:
        return model_own.wv.get_mean_vector(words)  # Compute the mean vector using Word2Vec model

X_train_w2v_own = []
X_test_w2v_own = []

# Extract Word2Vec features for training data
for e in tqdm(X_train_nltk_preprocessed):
    X_train_w2v_own.append(get_doc_embedding(e))

# Extract Word2Vec features for testing data
for e in tqdm(X_test):
    X_test_w2v_own.append(get_doc_embedding(e))
    
X_train_w2v_own = np.array(X_train_w2v_own)
X_test_w2v_own = np.array(X_test_w2v_own)

X_train_w2v_own.shape, X_test_w2v_own.shape


# In[49]:


def get_doc_embedding(doc):
    doc_str = ' '.join(doc)  # Join tokens within the document list to form a single string
    words = doc_str.lower().split()  # Split the string into words
    if not words:
        return np.zeros(300)  # Return a zero vector if no words are present
    else:
        return wv.get_mean_vector(words)  # Compute the mean vector using Word2Vec model

X_train_w2v = []
X_test_w2v = []

# Extract Word2Vec features for training data
for e in tqdm(X_train_nltk_preprocessed):
    X_train_w2v.append(get_doc_embedding(e))

# Extract Word2Vec features for testing data
for e in tqdm(X_test):
    X_test_w2v.append(get_doc_embedding(e))
    
X_train_w2v = np.array(X_train_w2v)
X_test_w2v = np.array(X_test_w2v)

X_train_w2v.shape, X_test_w2v.shape


# In[50]:


# Convert labels to NumPy arrays
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

# Convert labels to integers ranging from 0 to 2
y_train_encoded = [label - 1 for label in y_train_np]  # Assuming labels start from 1
y_test_encoded = [label - 1 for label in y_test_np]  # Assuming labels start from 1

# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

# Training Loop
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")

# Evaluation
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {correct/total}")


# ## Extract the Word2Vec Pre-trained Features and train a Multi Layer Perceptron

# In[51]:


# Convert data to PyTorch tensors
train_dataset_pre = TensorDataset(torch.tensor(X_train_w2v, dtype=torch.float32), torch.tensor(y_train_encoded, dtype=torch.long))
test_dataset_pre = TensorDataset(torch.tensor(X_test_w2v, dtype=torch.float32), torch.tensor(y_test_encoded, dtype=torch.long))

# Create DataLoader
train_loader_pre = DataLoader(train_dataset_pre, batch_size=32, shuffle=True)
test_loader_pre = DataLoader(test_dataset_pre, batch_size=32)

# Initialize the model
input_size = X_train_w2v.shape[1]  # Size of input features
hidden_size1 = 50
hidden_size2 = 10
output_size = 3  # Assuming 3 classes for sentiment analysis
model_pre = MLP(input_size, hidden_size1, hidden_size2, output_size)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_pre.parameters(), lr=0.001)

# Train the model
num_epochs = 50
train(model_pre, train_loader_pre, criterion, optimizer, num_epochs)

# Evaluate the model
evaluate(model_pre, test_loader_pre)


# ## Extract the Word2Vec Custom Features and train a Multi Layer Perceptron

# In[52]:


# Convert data to PyTorch tensors
train_dataset_own = TensorDataset(torch.tensor(X_train_w2v_own, dtype=torch.float32), torch.tensor(y_train_encoded, dtype=torch.long))
test_dataset_own = TensorDataset(torch.tensor(X_test_w2v_own, dtype=torch.float32), torch.tensor(y_test_encoded, dtype=torch.long))

# Create DataLoader
train_loader_own = DataLoader(train_dataset_own, batch_size=32, shuffle=True)
test_loader_own = DataLoader(test_dataset_own, batch_size=32)

# Initialize the model
input_size = X_train_w2v_own.shape[1]  # Size of input features
hidden_size1 = 50
hidden_size2 = 10
output_size = 3  # Assuming 3 classes for sentiment analysis
model_own = MLP(input_size, hidden_size1, hidden_size2, output_size)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_own.parameters(), lr=0.001)

# Train the model
num_epochs = 50
train(model_own, train_loader_own, criterion, optimizer, num_epochs)

# Evaluate the model
evaluate(model_own, test_loader_own)


# Here, I am extracting the first 10 Word2Vec features, and using them for the ternary classification as well as using the binary features that we have extracted to train the binary Multi layer perceptron on the training dataset and evaluating their performance on the testing dataset.

# ## Using the First 10 Word2Vec vectors on Ternary Classification for the Word2Vec Custom Model

# In[53]:


# Concatenate the first 10 Word2Vec vectors for each review
X_train_concat_own = []
X_test_concat_own = []

for review in X_train_w2v_own:
    review_reshaped = review.reshape(1, -1)  # Reshape to ensure it's a 2D array
    concatenated_vector = np.concatenate(review_reshaped[:10], axis=0)
    X_train_concat_own.append(concatenated_vector)

for review in X_test_w2v_own:
    review_reshaped = review.reshape(1, -1)  # Reshape to ensure it's a 2D array
    concatenated_vector = np.concatenate(review_reshaped[:10], axis=0)
    X_test_concat_own.append(concatenated_vector)

# Convert the concatenated features into PyTorch tensors
X_train_concat_tensor_own = torch.tensor(X_train_concat_own, dtype=torch.float32)
X_test_concat_tensor_own = torch.tensor(X_test_concat_own, dtype=torch.float32)

# Convert labels to ternary format
y_train_tensor_own = torch.tensor(y_train_encoded, dtype=torch.long)
y_test_tensor_own = torch.tensor(y_test_encoded, dtype=torch.long)

# Create DataLoader
train_dataset_concat_own = TensorDataset(X_train_concat_tensor_own, y_train_tensor_own)
test_dataset_concat_own = TensorDataset(X_test_concat_tensor_own, y_test_tensor_own)

train_loader_concat_own = DataLoader(train_dataset_concat_own, batch_size=32, shuffle=True)
test_loader_concat_own = DataLoader(test_dataset_concat_own, batch_size=32)

# Initialize the model
input_size = X_train_concat_tensor_own.shape[1]  # Size of input features
hidden_size1 = 50
hidden_size2 = 10
output_size = 3  # Assuming 3 classes for sentiment analysis
model_concat_own = MLP(input_size, hidden_size1, hidden_size2, output_size)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_concat_own.parameters(), lr=0.001)

# Train the model
num_epochs = 50
train(model_concat_own, train_loader_concat_own, criterion, optimizer, num_epochs)

# Evaluate the model
evaluate(model_concat_own, test_loader_concat_own)


# ## Using the First 10 Word2Vec vectors on Ternary Classification for the Word2Vec Pre-Trained Model

# In[54]:


# Concatenate the first 10 Word2Vec vectors for each review
X_train_concat = []
X_test_concat = []

for review in X_train_w2v:
    review_reshaped = review.reshape(1, -1)  # Reshape to ensure it's a 2D array
    concatenated_vector = np.concatenate(review_reshaped[:10], axis=0)
    X_train_concat.append(concatenated_vector)

for review in X_test_w2v:
    review_reshaped = review.reshape(1, -1)  # Reshape to ensure it's a 2D array
    concatenated_vector = np.concatenate(review_reshaped[:10], axis=0)
    X_test_concat.append(concatenated_vector)

# Convert the concatenated features into PyTorch tensors
X_train_concat_tensor = torch.tensor(X_train_concat, dtype=torch.float32)
X_test_concat_tensor = torch.tensor(X_test_concat, dtype=torch.float32)

# Convert labels to ternary format
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

# Create DataLoader
train_dataset_concat = TensorDataset(X_train_concat_tensor, y_train_tensor)
test_dataset_concat = TensorDataset(X_test_concat_tensor, y_test_tensor)

train_loader_concat = DataLoader(train_dataset_concat, batch_size=32, shuffle=True)
test_loader_concat = DataLoader(test_dataset_concat, batch_size=32)

# Initialize the model
input_size = X_train_concat_tensor.shape[1]  # Size of input features
hidden_size1 = 50
hidden_size2 = 10
output_size = 3  # Assuming 3 classes for sentiment analysis
model_concat = MLP(input_size, hidden_size1, hidden_size2, output_size)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_concat.parameters(), lr=0.001)

# Train the model
num_epochs = 50
train(model_concat, train_loader_concat, criterion, optimizer, num_epochs)

# Evaluate the model
evaluate(model_concat, test_loader_concat)


# ## Using the First 10 Word2Vec vectors on Binary Classification for the Word2Vec Pre-Trained Model

# In[55]:


# Concatenate the first 10 Word2Vec vectors for each review
X_train_concat = []
X_test_concat = []

for review in X_train_w2v_binary:
    review_reshaped = review.reshape(1, -1)  # Reshape to ensure it's a 2D array
    concatenated_vector = np.concatenate(review_reshaped[:10], axis=0)
    X_train_concat.append(concatenated_vector)

for review in X_test_w2v_binary:
    review_reshaped = review.reshape(1, -1)  # Reshape to ensure it's a 2D array
    concatenated_vector = np.concatenate(review_reshaped[:10], axis=0)
    X_test_concat.append(concatenated_vector)

# Convert the concatenated features into PyTorch tensors
X_train_concat_tensor = torch.tensor(X_train_concat, dtype=torch.float32)
X_test_concat_tensor = torch.tensor(X_test_concat, dtype=torch.float32)

# Convert labels to ternary format
y_train_tensor = torch.tensor(y_train_binary, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_binary, dtype=torch.long)

# Create DataLoader
train_dataset_concat = TensorDataset(X_train_concat_tensor, y_train_tensor)
test_dataset_concat = TensorDataset(X_test_concat_tensor, y_test_tensor)

train_loader_concat = DataLoader(train_dataset_concat, batch_size=32, shuffle=True)
test_loader_concat = DataLoader(test_dataset_concat, batch_size=32)

# Initialize the model
input_size = X_train_concat_tensor.shape[1]  # Size of input features
hidden_size1 = 50
hidden_size2 = 10
output_size = 2  # Assuming 2 classes for sentiment analysis
model_concat = MLP(input_size, hidden_size1, hidden_size2, output_size)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_concat.parameters(), lr=0.001)

# Train the model
num_epochs = 50
train(model_concat, train_loader_concat, criterion, optimizer, num_epochs)

# Evaluate the model
evaluate(model_concat, test_loader_concat)


# ## Using the First 10 Word2Vec vectors on Binary Classification for the Word2Vec Custom Model

# In[56]:


# Concatenate the first 10 Word2Vec vectors for each review
X_train_concat = []
X_test_concat = []

for review in X_train_w2v_own_binary:
    review_reshaped = review.reshape(1, -1)  # Reshape to ensure it's a 2D array
    concatenated_vector = np.concatenate(review_reshaped[:10], axis=0)
    X_train_concat.append(concatenated_vector)

for review in X_test_w2v_own_binary:
    review_reshaped = review.reshape(1, -1)  # Reshape to ensure it's a 2D array
    concatenated_vector = np.concatenate(review_reshaped[:10], axis=0)
    X_test_concat.append(concatenated_vector)

# Convert the concatenated features into PyTorch tensors
X_train_concat_tensor = torch.tensor(X_train_concat, dtype=torch.float32)
X_test_concat_tensor = torch.tensor(X_test_concat, dtype=torch.float32)

# Convert labels to ternary format
y_train_tensor = torch.tensor(y_train_binary, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_binary, dtype=torch.long)

# Create DataLoader
train_dataset_concat = TensorDataset(X_train_concat_tensor, y_train_tensor)
test_dataset_concat = TensorDataset(X_test_concat_tensor, y_test_tensor)

train_loader_concat = DataLoader(train_dataset_concat, batch_size=32, shuffle=True)
test_loader_concat = DataLoader(test_dataset_concat, batch_size=32)

# Initialize the model
input_size = X_train_concat_tensor.shape[1]  # Size of input features
hidden_size1 = 50
hidden_size2 = 10
output_size = 2  # Assuming 2 classes for sentiment analysis
model_concat = MLP(input_size, hidden_size1, hidden_size2, output_size)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_concat.parameters(), lr=0.001)

# Train the model
num_epochs = 50
train(model_concat, train_loader_concat, criterion, optimizer, num_epochs)

# Evaluate the model
evaluate(model_concat, test_loader_concat)


# The comparison between MLPs and simpler models like perceptrons and Support Vector Machines (SVMs) highlights the trade-offs between model complexity and performance. While the MLPs have definetly over-performed the simple models in terms of the testing accuracy. However the training time between the MLPs and Simple Models is significantly higher, so that is a negative of the MLPs. Perceptrons and SVMs, while less complex, are easier to interpret and computationally more efficient.

# ## 5. Convolutional Neural Networks

# ## CNN with Binary Classification using the Word2Vec Pre-Trained Vectors

# Here I am splitting the dataset again from the initial data, and removing the sentiment values with class 3, as we are doing a binary classification. and encoding the class labels fro 1 and 2 to  and 1 as the class labels have to start from '0'.
# 
# I am using a CNN, with the input size as 300 features from the Word2Vec feature vectors and the output size to be 2 as we are working on a binary classification.
# 
# I am also printing the accuracy of the model on the test set.

# In[37]:


# Joining features and targets for training dataset
train_data = pd.concat([X_train_nltk_preprocessed, y_train], axis=1)
train_data_filtered = train_data[train_data['sentiment'] != 3]
X_train_binary = train_data_filtered['review_body']
y_train_binary = train_data_filtered['sentiment']

# Joining features and targets for testing dataset
test_data = pd.concat([X_test, y_test], axis=1)
test_data_filtered = test_data[test_data['sentiment'] != 3]
X_test_binary = test_data_filtered['review_body']
y_test_binary = test_data_filtered['sentiment']


# In[38]:


import gensim.downloader as api
wv = api.load('word2vec-google-news-300')


# In[39]:


# Preprocess text data
def preprocess_text(text, max_length=50):
    tokens = text.split()[:max_length]  # Limit maximum review length
    padded_tokens = tokens + ['<PAD>'] * (max_length - len(tokens))  # Pad shorter reviews
    return padded_tokens

# Convert text data into Word2Vec vectors
def text_to_vectors(texts, wv, max_length=50):
    vectors = []
    for text in texts:
        tokens = preprocess_text(text, max_length)
        vector = [wv[word] if word in wv else np.zeros(wv.vector_size) for word in tokens]
        vectors.append(vector)
    return np.array(vectors)

# Prepare data
X_train_vectors = text_to_vectors(X_train_binary, wv)
X_test_vectors = text_to_vectors(X_test_binary, wv)

# Convert labels to binary format (class 1 and class 2)
y_train_binary = np.where(y_train_binary == 1, 0, 1)
y_test_binary = np.where(y_test_binary == 1, 0, 1)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_vectors, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_binary, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_vectors, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_binary, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# In[40]:


# Define the CNN model
class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(50, 10, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(10 * 12, output_size)  # Adjust the input size for the linear layer

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 10 * 12)  # Adjust the reshaping operation
        x = self.fc(x)
        return nn.functional.softmax(x, dim=1)

# Train the CNN model
def train_cnn(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.permute(0, 2, 1))  # Permute input dimensions for Conv1D
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")

# Evaluation
def evaluate_cnn(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.permute(0, 2, 1))  # Permute input dimensions for Conv1D
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {correct / total}")


# In[41]:


# Initialize the model
input_size = X_train_vectors.shape[2]  # Size of input features
output_size = 2  # Binary classification
model = CNN(input_size, output_size)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Train the model
train_cnn(model, train_loader, criterion, optimizer, num_epochs=10)

# Evaluate the model
evaluate_cnn(model, test_loader)


# ## CNN with Binary Classification using the Word2Vec Custom Vectors

# In[42]:


# Joining features and targets for training dataset
train_data = pd.concat([X_train_nltk_preprocessed, y_train], axis=1)
train_data_filtered = train_data[train_data['sentiment'] != 3]
X_train_binary = train_data_filtered['review_body']
y_train_binary = train_data_filtered['sentiment']

# Joining features and targets for testing dataset
test_data = pd.concat([X_test, y_test], axis=1)
test_data_filtered = test_data[test_data['sentiment'] != 3]
X_test_binary = test_data_filtered['review_body']
y_test_binary = test_data_filtered['sentiment']


# In[43]:


from gensim.models import Word2Vec
# Train the Word2Vec model
model_own = Word2Vec(X_train_nltk_preprocessed_new, vector_size=300, window=11, min_count=10)


# In[44]:


# Prepare data
X_train_vectors = text_to_vectors(X_train_binary, model_own.wv)
X_test_vectors = text_to_vectors(X_test_binary, model_own.wv)

# Convert labels to binary format (class 1 and class 2)
y_train_binary = np.where(y_train_binary == 1, 0, 1)
y_test_binary = np.where(y_test_binary == 1, 0, 1)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_vectors, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_binary, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_vectors, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_binary, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# In[45]:


# Initialize the model
input_size = X_train_tensor.shape[2]  # Size of input features
output_size = 2  # Binary classification
model = CNN(input_size, output_size)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Train the model
train_cnn(model, train_loader, criterion, optimizer, num_epochs=10)

# Evaluate the model
evaluate_cnn(model, test_loader)


# ## CNN with Ternary Classification with Pre-Trained Word2Vec model

# Here I am using the dataset again from the initial data, and encoding the class labels from 1, 2 and 3 to 0, 1 and 2 as the class labels have to start from '0'.
# 
# I am using a CNN, with the input size as 300 features from the Word2Vec feature vectors and the output size to be 3 as we are working on a Ternary classification.
# 
# I am also printing the accuracy of the model on the test set.

# In[16]:


# Joining features and targets for training dataset
train_data = pd.concat([X_train_nltk_preprocessed, y_train], axis=1)
X_train = train_data['review_body']
y_train = train_data['sentiment']

# Joining features and targets for testing dataset
test_data = pd.concat([X_test, y_test], axis=1)
X_test = test_data['review_body']
y_test = test_data['sentiment']


# In[20]:


import gensim.downloader as api
wv = api.load('word2vec-google-news-300')


# In[24]:


# Preprocess text data
def preprocess_text(text, max_length=50):
    tokens = text.split()[:max_length]  # Limit maximum review length
    padded_tokens = tokens + ['<PAD>'] * (max_length - len(tokens))  # Pad shorter reviews
    return padded_tokens

# Convert text data into Word2Vec vectors
def text_to_vectors(texts, wv, max_length=50):
    vectors = []
    for text in texts:
        tokens = preprocess_text(text, max_length)
        vector = [wv[word] if word in wv else np.zeros(wv.vector_size) for word in tokens]
        vectors.append(vector)
    return np.array(vectors)

# Prepare data
X_train_vectors = text_to_vectors(X_train, wv)
X_test_vectors = text_to_vectors(X_test, wv)

# Convert labels to ternary format (classes 1, 2, and 3)
y_train_ternary = np.array(y_train)  # Convert Pandas Series to NumPy array
y_test_ternary = np.array(y_test)  # Convert Pandas Series to NumPy array

# Convert labels to the range [0, 1, 2]
y_train_ternary = y_train_ternary - 1
y_test_ternary = y_test_ternary - 1

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_vectors, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_ternary, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_vectors, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_ternary, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# In[16]:


# Define the CNN model for three classes
class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(50, 10, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(10 * 12, output_size)  # Adjust the input size for the linear layer

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 10 * 12)  # Adjust the reshaping operation
        x = self.fc(x)
        return nn.functional.softmax(x, dim=1)

# Train the CNN model for three classes
def train_cnn(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.permute(0, 2, 1))  # Permute input dimensions for Conv1D
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")

# Evaluation for three classes
def evaluate_cnn(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.permute(0, 2, 1))  # Permute input dimensions for Conv1D
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {correct / total}")


# In[26]:


# Initialize the model for ternary classification
input_size = X_train_vectors.shape[2]  # Size of input features
output_size = 3  # Ternary classification
model = CNN(input_size, output_size)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_cnn(model, train_loader, criterion, optimizer, num_epochs=10)

# Evaluate the model
evaluate_cnn(model, test_loader)


# ## CNN with Ternary Classification using the Word2Vec Custom Model

# In[17]:


# Joining features and targets for training dataset
train_data = pd.concat([X_train_nltk_preprocessed, y_train], axis=1)
X_train = train_data['review_body']
y_train = train_data['sentiment']

# Joining features and targets for testing dataset
test_data = pd.concat([X_test, y_test], axis=1)
X_test = test_data['review_body']
y_test = test_data['sentiment']


# In[28]:


from gensim.models import Word2Vec
# Train the Word2Vec model
model_own = Word2Vec(X_train_nltk_preprocessed_new, vector_size=300, window=11, min_count=10)


# In[18]:


# Preprocess text data
def preprocess_text(text, max_length=50):
    tokens = text.split()[:max_length]  # Limit maximum review length
    padded_tokens = tokens + ['<PAD>'] * (max_length - len(tokens))  # Pad shorter reviews
    return padded_tokens

# Convert text data into Word2Vec vectors
def text_to_vectors(texts, wv, max_length=50):
    vectors = []
    for text in texts:
        tokens = preprocess_text(text, max_length)
        vector = [wv[word] if word in wv else np.zeros(wv.vector_size) for word in tokens]
        vectors.append(vector)
    return np.array(vectors)

# Prepare data
X_train_vectors = text_to_vectors(X_train, model_own.wv)
X_test_vectors = text_to_vectors(X_test, model_own.wv)

# Convert labels to ternary format (classes 1, 2, and 3)
y_train_ternary = np.array(y_train)  # Convert Pandas Series to NumPy array
y_test_ternary = np.array(y_test)  # Convert Pandas Series to NumPy array

# Convert labels to the range [0, 1, 2]
y_train_ternary = y_train_ternary - 1
y_test_ternary = y_test_ternary - 1

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_vectors, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_ternary, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_vectors, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_ternary, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# In[19]:


# Initialize the model for ternary classification
input_size = X_train_vectors.shape[2]  # Size of input features
output_size = 3  # Ternary classification
model = CNN(input_size, output_size)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_cnn(model, train_loader, criterion, optimizer, num_epochs=10)

# Evaluate the model
evaluate_cnn(model, test_loader)

