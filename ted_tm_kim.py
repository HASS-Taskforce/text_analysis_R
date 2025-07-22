# 1. Import libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# 2. Load TED Talks dataset
url = "https://raw.githubusercontent.com/kinnaird-laudun/data/refs/heads/main/Release_v0/TEDonly_final.csv"
df = pd.read_csv(url)

# 3. Basic cleaning: remove stage directions like (Music), (Applause)
df['text'] = df['text'].astype(str).apply(lambda x: re.sub(r"\([^)]*\)", "", x))

# 4. Tokenization and stopword removal
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(preprocess)

# 5. Create a document-term matrix
vectorizer = CountVectorizer(max_df=0.95, min_df=5)  # filter very rare and very common terms
dtm = vectorizer.fit_transform(df['clean_text'])

# 6. Fit LDA model
lda = LatentDirichletAllocation(n_components=4, random_state=42)
lda.fit(dtm)

# 7. Display top words per topic
def display_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}: ", end='')
        print(" | ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

feature_names = vectorizer.get_feature_names_out()
display_topics(lda, feature_names, 7)