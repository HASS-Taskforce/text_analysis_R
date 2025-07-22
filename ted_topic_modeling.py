import pandas as pd
import re
from gensim import corpora
from gensim.models import LdaModel
import spacy
from nltk.corpus import stopwords


# Load TED data
df = pd.read_csv("https://raw.githubusercontent.com/kinnaird-laudun/data/refs/heads/main/Release_v0/TEDonly_final.csv")

# Remove stage directions (e.g., (Music), (Applause))
df['text'] = df['text'].apply(lambda x: re.sub(r"\([^)]*\)", "", str(x)))

# Load English language model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
stop_words = set(stopwords.words('english'))

# Tokenize and clean
def preprocess(text):
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc
            if token.is_alpha and token.lemma_ not in stop_words and len(token) > 2]

df['tokens'] = df['text'].map(preprocess)

# Create dictionary and corpus
dictionary = corpora.Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(text) for text in df['tokens']]

# Train LDA model
lda_model = LdaModel(corpus=corpus,
                     id2word=dictionary,
                     num_topics=4,
                     random_state=42,
                     passes=10,
                     per_word_topics=True)

# Show topics
topics = lda_model.print_topics(num_words=7)
for i, topic in topics:
    print(f"Topic {i}: {topic}")
