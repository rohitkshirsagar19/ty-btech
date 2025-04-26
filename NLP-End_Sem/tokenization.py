import nltk
from nltk.tokenize import WhitespaceTokenizer, word_tokenize, TreebankWordTokenizer, TweetTokenizer, MWETokenizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

# Downloads
nltk.download('punkt')
nltk.download('wordnet')

# Sample sentence
sentence = "Cats, dogs, and foxes are running quickly! Can't you see them?"
print("Original Sentence:", sentence)

# 1. Tokenization
print("\n--- Tokenization ---")

# Whitespace Tokenizer
white_tokens = WhitespaceTokenizer().tokenize(sentence)
print("Whitespace Tokenizer:", white_tokens)

# Punctuation-based (simple word_tokenize)
punct_tokens = word_tokenize(sentence)
print("Punctuation-based Tokenizer:", punct_tokens)

# Treebank Tokenizer
treebank_tokens = TreebankWordTokenizer().tokenize(sentence)
print("Treebank Tokenizer:", treebank_tokens)

# Tweet Tokenizer
tweet_tokens = TweetTokenizer().tokenize(sentence)
print("Tweet Tokenizer:", tweet_tokens)

# MWE Tokenizer (Multi-Word Expression Tokenizer)
mwe_tokenizer = MWETokenizer([('New', 'York'), ('ice', 'cream')])
mwe_tokens = mwe_tokenizer.tokenize(word_tokenize("I love New York and ice cream."))
print("MWE Tokenizer:", mwe_tokens)

# 2. Stemming
print("\n--- Stemming ---")

porter = PorterStemmer()
snowball = SnowballStemmer('english')

# Example words
words = ['running', 'flies', 'easily', 'fairly']

print("Porter Stemmer:", [porter.stem(w) for w in words])
print("Snowball Stemmer:", [snowball.stem(w) for w in words])

# 3. Lemmatization
print("\n--- Lemmatization ---")

lemmatizer = WordNetLemmatizer()

# Lemmatize words
print("Lemmatized words:", [lemmatizer.lemmatize(w) for w in words])
