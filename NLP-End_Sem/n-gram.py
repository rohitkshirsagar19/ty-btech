import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
# Download the punkt tokenizer
nltk.download('punkt')


# Sample text
text = "I love playing Badminton and Cricket. I also enjoy watching movies."
print("\n \nOriginal Text:", text)


# Tokenize the text
tokens = word_tokenize(text)

# Generate bigrams (2-grams)
bigrams = list(ngrams(tokens, 2))
print("\nBigrams:", bigrams)

# Generate trigrams (3-grams)
trigrams = list(ngrams(tokens, 3))
print("\nTrigrams:", trigrams)

# You can generate any n-gram by changing the number
n = 4
fourgrams = list(ngrams(tokens, n))
print(f"\n{n}-grams:", fourgrams)
