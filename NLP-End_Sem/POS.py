import nltk
from nltk import pos_tag, word_tokenize

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Input sentence
sentence = "The quick brown fox jumps over the lazy dog."

# Tokenize the sentence
tokens = word_tokenize(sentence)

# Perform POS tagging
pos_tags = pos_tag(tokens)

# Print the tokens and their corresponding POS tags
for token, tag in pos_tags:
    print(f"{token}: {tag}")

