import nltk
from nltk.corpus import wordnet as wn

# Download the necessary NLTK resources
nltk.download('wordnet')

# Word to analyze
word = "fast"

synonyms = []
antonyms = []

# Get synonyms and antonyms from WordNet
for syn in wn.synsets(word):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())

# Remove duplicates
synonyms = set(synonyms)
antonyms = set(antonyms)

# Print the results
print(f"Word: {word}")
print("Synonyms:", synonyms)
print("Antonyms:", antonyms)