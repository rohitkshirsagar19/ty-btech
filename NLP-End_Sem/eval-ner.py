import spacy

# Load the English model
nlp = spacy.load("en_core_web_sm")


# Sample text for NER
text = "Apple is planning to open a new office in San Francisco by 2026. Tim Cook will lead the project."

# Process the text
doc = nlp(text)

# Extract and print entities
print("Named Entities, Phrases, and Concepts:")
for ent in doc.ents:
    print(ent.text, "->", ent.label_)

