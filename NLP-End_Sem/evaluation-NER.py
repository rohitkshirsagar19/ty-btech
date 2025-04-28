import spacy
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


# Load English NER model
nlp = spacy.load("en_core_web_sm")
# python -m spacy download en_core_web_sm

# Sample text
text = "Elon Musk founded SpaceX in the United States."

# Process text
doc = nlp(text)

# Predicted entities
pred_entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Predicted Entities:", pred_entities)

# Expected entities manually
true_entities = [('Elon Musk', 'PERSON'), ('SpaceX', 'ORG'), ('United States', 'GPE')]

# Separate labels for evaluation
true_labels = [label for _, label in true_entities]
pred_labels = [label for _, label in pred_entities]

print("\nTrue Labels:", true_labels)
print("Predicted Labels:", pred_labels)
# Fix: Check if true and pred lengths mismatch
min_len = min(len(true_labels), len(pred_labels))
true_labels = true_labels[:min_len]
pred_labels = pred_labels[:min_len]

# Evaluate
print("\nEvaluation Report:\n")
print(classification_report(true_labels, pred_labels))

for ent in doc.ents:
    print(ent.text, "->", ent.label_)
