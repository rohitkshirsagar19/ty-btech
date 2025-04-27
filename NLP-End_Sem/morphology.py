# List of word pairs (original -> transformed)
words = [
    ("walk", "walked"),
    ("make", "making"),
    ("run", "runner"),
    ("study", "studies"),
    ("happy", "happier")
]

print("Original    Transformed    Add    Delete")
print("-----------------------------------------")

# Loop through each word pair
for orig, trans in words:
    add = ""
    delete = ""
    
    # If original is part of transformed
    if trans.startswith(orig):
        add = trans[len(orig):]
    else:
        # Find matching part
        i = 0
        while i < min(len(orig), len(trans)) and orig[i] == trans[i]:
            i += 1
        delete = orig[i:]
        add = trans[i:]
    
    # Print results
    print(f"{orig:<12}{trans:<15}{add:<7}{delete}")
