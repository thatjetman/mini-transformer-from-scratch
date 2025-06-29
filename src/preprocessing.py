
import pandas as pd
from collections import defaultdict

def load_translation_data(filepath):
    # Read CSV
    df = pd.read_csv(filepath)
    
    # Debug: Print column names
    print("CSV columns in preprocessing.py:", df.columns.tolist())
    
    # Normalize column names (handle case sensitivity)
    df.columns = df.columns.str.lower().str.strip()
    
    # Rename columns to match expected names
    if 'en' in df.columns and 'fr' in df.columns:
        df = df.rename(columns={'en': 'source_text', 'fr': 'target_text'})
    elif 'source_text' in df.columns and 'target_text' in df.columns:
        pass  # Already correctly named
    else:
        raise ValueError(f"Expected columns 'en/fr' or 'source_text/target_text', got {df.columns.tolist()}")

    # Convert sentences to lists of words
    source_sentences = [str(s).lower().split() for s in df['source_text']]
    target_sentences = [str(s).lower().split() for s in df['target_text']]
    return source_sentences, target_sentences

def build_vocab(sentences):
    vocab = defaultdict(lambda: len(vocab))
    vocab["<pad>"]
    vocab["<sos>"]
    vocab["<eos>"]
    for sentence in sentences:
        for word in sentence:
            vocab[word]
    return dict(vocab)

def tensorize(sentences, vocab, max_len=10):
    tensor_data = []
    for sentence in sentences:
        tensor = [vocab["<sos>"]] + [vocab[word] for word in sentence if word in vocab] + [vocab["<eos>"]]
        tensor = tensor[:max_len] + [vocab["<pad>"]] * (max_len - len(tensor))
        tensor_data.append(tensor)
    return tensor_data
