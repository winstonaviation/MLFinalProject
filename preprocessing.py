import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_and_split(filepath, text_col, label_col, test_size=0.2, val_size=0.1, random_state=42, extra_text_cols=None):
    df = pd.read_csv(filepath)

    # combine multiple text columns before dropping nulls
    if extra_text_cols:
        for col in extra_text_cols:
            df[text_col] = df[text_col].fillna('') + ' ' + df[col].fillna('')
        df = df[[text_col, label_col]].dropna()
    else:
        df = df[[text_col, label_col]].dropna()
    #only needed for big reddit dataset
    #df[text_col] = df[text_col].str[:1000]
    X = df[text_col].to_numpy(dtype=str)
    y = df[label_col].to_numpy()
    
    #first split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # split off validation from remaining
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=val_size/(1-test_size), 
        random_state=random_state, 
        stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_tfidf(X_train):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        lowercase=True,
        # sublinear_tf left false (default) to preserve raw term frequency
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    return vectorizer, X_train_tfidf


def transform(vectorizer, X):
    return vectorizer.transform(X)