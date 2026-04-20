from sklearn.naive_bayes import MultinomialNB

def get_models():
    return {
        "Naive Bayes": MultinomialNB(alpha=1.0),  #default for laplace smooth
    }