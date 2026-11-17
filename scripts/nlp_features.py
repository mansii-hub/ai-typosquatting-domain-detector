# from sklearn.feature_extraction.text import TfidfVectorizer

# def extract_tfidf_features(domains):
#     vectorizer = TfidfVectorizer(
#         analyzer="char",
#         ngram_range=(2, 4),
#         max_features=2000,  # can increase for better accuracy
#         min_df=2
#     )
#     tfidf_features = vectorizer.fit_transform(domains)
#     return tfidf_features, vectorizer


from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_features(texts):

    vectorizer = TfidfVectorizer(
        ngram_range=(1,3),
        max_features=5000,
        stop_words='english'
    )

    X = vectorizer.fit_transform(texts)

    return X, vectorizer
