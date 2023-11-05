import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def extract_feature_opinion_pairs(reviews):
    """Extracts feature-opinion pairs from reviews.

    Args:
        reviews: A list of reviews.

    Returns:
        A list of feature-opinion pairs, where each pair is a tuple of (feature, opinion).
    """

    feature_opinion_pairs = []
    for review in reviews:
        # Identify feature candidates.
        feature_candidates = []
        for word in review.split():
            if word.isalpha() and len(word) > 2:
                feature_candidates.append(word)

        # Filter feature candidates using a dictionary of known features.
        feature_dictionary = pd.read_csv('features.csv')
        features = [feature for feature in feature_candidates if feature in feature_dictionary['feature'].tolist()]

        # Identify opinion words.
        sentiment_analyzer = SentimentIntensityAnalyzer()
        opinion_words = []
        for sentence in review.split('.'):
            for word in sentence.split():
                sentiment_score = sentiment_analyzer.polarity_scores(word)['compound']
                if sentiment_score > 0:
                    opinion_words.append(word)

        # Map opinion words to features.
        for feature in features:
            for opinion_word in opinion_words:
                if opinion_word in review and feature in review:
                    feature_opinion_pairs.append((feature, opinion_word))

    return feature_opinion_pairs


def main():
    # Load the reviews.
    reviews = pd.read_csv('reviews.csv')

    # Extract feature-opinion pairs.
    feature_opinion_pairs = extract_feature_opinion_pairs(reviews['review'].tolist())

    # Save the feature-opinion pairs to a file.
    with open('feature_opinion_pairs.csv', 'w') as f:
        for feature_opinion_pair in feature_opinion_pairs:
            f.write('{},{}\n'.format(feature_opinion_pair[0], feature_opinion_pair[1]))


if __name__ == '__main__':
    main()
