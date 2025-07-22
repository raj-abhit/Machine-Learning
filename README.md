# Yelp-Reviews-sentiment-analysis-using-NLP

# 📘 Introduction
In the current digital landscape, online reviews are paramount in influencing consumer choices and shaping business reputations. The ability to understand and categorize these reviews offers invaluable insights for businesses, enabling them to quickly gauge public sentiment and pinpoint areas for enhancement. However, manually sorting through countless reviews to determine their overall sentiment is often an overwhelming and time-consuming endeavor.

This project addresses this challenge by developing an AI-powered sentiment classification system. This system categorizes Yelp reviews as either 1-star (negative) or 5-star (positive), based solely on their textual content. Rather than focusing on complex deep learning architectures, this project emphasizes the practical application of efficient Natural Language Processing (NLP) pipeline methods to streamline the classification process.

# 🎯 Objectives
The primary objectives of this project are:

Classify Yelp reviews into two distinct sentiment categories: 1-star (negative) or 5-star (positive), based exclusively on the textual content of the review.

Utilize and demonstrate the effectiveness of NLP pipeline methods for text classification.

Evaluate the model's performance using standard classification metrics to assess its accuracy and reliability in distinguishing between positive and negative reviews.

This system aims to empower businesses and analysts to quickly extract valuable sentiment insights from large volumes of Yelp reviews, thereby enabling more informed, data-driven decisions to enhance customer satisfaction and overall business performance.

# ⚙️ Technologies & Libraries

Python 3.x

Pandas: For data handling and manipulation.

Scikit-learn: For machine learning models, pipeline utilities, and evaluation metrics.

NLTK (Natural Language Toolkit) / SpaCy: For text preprocessing (tokenization, stopwords, etc.).

Joblib / Pickle: For saving and loading the trained model and pipeline.



# 📊 Model Evaluation
The model's performance is rigorously assessed using standard classification metrics, which may include:

Accuracy: Overall correctness of predictions.

Precision: The proportion of positive identifications that were actually correct.

Recall: The proportion of actual positives that were identified correctly.

F1-score: The harmonic mean of precision and recall.

Confusion Matrix: A table describing the performance of a classification model.
