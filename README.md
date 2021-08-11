# Sentiment-Analysis-using-IMDB-movies-reviews
This is an NLP and Flask based application which involves predicting the sentiments of the sentences as positive or negative. The classifier is trained on a huge dataset of IMDB movies reviews.  The model is then hosted using Flask to be used by end users.

This project has text pre-processing done through NLTK and Regex and EDA for understanding the features and data well.
The text is then coverted into vectors using 2 techniques - **Countvectorize and TF-IDF**.
Two Machine Learning algorithms **(Naive Bayes and SVM)** are then used with combonitions of above 2 techniques and it is found that Naive Bayes with TF-IDF.

The model is then saved in a **Pickle file** and used in the **Flask Application** to host the website on localhost.




