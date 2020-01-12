# import statements

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ----------------Input Data and Pre-processing-----------------

# 1.Read the CSV file

df = pd.read_csv('spam.csv')

# 2.Drop the columns Unnamed:2, Unnamed:3, Unnamed:4

df = df.drop(df.columns[2:5], axis=1)

# 3. Rename the columns v1 as label and v2 as message

df.columns = ['label', 'message']

# 4. Map all ham labels to 0 and spam values to 1

labels = df['label'].unique()
labels_dict = dict(zip(labels, range(len(labels))))
df = df.replace({"label": labels_dict})

# 5. Assign Message column to X
X = df['message']

# 6. Assign label column to Y
Y = df['label']

# ---------------------------Feature Extraction----------------

# 7.Initialise the countvectorizer
cv = CountVectorizer()

# 8.Fit transform the data X in the vectorizer and store the result in X
X = cv.fit_transform(X)
X = X.toarray()
# 9.save your vectorizer in 'vector.pkl' file
pickle.dump(cv, open("vector.pkl", "wb"))

# ------------------------Classification---------------------

'''10. Split the dataset into training data and testing data with train_test_split function
Note: parameters test_size=0.33, random_state=42'''

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# 11. Initialise multimimial_naive_bayes classifier

clf = MultinomialNB()

# 12.Fit the training data with labels in Naive Bayes classifier 'clf'

clf.fit(X_train, y_train)

# 13. Store your classifier in 'NB_spam_model.pkl' file
joblib.dump(clf, 'NB_spam_model.pkl')
