# import libraries
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

warnings.filterwarnings('ignore')

# read the data set
data = pd.read_csv('credit_card.csv')

# no of rows and columns
print('Total rows and columns\n\n', data.shape, '\n')

# Dependent and independent variable
X = data.iloc[:, 1:30]

y = data['Class']


# Determine number of fraud cases and valid cases in DataSet
Fraud = data[y == 1]
Valid = data[y == 0]

print('Fraud Cases: ', len(Fraud))
print('Valid Cases: ', len(Valid))
print()

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build the model
clf = LogisticRegression()

# Train the classifier
clf.fit(X_train, y_train)

# test the model
y_predict = clf.predict(X_test)

# Accuracy score
a = (metrics.accuracy_score(y_test, y_predict))
print('Accuracy score:', round(a, 5))
print()

# print the actual and predicted labels
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
print(df1.head(20))

# Graphical representation of accuracy
plt.plot(y_test, y_predict, "c*-")
plt.show()
