from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import seaborn as sns

data = pd.read_csv("data.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(data.head())

col = data.columns
# print(col)

redundant_feature_list = ['Unnamed: 32', 'id', 'diagnosis']

X = data.drop(redundant_feature_list, axis=1)
y = data['diagnosis']

y = y.replace("B", 0)
y = y.replace("M", 1)

# print(X.head())

# print(X.describe())

# Correlation map
f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()

drop_list = ['perimeter_mean', 'radius_mean', 'compactness_mean', 'concave points_mean', 'radius_se', 'perimeter_se',
             'radius_worst', 'perimeter_worst', 'compactness_worst', 'concave points_worst', 'compactness_se',
             'concave points_se', 'texture_worst', 'area_worst']

X = X.drop(drop_list, axis=1)

# print(X.head())

# Data normalization
standardizer = StandardScaler()
X = standardizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

models = {'Logistic Regression': LogisticRegression(), 'Support Vector Machines': LinearSVC(),
          'Decision Trees': DecisionTreeClassifier(), 'Random Forest': RandomForestClassifier(),
          'Naive Bayes': GaussianNB(), 'K-Nearest Neighbor': KNeighborsClassifier()}

accuracy, precision, recall, f_score = {}, {}, {}, {}

for key in models.keys():
    # Fit the classifier model
    models[key].fit(X_train, y_train)

    # Classification
    classification = models[key].predict(X_test)

    # Calculate Accuracy, Precision, Recall and F Score Metrics
    accuracy[key] = accuracy_score(classification, y_test)
    precision[key] = precision_score(classification, y_test)
    recall[key] = recall_score(classification, y_test)
    f_score[key] = f1_score(classification, y_test)

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall', "F Score"])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()
df_model["F Score"] = f_score.values()

print(df_model)

ax = df_model.plot.bar(rot=90)
ax.legend(ncol=len(models.keys()), bbox_to_anchor=(0, 1), loc='lower left', prop={'size': 12})
plt.tight_layout()

plt.show()
