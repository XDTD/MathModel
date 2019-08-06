from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
sns.set(color_codes=True)

column_names = ["sex", "length", "diameter", "height", "whole weight",
                "shucked weight", "viscera weight", "shell weight", "rings"]
data = pd.read_csv("Data/abalone.data", names=column_names)
print("Number of samples: %d" % len(data))
print(data.head())

data.sex = data.sex.map({'M': 1, 'I': 0, 'F': 2})
print(data.head())
df = data
sns.set_style("whitegrid")
plot = sns.pairplot(data, hue = "sex", height=2.5)
plot = sns.boxplot(x="sex", y="rings",  data=df, palette="PRGn")
sns.despine(offset=10, trim=True)
plt.figure(figsize=(10,10))
correlation = df.corr()
sns.heatmap(correlation, square=True,annot=True,vmin=0,vmax=1,cmap='jet')
plt.title('Correlation Matrix')
plt.show()
X = data.drop(["rings"],axis = 1)
y = data["rings"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf_rbf = svm.SVC(kernel='rbf', gamma='auto')
clf_rbf.fit(X_train,y_train)
score_rbf = clf_rbf.score(X_test,y_test)
print("The score of rbf is : %f"%score_rbf)

# kernel = 'linear'
clf_linear = svm.SVC(kernel='linear', gamma='auto')
clf_linear.fit(X_train,y_train)
score_linear = clf_linear.score(X_test,y_test)
print("The score of linear is : %f"%score_linear)

# kernel = 'poly'
clf_poly = svm.SVC(kernel='poly', gamma='auto')
clf_poly.fit(X_train,y_train)
score_poly = clf_poly.score(X_test,y_test)
print("The score of poly is : %f"%score_poly)

