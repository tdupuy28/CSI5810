import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'D:\Y6S1\Info Retrival\Project 1\train.csv')
df = pd.DataFrame(data)

X = df.loc[:, 'Elevation':'Soil_Type40']

norm_X = X.copy().drop(columns=['Soil_Type7', 'Soil_Type15'])
for column in norm_X.columns:
    norm_X[column] = \
        (norm_X[column] - norm_X[column].min()) / (norm_X[column].max() - norm_X[column].min())
# norm_X.to_csv(path_or_buf='D:\Y6S1\Info Retrival\out.csv')

y = df.Cover_Type
types = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow",
         "Aspen", "Douglas-fir", "Krummholz"]

# X_train, X_test, y_train, y_test = train_test_split(norm_X, y, test_size=0.2, random_state=1)

folds = 10

ks = [1, 3, 5, 7]
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, norm_X, y, cv=folds)
    print("kNN (" + str(k) + " neighbors):")
    print("\tMean Accuracy: ", round(np.mean(cv_scores), 5))
    print("\tStandard Deviation: ", round(np.var(cv_scores), 5))
print()

# acc = [[0, 0, 0]]
depths = [4, 6, 10, 13] # range(1, 21)
for depth in depths:
    dtc = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=1)
    cv_scores = cross_val_score(dtc, norm_X, y, cv=folds)
    print("Decision Tree (" + str(depth) + " max depth):")
    # acc.append([depth, np.mean(cv_scores), np.var(cv_scores)])
    print("\tMean Accuracy: ", round(np.mean(cv_scores), 5))
    print("\tStandard Deviation: ", round(np.var(cv_scores), 5))
print()

# plt.plot(acc)
# plt.grid(linestyle='--')
# plt.show()

# acc = [[0, 0]]
estimators = [4, 7, 10] # range(1, 11)
depths = [4, 6, 10, 13]
for estimator in estimators:
    print("Random Forest (" + str(estimator) + " estimators):")
    for depth in depths:
        rf = RandomForestClassifier(n_estimators=estimator, max_depth=depth, random_state=1)
        cv_scores = cross_val_score(rf, norm_X, y, cv=folds)
        print("\t" + str(depth), "max depth:")
        # acc.append([np.mean(cv_scores), np.var(cv_scores)])
        print("\t\tMean Accuracy:", round(np.mean(cv_scores), 5))
        print("\t\tStandard Deviation:", round(np.var(cv_scores), 5))

# plt.plot(acc)
# plt.xticks(np.arange(0, 41, 1))
# plt.grid(linestyle='--')
# plt.show()

rf = RandomForestClassifier(n_estimators=50, max_depth=30, random_state=1)
cv_scores = cross_val_score(rf, norm_X, y, cv=folds)
print("Random Forest (50 estimators, 30 max depth):")
print("\tMean Accuracy:", round(np.mean(cv_scores), 5))
print("\tStandard Deviation:", round(np.var(cv_scores), 5))
