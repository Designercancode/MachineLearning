import tensorflow
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data.txt")
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(data["buying"])
maint = le.fit_transform(data["maint"])
door = le.fit_transform(data["door"])
person = le.fit_transform(data["person"])
lug_boot = le.fit_transform(data["lug_boot"])
safety = le.fit_transform(data["safety"])
cl = le.fit_transform(data["class"])

predict = "class"

x = list(zip(buying, maint, door, person, lug_boot, safety))
y = list(cl)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(x_test)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", y_test[x])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N:", n)