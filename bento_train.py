import bentoml

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
iris_data = load_iris()
X = iris_data.data[:, :4]
Y = iris_data.target

model.fit(X, Y)


bento_model = bentoml.sklearn.save_model('Kneighbors',model)
print(f'model saved: {bento_model}')