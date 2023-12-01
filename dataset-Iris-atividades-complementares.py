# Importação de bibliotecas
from sklearn.datasets import load_iris
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

# Carregamento do conjunto de dados Iris
iris = load_iris()
X, y = iris.data, iris.target

# Visualização 3D dos dados
fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=y, labels={'0': 'Sepal Length', '1': 'Sepal Width', '2': 'Petal Length'})
fig.show()

# Divisão dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Predição e Avaliação KNN
knn_predictions = knn_model.predict(X_test)
knn_metrics = classification_report(y_test, knn_predictions)

# Treinamento do modelo K-means
kmeans_model = KMeans(n_clusters=3, random_state=42)
kmeans_model.fit(X_train)

# Predição e Avaliação K-means (não há rótulos verdadeiros para avaliação)
kmeans_predictions = kmeans_model.predict(X_test)

# Análise de Desempenho
# - Métricas do KNN
print("Métricas do KNN:")
print(knn_metrics)

# - Métricas do K-means (sem rótulos verdadeiros para avaliação direta)
# - Pode-se usar análises adicionais dependendo do contexto
