#!/usr/bin/env python
# coding: utf-8

# # Cars MPG dataset
# ### Primero importamos el dataset

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
data= pd.read_csv("cars.csv", sep = ";")
data= data.drop(0)
data.head()


# Buscamos a ver si hay valores en blanco/null

# In[2]:


data.isnull().sum()


# Recopilar la variable objetivo y separar las variables de clasificacion

# In[3]:


columnas_a_convertir = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model']
for columna in columnas_a_convertir:
    data[columna] = pd.to_numeric(data[columna], errors='coerce')

sns.pairplot(data)
plt.show()


# In[4]:


#Variable objetivo
y = data['Cylinders']
#Variables de clasificacion
X = data.drop(['Car','Cylinders'], axis = 1)


# In[5]:


#df de variables
X.head()


# In[6]:


#Tenemos la variable Origin, vamos a representarla como categorica
valores_unicos = data['Origin'].unique()
valores_unicos


# In[7]:


data.loc[data.loc[:,"Origin"] == "US", "Origin"] = 1 #Seleccion de la class == "US"
data.loc[data.loc[:,"Origin"] == "Europe", "Origin"] = 2 #Seleccion de la class == "Europe"
data.loc[data.loc[:,"Origin"] == "Japan", "Origin"] = 3#Seleccion de la class == "Japan"
#Variables de clasificacion
X = data.drop(['Car','Cylinders'], axis = 1)
X.head()


# In[8]:


y #Variable a predecir, seria nuestro classTrain


# ## Construimos el modelo SVM

# In[9]:


from sklearn.metrics import accuracy_score, classification_report
SVMachine = SVC(kernel='linear', C=1.0) # Utilizaremos un clasificador de SVM lineal
#El parametro C es un hiperparámetro que controla la penalización por error de clasificación. Mientras mas bajo,
# mas errores se van a permitir, por defecto utilizaremos C = 1


# In[10]:


SVMachine.fit(X,y) # Entrenar el modelo con los datos de entrenamiento


# Ahora vamos a probar con la data de prueba

# In[11]:


X_test= pd.read_csv("test_cars.csv", sep = ";")

X_test.loc[X_test.loc[:,"Origin"] == "US", "Origin"] = 1 #Seleccion de la class == "US"
X_test.loc[X_test.loc[:,"Origin"] == "Europe", "Origin"] = 2 #Seleccion de la class == "Europe"
X_test.loc[X_test.loc[:,"Origin"] == "Japan", "Origin"] = 3#Seleccion de la class == "Japan"
y_true = X_test["Cylinders"] #Guardamos los resultados para ver que tan exacto es nuestro modelo
X_test = X_test.drop(['Car','Cylinders'], axis = 1)
X_test


# Ahora vamos a predecir con nuestra data de prueba...

# In[12]:


y_pred = SVMachine.predict(X_test)
print(y_pred)


# In[13]:


y_pred.tolist()
y_true.tolist()
accuracy = accuracy_score(y_true, y_pred)
print("Obtuvimos un porcentaje de precision de", accuracy * 100, "%")
print("Classification Report:\n", classification_report(y_true, y_pred))


# Vamos a ver un ejemplo de los resultados obtenidos...
# # Grafico de valores reales vs valores predichos

# In[14]:


plt.scatter(X_test["MPG"], y_true, color='black', label='Real')
plt.scatter(X_test["MPG"], y_pred, color='red', marker='x', label='Predicho')
plt.title('Valores Reales vs Valores Predichos')
plt.xlabel('MPG')
plt.ylabel('Cylinders')
plt.legend()
plt.show()


# In[15]:


# Creamos tres subgráficos uno al lado del otro para ver otras variables
fig, axs = plt.subplots(1, 3, figsize=(10, 4))

axs[0].scatter(X_test["Displacement"], y_true, color='black', label='Real')
axs[0].scatter(X_test["Displacement"], y_pred, color='red', marker='x', label='Predicho')
axs[0].set_title('Real vs Pred')
axs[0].set_xlabel('Displacement')
axs[0].set_ylabel('Cylinders')


axs[1].scatter(X_test["Weight"], y_true, color='black', label='Real')
axs[1].scatter(X_test["Weight"], y_pred, color='red', marker='x', label='Predicho')
axs[1].set_title('Real vs Pred')
axs[1].set_xlabel("Weight")

axs[2].scatter(X_test["Horsepower"], y_true, color='black', label='Real')
axs[2].scatter(X_test["Horsepower"], y_pred, color='red', marker='x', label='Predicho')
axs[2].set_title('Real vs Pred')
axs[2].set_xlabel('Horsepower')

# Evitar el grafico superpuesto
plt.legend()
plt.tight_layout()
plt.show()


# # Utilizando el algoritmo KNN

# In[16]:


from sklearn.neighbors import KNeighborsClassifier

#Generamos el modelo con K = 3
knn_classifier = KNeighborsClassifier(n_neighbors=3)

knn_classifier.fit(X, y)

# Calculamos la prediccion nuevamente
y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_true, y_pred)
print("Obtuvimos un porcentaje de precision de", accuracy * 100, "%")
print("Classification Report:\n", classification_report(y_true, y_pred))


# In[17]:


plt.scatter(X_test["MPG"], y_true, color='black', label='Real')
plt.scatter(X_test["MPG"], y_pred, color='red', marker='x', label='Predicho')
plt.title('Valores Reales vs Valores Predichos (KNN)')
plt.xlabel('MPG')
plt.ylabel('Cylinders')
plt.legend()
plt.show()


# ## Construimos el modelo Kmeans, algoritmo no supervisado
# Ahora utilizaremos el Kmeans para el mismo proposito

# In[18]:


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

#Seleccionamos origin 1, vamos a utilizar los datos que cumplen con esta caracteristica
#Son vehiculos americanos
data = pd.read_csv("auto-mpg.csv", sep = ',',na_values="?")
data = data[data.origin == 1]
label_cars = data["car name"]
data = data.drop(["origin", "car name"], axis = 1)


# In[ ]:


data.groupby("cylinders").size() # Los autos americanos solo existen en 4, 6 o 8 cylindros


# In[ ]:


#Matriz de correlacion
import seaborn as sns
corr = data.corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20,220, n=200),
    square=True

)


# In[ ]:


#Vamos a normalizar la data para aplicar PCA y Kmeans
from sklearn import preprocessing
x = data.values
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
X_norm = pd.DataFrame(x_scaled)
print(X_norm.head())
print(data.head())


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (10,6))

ax1.set_title("Antes de la transformacion")
ax2.set_title("Despues de la transformacion")
sns.kdeplot(data, ax=ax1)
sns.kdeplot(X_norm, ax=ax2)
plt.show()
#Observams la gran diferencia entre no normalizar y si normalizar la data


# In[ ]:


X_norm.info;


# In[ ]:


#genera la reduccion de dimensionalidad pca para generar el modelo Kmeans
X_norm = X_norm.dropna()
from sklearn.decomposition import PCA
#Vamos a reducir a dos componentes
pca = PCA(n_components = 2)
reduced = pd.DataFrame(pca.fit_transform(X_norm))
reduced

import matplotlib.pyplot as plt

# Grafica los puntos de datos coloreados según sus asignaciones de cluster
plt.scatter(reduced.iloc[:, 0], reduced.iloc[:, 1], cmap='viridis', edgecolors='k')

plt.title('Reduccion de dimensionalidad')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend()
plt.show()


# In[ ]:


from sklearn.cluster import KMeans
#numero de clusters
kmeans = KMeans(n_clusters = 3)
#genera el modelo
kmeans = kmeans.fit(reduced)

labels = kmeans.labels_
centers = kmeans.cluster_centers_


# In[ ]:


import matplotlib.pyplot as plt

# Grafica los puntos de datos coloreados según sus asignaciones de cluster
plt.scatter(reduced.iloc[:, 0], reduced.iloc[:, 1], c=labels, cmap='viridis', edgecolors='k')

# Grafica los centros de los clusters
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='o', s=300, label='Centroides')


plt.title('Resultado de K-Means')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend()
plt.show()


# # K-means y selector de features utilizando River

# In[ ]:


from river import cluster
from river import stream
from river import feature_selection

from pprint import pprint
from river import stats
from river import stream
from sklearn import datasets

#Elegimos features y target
X = data.drop(["cylinders"], axis = 1)
y = data["cylinders"]


# In[ ]:


import pandas as pd
import numpy as np
#Convertimos a numpy darray para poder utilizar los datos
X = X.values
y = y.values


# In[ ]:


#Vamos a elegir los features mas relevantes
selector = feature_selection.SelectKBest(
    similarity=stats.PearsonCorr(),
    k=2
)
selector


# In[ ]:


for xi, yi, in stream.iter_array(X, y):
    selector = selector.learn_one(xi, yi)
pprint(selector.leaderboard) #Elegiremos los features 1 y 3


# In[ ]:


X = data.drop(["cylinders"], axis = 1)
X = X[['displacement', 'weight']]
#Convertimos a numpy darray para poder utilizar los datos
#Vamos a normalizar la data
from sklearn import preprocessing
x = X.values
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
X_norm = pd.DataFrame(x_scaled)
print(X_norm.head())
print(X.head())


# In[ ]:


X = X_norm.values 
X = X.tolist()


# In[ ]:


k_means = cluster.KMeans(n_clusters=2, halflife=0.0001, sigma=3, seed=42)
for i, (x, _) in enumerate(stream.iter_array(X)):
    k_means = k_means.learn_one(x)
    print(f'A este elemento {X[i]} se asigna al cluster {k_means.predict_one(x)}')


# In[ ]:


X


# # Hoeffding Tree (HT)

# In[ ]:


import matplotlib.pyplot as plt
import datetime as dt

from river import datasets
from river import evaluate
from river import metrics
from river import tree #importaremos HT Clasificador
from river import preprocessing
from river.datasets import synth

dataset = datasets.Bananas()
dataset


# In[ ]:


#generamos modelo

model = tree.HoeffdingTreeClassifier(grace_period=50)

#introducimos datos el modelo
for x, y in dataset:
    model.learn_one(x, y)


# In[ ]:


model


# In[ ]:


model.summary


# In[ ]:


model.to_dataframe().iloc[:10, :5]


# ### Predicción de instancias individuales

# In[ ]:


x, y = next(iter(dataset))  # Vamos a tomar la siguiente observación del dataset 
x, y


# In[ ]:


for x, y in dataset:
    print("Variable", x,"target", y, "prediccion", model.debug_one(x))

