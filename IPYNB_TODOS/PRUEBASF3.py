import pandas as pd

airbnb = pd.read_csv('https://raw.githubusercontent.com/HarryVargas96/UdeCataluna/main/data/aibnb_limpio_sinnas_sincat.csv')



#dividir el dataset airbnb en X y y siendo y el campo de price
X = airbnb.drop(['price'], axis=1)
y = airbnb[['price']]
#ahora se divide en entrenamiento y test

#entrenamiento y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(random_state= 5)
tree.fit(X_train,y_train)

'''
# Respuesta
from sklearn.metrics import mean_squared_error
# Lista vacía para almacenar los errores
mse_tree = []
# Lista con los valores de profundidad, valores de 1 a 40
valores_profundidad = range(1,40,1)

for profundidad in valores_profundidad:
  tree =DecisionTreeRegressor(max_depth=profundidad, random_state=5)  
  tree.fit(X_train,y_train)
  y_pred = tree.predict(X_test)
  #calcular el mse_tree      
  error = mean_squared_error(y_test, y_pred)
  mse_tree.append()


#transformar la lista en un dataframe
mse_tree = pd.DataFrame(mse_tree, valores_profundidad, columns=['MSE'])

#ordenar el dataframe de menor a mayor de los mse
mse_tree = mse_tree.sort_values(by='MSE')
#imprimir el dataframe head(3)
mse_tree.head(3)


'''



from sklearn.metrics import mean_squared_error
# Lista vacía para almacenar los errores
mse_tree = []
# Lista con los valores de profundidad, valores de 1 a 40
valores_profundidad = range(7,7,1)


tree =DecisionTreeRegressor(max_depth=7, random_state=5)  
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
#calcular el mse_tree      
error = mean_squared_error(y_test, y_pred)
print(error)


y_predict = tree.predict(X_test)
y_pred2 = tree.predict(X_train)


#calcular el root mean squared error
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test, y_predict)
rmse = mean_squared_error(y_test, y_predict)**0.5

#calcular el error absolute error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_predict)

#calcular coeficiente r2 para el modelo
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)


print('El MSE es:' ,mse )
print('Raíz del error cuadrático medio (RMSE):' ,rmse )
print('Error absoluto medio (MAE):' ,mae )
print('Coeficiente de determinación R2:' ,r2 )

#determinar cuales son las variables mas importantes del modelo
importances = tree.feature_importances_
indices = pd.DataFrame(X_train.columns)

print('Las variables mas importantes son:', indices[importances>0.01])



#haga un grafico del arbol de decision que se vea mas claro
import matplotlib.pyplot as plt
import six
import sys
sys.modules['sklearn.externals.six'] = six
from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus
dot_data = StringIO()
export_graphviz(tree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
































