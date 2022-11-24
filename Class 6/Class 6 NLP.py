import nltk
nltk.download('stopwords')
from os import getcwd
import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples 
from utils import process_tweet, build_freqs


nltk.download('twitter_samples')
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
print(all_positive_tweets[:5])

all_negative_tweets = twitter_samples.strings('negative_tweets.json')
print(all_negative_tweets[:5])

##Dividiendo nuestros datos 
#train test splot 20%
test_pos = all_positive_tweets[4000:]
print(test_pos[:5])


train_pos = all_positive_tweets[:4000]
print(train_pos[:5])

test_neg = all_negative_tweets[4000:]
print(test_neg[:5])

train_neg = all_negative_tweets[:4000]
print(train_neg[:5])


##Set train y test juntos
train_x = train_pos + train_neg
test_x = test_pos + test_neg

###Etiquetas 
##Recodar que:
  ##1. EL conjunto de training quedó con 8000 datos
  ##2. El conjunto de testing queda con 2000 datos len(filas,columnas) y haga la variable en filas
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis = 0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis = 0)

##Creando el diccionario con el vocabulario
freqs = build_freqs(train_x, train_y)
##Chequeando la salida
print("type of freqs = " + str(type(freqs)))
print("len of freqs = " + str(len(freqs.keys())))

print('example with process_tweet function :',  train_x[0])
print('example  aftern with process_tweet function :',  process_tweet(train_x[0]))

#Función sigmoide
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)

def sigmoid(z):
  #la salida de la funcion es h que es la funcion de la regesion al pasar por la funcion de activacion
  h = 1 / (1 + np.exp(-z))
  return h



if (sigmoid (0) == 0.5):
  print('Good')
else:
    print('oh no')

if (sigmoid (4.92) == 0.992):
  print('Good')
else:
  print('Oh no')
  
  
def gradientDescent(x, y, theta, alpha, num_iters):
  ###Nota, el gradiente recuerda que trata de optimizar los parámetros de theta
    ##Cuyo fin es optimizar la función del error para obtener un mejor accuracy 
    ##sobre nuestro modelo
    ##Vimos que necesitamos unos parámetros iniciales para que el pueda iterar
    ##A partir de las derivadas parciales y parámetros como:
    ##Theta, número de iteraciones (ciclo iterativo), ratio de aprendizaje (alpha)
    ## Y por supuesto que obvio que yes que los argumentos de entrada X y Y.
    ##x.shape nos da el número de filas o sea las observaciones
    #Recuerda que cuando usamos shape nos indica (filas[0], columnas[1])
    m = x.shape[0]     
    for i in range(0, num_iters):
        
        # Z es el producto punto de X con theta
        z = np.dot(x, theta)
        
        # La función de activación se hace sobre Z
        ##Recuerda que esta función tiene varias ventajas:
        ##1. Distorciona no linealmente un plano lineal de tal forma que moldee 
        ## y modifique nuestro plano para obtener un rango de probabilidades más
        ##optimo
        ##2. minimiza el coste computacional
        ##limita mejor las probabilidades como era nuestro problema binario
        ##debido a que necesitamos salidas 1 o 0. 
        #3. Si deseas saber más de estas bellezas, lee :v
        
        h = sigmoid(z)
        
        # calcula la función de coste
        ##Recuerda que para entender esta función la dividimos en A y B
        ##Qué pasa con el coste para etiquetas postivas A
        ##Qué pasa con el coste para etiquetas negativas B
        J = -1./m * (np.dot(y.transpose(), np.log(h)) + np.dot((1 - y).transpose(),np.log(1 - h)))                                                    

        # Actualiza los valores de theta, recuerda que para el ejemplo del gradiente
        ##cuando decíamos que éramos nosotros con los ojos vendados, el segundo paso
        ##es actualizas la posición
        ##Posición igual a nueva actualizaación de parámetro theta
        theta = theta - (alpha/m) * np.dot(x.transpose(),(h - y))
        
    ### END CODE HERE ###
    J = float(J)
    return J, theta

###Extraer nuestras características
    
def extract_features(tweet, freqs):
    ### tweet una lista de palabras por cada tweet
    ##freqs el diccionario de frecuencias tupla (palabra, etiqueta)
    
    
    word_1 = process_tweet(tweet)
    x = np.zeros((1,3)) ###creación del arreglo para las caracte
    x[0,0] = 1 ##fijar el bias
    
    for word in word_1:
        x[0,1] += freqs.get((word, 1.0), 0)
        x[0,2] += freqs.get((word, 0.0), 0)
        
        
    assert(x.shape == (1, 3))
    return x

# la función assert de haber sido por ti quizña tenga errores, es normal somos humanos 
 # y es común tener errores
# assert te permite expresar una condición que ha de ser cierta siempre
#ya que de no serlo se interrumpirá el programa. Podríamos añadir un assert!!
    
##veriquemos la función
ej1 = extract_features(train_x[0], freqs)
print(ej1)

ej2 = extract_features('blob bbof', freqs)
print(ej2)

##Obtener las características sobre conjunto de X
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i,:] = extract_features(train_x[i], freqs)
    
###Etiquetas del entrenmiento de X
Y = train_y

###Parámetros del gradiente son: x,y,theta,alpha,num_iteraciones
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")
print(f"Weights is, {theta}.") 

def predict_tweet(tweet, freqs, theta):
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    
    return y_pred


for tweet in ['Im happy today','pepa pig quiere morir','im feeeling blue', 'estoy muy feliz']:
    print( '%s -> %f' % (tweet, predict_tweet(tweet, freqs, theta)))
    # s de cadena y f de flotante
    
my_tweet = 'We are learning NLP :)'
predict_tweet(my_tweet, freqs, theta)


def test_logistic_regression(test_x, test_y, freqs, theta):
    y_hat = []
    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)
        if y_pred > 0.5:
            y_hat.append(1)
            
        else:
            y_hat.append(0)
            
    accuracy = (y_hat == np.squeeze(test_y)).sum() / len(test_x)
    
    return accuracy

tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"logistic regression model's accuracy = {tmp_accuracy: .4f}")

print('Label predicted Tweets: \n')
for x, y in zip (test_x, test_y):
    y_hat = predict_tweet(x, freqs, theta)
    
    if np.abs(y - (y_hat > 0.5)) > 0:
        print('the tweet is: ', x)
        print('the tweet processed is: ', process_tweet(x))
        print('%d\t%0.8f\t%s' % (y, y_hat, ' '.join(process_tweet(x)).encode('ascii', 'ignore')))
        