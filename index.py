# Importação das bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, RMSprop

dados = pd.read_csv('./DadosTreino_Cardiopatas.csv' , delimiter=';')

dados['2nd_AtaqueCoracao'] = dados['2nd_AtaqueCoracao'].map({'Sim': 1, 'Nao': 0})

X = dados.drop(columns=['2nd_AtaqueCoracao'])
y = dados['2nd_AtaqueCoracao']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def criar_modelo(neuronios, func_ativacao, algoritmo):

    model = Sequential()
    model.add(Dense(neuronios, input_dim=X_train_scaled.shape[1], activation=func_ativacao))
    model.add(Dense(1, activation='sigmoid'))  
    model.compile(optimizer=algoritmo, loss='binary_crossentropy', metrics=['accuracy'])
    return model

arquiteturas = [
    (5, 'relu', Adam()),
    (5, 'relu', RMSprop()),
    (5, 'sigmoid', Adam()),
    (5, 'sigmoid', RMSprop()),
    (9, 'relu', Adam()),
    (9, 'relu', RMSprop()),
    (9, 'sigmoid', Adam()),
    (9, 'sigmoid', RMSprop())
]

resultados = []
for neuronios, func_ativacao, algoritmo in arquiteturas:
    print(f"Treinando modelo: {neuronios} neurônios, {func_ativacao} ativação, {algoritmo._name} algoritmo...")
    modelo = criar_modelo(neuronios, func_ativacao, algoritmo)
    modelo.fit(X_train_scaled, y_train, epochs=50, batch_size=10, verbose=0)  # Treinamento silencioso
    y_pred = (modelo.predict(X_test_scaled) > 0.5).astype("int32")  # Previsões binárias
    acc = accuracy_score(y_test, y_pred)
    matriz_confusao = confusion_matrix(y_test, y_pred)
    resultados.append({
        'neuronios': neuronios,
        'func_ativacao': func_ativacao,
        'algoritmo': algoritmo._name,
        'acuracia': acc,
        'matriz_confusao': matriz_confusao
    })

for resultado in resultados:
    print(f"Neurônios: {resultado['neuronios']}, Função de Ativação: {resultado['func_ativacao']}, "
          f"Algoritmo: {resultado['algoritmo']}, Acurácia: {resultado['acuracia']:.2f}")
    print(f"Matriz de Confusão:\n{resultado['matriz_confusao']}\n")
