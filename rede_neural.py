import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import os
import warnings
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tabulate import tabulate
import random

def carregar_dados(caminho_train, caminho_test):
    data_train = pd.read_csv(caminho_train, sep=';', encoding='utf-8')
    data_test = pd.read_csv(caminho_test, sep=';', encoding='utf-8')
    data = pd.concat([data_train, data_test], ignore_index=True)
    return data

def preprocessar_dados(data):
    data['Attrition'] = data['Attrition'].replace({'Left': 1, 'Stayed': 0}).astype(int)
    data_filtrado = data[['Gender', 'Work-Life Balance', 'Marital Status', 'Job Level', 'Remote Work', 'Attrition']]
    data_filtrado = pd.get_dummies(data_filtrado, drop_first=True)
    return data_filtrado

def dividir_dados(data_filtrado):
    X = data_filtrado.drop('Attrition', axis=1)
    y = data_filtrado['Attrition']
    return train_test_split(X, y, test_size=0.3, random_state=42)

def normalizar_dados(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_test_n = scaler.transform(X_test)
    return pd.DataFrame(X_train_n, columns=X_train.columns), pd.DataFrame(X_test_n, columns=X_test.columns)

def mostrar_balanceamento(y_train):
    balanceamento = y_train.value_counts(normalize=True).rename_axis('Classe').reset_index(name='Proporção')
    balanceamento['Proporção (%)'] = (balanceamento['Proporção'] * 100).round(2)
    balanceamento['Classe'] = balanceamento['Classe'].map({0: 'Stayed', 1: 'Left'})
    print('\nBalanceamento das Classes:\n')
    print(tabulate(balanceamento, headers='keys', tablefmt='grid'))

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def construir_modelo(input_dim):
    model = Sequential([
        Dense(400, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(200, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dense(50, activation='relu'),
        BatchNormalization(), 
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy', 'AUC'])
    return model

def treinar_modelo(model, X_train, y_train):
    print('\nTreinando o Modelo.\n')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.3, callbacks=[early_stopping])
    return history

def avaliar_modelo(model, X_test, y_test):
    loss, accuracy, auc = model.evaluate(X_test, y_test)
    print(f'\nAcurácia no teste: {accuracy:.2f}')
    return loss, accuracy, auc

def mostrar_resultados(y_test, y_pred_prob):
    y_pred = (y_pred_prob > 0.5).astype(int)
    resultados = pd.DataFrame({
        'Real': y_test.values,
        'Probabilidade': y_pred_prob.flatten().round(4),
        'Previsto': y_pred.flatten()
    })
    print('\nResultados das Previsões:\n')
    print(tabulate(resultados.head(5), headers='keys', tablefmt='grid'))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Left"]))
    return y_pred

def plotar_matriz_confusao(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stayed", "Left"])
    disp.plot(cmap='Blues')
    plt.title("Matriz de Confusão")
    plt.grid(False)
    plt.show()

def plotar_curva_roc(y_test, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test, y_pred_prob)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC')
    plt.legend()
    plt.grid()
    plt.show()

def plotar_metricas_treinamento(history):

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia por época')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Loss por época')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    warnings.filterwarnings('ignore')

    set_seed(42)
    ##
    data = carregar_dados('train-atualizada.csv', 'test-atualilzada.csv')
    data_filtrado = preprocessar_dados(data)
    print("\nPrévia dos dados após transformação da coluna 'Attrition':\n")
    print(data_filtrado)

    print("\nContagem de classes numéricas (0 = Stayed, 1 = Left):\n")
    print(data_filtrado['Attrition'].value_counts())


    X_train, X_test, y_train, y_test = dividir_dados(data_filtrado)
    
    mostrar_balanceamento(y_train)
    X_train_norm, X_test_norm = normalizar_dados(X_train, X_test)
    
    model = construir_modelo(X_train_norm.shape[1])
    history = treinar_modelo(model, X_train_norm, y_train)
    avaliar_modelo(model, X_test_norm, y_test)
    
    y_pred_prob = model.predict(X_test_norm)
    y_pred = mostrar_resultados(y_test, y_pred_prob)
    
    plotar_matriz_confusao(y_test, y_pred)
    plotar_curva_roc(y_test, y_pred_prob)
    plotar_metricas_treinamento(history)

if __name__ == "__main__":
    main()
