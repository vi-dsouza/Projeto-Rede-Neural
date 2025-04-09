
# 🤖 Rede Neural

Esse projeto contempla o pré-processamento, treinamento, validação de métricas e visualizações de uma rede neural com keras para prever a rotatividade de funcionários.

`carregar_dados(caminho_tran, caminho_test)`

Carrega os arquivos csv de treino e teste. 

**Retorno:** Ela retorna os dois conjuntos de dados concatenados.

-------

`processar_dados(data)`

Pré-processa os dados, convertendo a variável alvo em numérica e aplica one-hot encoding.

**Retorno:** Os dados com as colunas relevantes e codificadas.

-------

`dividir_dados(data filtrado)`

Faz a divisão dos dados em treino e teste.

**Retorno:** Tupla(X_train, X_test, y_train, y_test)

-------

`normalizar_dados(X_train, X_test)` 

Normaliza os dados com MinMaxScaler.

**Retorno:** Os dados normalizados.

-------

`mostrar_balanceamento(y_train)`

Mostra a distribuição das classes no conjunto de treino.

-------

`set_seed(42)`

Define uma semente fixa para reprodutibilidade.
Valor padrão do seed **42**.

-------

`construir_modelo(input_dim)`

Constrói e compila a rede neural com keras.

input_dim (int): Número de features de entrada.

**Retorno:** Modelo Sequential compilado.

-------

`treinar modelo(model, X_train, y_train)`

Treina o modelo com validação e early stopping.

**Parâmetros:**

	model: Modelo keras
	X_train, y_train: Dados de treino.
**Retorno:** Objeto history do keras.

**Early Stopping:** técnica para evitar overfiting durante o trinamento. Para de treinar o modelo assim que ele parar de melhorar na validação. 
Se a métrica não melhora por X épocas seguidas (patience), ele interrompe o treinamento. Monitora uma métrica (como val_loss).

Patience = 5

Treinamento: 70 / 30

-------

`avaliar_modelo(model, X_test, y_test)`

Avalia o modelo no conjunto teste.

**Parâmetros:**

	Model: Modelo treinado.
	X_test, y_test: Dados de teste.

**Retorno:** (loss, accuracy, auc)


-------

`mostrar_resultados(y_test, y_pred_prob)`

Exibe os resultados de previsão, incluindo relatório de classificação.

**Prametros:**

	y_test: Valores reais.
	y_pred_prob: Probabilidades previstas.

**Retorno:** Valores de classes previstas.

-------

`plotar_matriz_confusão(y_test, y_pred)`

Plota a matriz de confusão.

**Parâmetros:**

	y_test, y_pred: Valores reais e previstos.

-------

`plotar_curva_roc(y_test, y_pred_prob)`

Plota a curva ROC e calcula AUC.

**Parâmetros:**

	y_test, y_pred_prob: Valores reais e probabilidade prevista.

-------

`plotar_metricas_treinamento(history)`

Plota o gráfico da acurácia e loss por época.

* Treino x Validação - Curvas próximas = bom poder de generalização


## ⚙️ Função Principal

`main()` 

Executa o processo de carregar, preparar, treinar, avaliar e visualizar os resultados do modelo.


-------

## 📈 Visualizar atributos com ROC

`atributos_roc.py`

Plota o gráfico de comparação dos melhores atributos.