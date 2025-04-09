import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data_train = pd.read_csv('train-atualizada.csv', sep=';', encoding='utf-8')
data_test = pd.read_csv('test-atualilzada.csv', sep=';', encoding='utf-8')

data = pd.concat([data_train, data_test], ignore_index=True)

print(data)

#converte coluna alvo para binária
data['Attrition'] = LabelEncoder().fit_transform(data['Attrition'])

feature_cols = data.select_dtypes(include=[np.number]).columns.drop('Attrition')

auc_resultados = {}

for coluna in feature_cols:
    try:
        #usar o valor da coluna como sacore
        auc = roc_auc_score(data['Attrition'], data[coluna])
        auc_resultados[coluna] = auc
    except Exception as e:
        print(f'Erro na coluna {coluna}: {e}')

#ordenar por auc decrescente
sorted_auc = sorted(auc_resultados.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)

#mostrar os melhores atributos
for feature, auc in sorted_auc:
    print(f'{feature}: AUC = {auc:.3f}')
    
#grafico
top_features = [f for f, _ in sorted_auc[:10]]

plt.figure(figsize=(10, 5), dpi=150)
cores = plt.cm.plasma(np.linspace(0, 1, len(top_features)))

for i, feature in enumerate(top_features):
    y_score = data[feature]
    fpr, tpr, _ = roc_curve(data['Attrition'], y_score)
    auc = roc_auc_score(data['Attrition'], y_score)
    plt.plot(fpr, tpr, label=f'{feature} (AUC = {auc:.2f})', color=cores[i], linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chute (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curvas ROC dos Top 10 Atributos com Maior AUC', fontsize=14)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()
