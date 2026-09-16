import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Carrega os dados gerados pelo Libras.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(SCRIPT_DIR, "dataset_libras.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError("O arquivo 'dataset_libras.csv' não foi encontrado. Execute o Libras.py primeiro!")

df = pd.read_csv(csv_path)

# 2. Separa os recursos (coordenadas) e o rótulo (letra)
X = df.drop(columns=['label'])
y = df['label']

# 3. Verifica se é possível usar stratify (requer no mínimo 2 amostras por classe)
menor_classe = y.value_counts().min()

if menor_classe < 2:
    print(" Algumas letras possuem apenas 1 amostra. Ajustando a divisão de treino/teste...")
    # Divide sem a restrição de stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    # Divide mantendo a proporção exata por letra
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

print("Treinando o modelo de Inteligência Artificial (Random Forest)...")

# 4. Inicializa e treina o classificador
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Avalia a precisão do modelo
y_pred = model.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)

print("\n")
print(" TREINAMENTO CONCLUÍDO COM SUCESSO!")
print(f"Acurácia do modelo: {acuracia * 100:.2f}%")
print("\n")
print("\nRelatório de desempenho por letra:")
print(classification_report(y_test, y_pred, zero_division=0))

# 6. Salva o modelo treinado
modelo_path = os.path.join(SCRIPT_DIR, "modelo_libras.pkl")
joblib.dump(model, modelo_path)

print(f"\nModelo salvo com sucesso em: {modelo_path}")