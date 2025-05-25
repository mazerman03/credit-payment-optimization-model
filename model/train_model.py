import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_dataset():
    df = pd.read_csv("data/output/dataset_modelo.csv")
    df["strategyId"] = df["strategyId"].astype(int)
    return df

def entrenar_modelo():
    df = cargar_dataset()

    # Separar features y etiquetas
    feature_cols = [
        'horaCobro', 'diaSemanaCobro', 'dias_envio_cobro',
        'montoCobrar', 'montoExigible', 'monto_ratio_exigible_cobrar',
        'historial_exitos', 'historial_fallas', 'intentos', 'tipoEnvio'
    ]
    X = df[feature_cols]
    y = df['strategyId']

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Crear y entrenar modelo
    num_classes = y.nunique()
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Evaluación
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt='d')
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig("data/output/matriz_confusion.png")
    plt.close()
    print("📊 Matriz de confusión guardada en 'data/output/matriz_confusion.png'")

    # Guardar modelo
    joblib.dump(model, "data/output/modelo_entrenado.joblib")
    print("✅ Modelo guardado en 'data/output/modelo_entrenado.joblib'")

if __name__ == "__main__":
    entrenar_modelo()
