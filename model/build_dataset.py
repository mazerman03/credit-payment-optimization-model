import pandas as pd
# Cargar archivos base
def cargar_datos():
    transacciones = pd.read_csv("data/ListaCobroDetalle2025.csv")
    emisora = pd.read_csv("data/ListaCobroEmisora.csv")
    cat_emisora = pd.read_csv("data/CatEmisora.csv")
    lista_cobro = pd.read_csv("data/ListaCobro.csv")

    # Convertir fechas
    transacciones['fechaCobroBanco'] = pd.to_datetime(transacciones['fechaCobroBanco'], dayfirst=True, errors='coerce')
    lista_cobro['fechaEnvioCobro'] = pd.to_datetime(lista_cobro['fechaEnvioCobro'], dayfirst=True, errors='coerce')
    lista_cobro['fechaCreacionLista'] = pd.to_datetime(lista_cobro['fechaCreacionLista'], dayfirst=True, errors='coerce')

    return transacciones, emisora, cat_emisora, lista_cobro

# Agregar features útiles
def construir_dataset():
    transacciones, emisora, cat_emisora, lista_cobro = cargar_datos()

    # Asociar transacción con estrategia (idEmisora)
    transacciones = transacciones.merge(emisora, on="idListaCobro", how="left")
    transacciones = transacciones.merge(cat_emisora, on="idEmisora", how="left")
    transacciones = transacciones.merge(lista_cobro, on="idListaCobro", how="left", suffixes=("", "_lista"))

    # Renombrar idEmisora como strategyId
    transacciones = transacciones.rename(columns={"idEmisora": "strategyId"})

    # Features adicionales
    transacciones['horaCobro'] = transacciones['fechaCobroBanco'].dt.hour
    transacciones['diaSemanaCobro'] = transacciones['fechaCobroBanco'].dt.dayofweek
    transacciones['dias_envio_cobro'] = (transacciones['fechaCobroBanco'] - transacciones['fechaEnvioCobro']).dt.days
    transacciones['monto_ratio_exigible_cobrar'] = transacciones['montoCobrar'] / (transacciones['montoExigible'] + 1e-6)

    # Agregar historial de éxito por crédito
    historial = transacciones.groupby('idCredito').agg(
        historial_exitos=('idRespuestaBanco', lambda x: (x == '00').mean()),
        historial_fallas=('idRespuestaBanco', lambda x: (x != '00').mean()),
        intentos=('idRespuestaBanco', 'count')
    ).reset_index()

    transacciones = transacciones.merge(historial, on='idCredito', how='left')

    # Convertir columnas categóricas si es necesario
    transacciones['tipoEnvio'] = transacciones['TipoEnvio'].astype('category').cat.codes

    # Solo eliminar filas que no tengan strategyId, idRespuestaBanco o montoCobrado
    dataset = transacciones.dropna(subset=['strategyId', 'idRespuestaBanco', 'montoCobrado'])

    # Rellenar nulos en otras columnas numéricas
    dataset['historial_exitos'] = dataset['historial_exitos'].fillna(0)
    dataset['historial_fallas'] = dataset['historial_fallas'].fillna(0)
    dataset['intentos'] = dataset['intentos'].fillna(0)
    dataset['horaCobro'] = dataset['horaCobro'].fillna(-1)
    dataset['diaSemanaCobro'] = dataset['diaSemanaCobro'].fillna(-1)
    dataset['dias_envio_cobro'] = dataset['dias_envio_cobro'].fillna(-1)
    dataset['monto_ratio_exigible_cobrar'] = dataset['monto_ratio_exigible_cobrar'].fillna(0)
    dataset['tipoEnvio'] = dataset['tipoEnvio'].fillna(-1)

    # Columnas finales para el modelo
    features = [
        'idCredito', 'strategyId', 'horaCobro', 'diaSemanaCobro', 'dias_envio_cobro',
        'montoCobrar', 'montoExigible', 'monto_ratio_exigible_cobrar',
        'historial_exitos', 'historial_fallas', 'intentos', 'tipoEnvio'
    ]

    dataset_final = dataset[features + ['idRespuestaBanco', 'montoCobrado']]

    dataset_final.to_csv("data/output/dataset_modelo.csv", index=False)
    print("✅ Dataset generado en 'output/dataset_modelo.csv'")

if __name__ == '__main__':
    construir_dataset()
