import pandas as pd

df1 = pd.read_csv(r"data\raw\ListaCobroDetalle2022.csv")

df2 = pd.read_csv(r"data\raw\ListaCobroDetalle2023.csv")

id_bancos = pd.read_csv(r"data\raw\CatBanco.csv")

train_data = pd.concat([df1,df2], ignore_index=False)

print(train_data.head())

train_data = train_data.merge(id_bancos, left_on="idBanco", right_on="IdBanco", how="left")

print(train_data.cols())

