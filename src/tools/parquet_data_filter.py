import pandas as pd

# Leer archivo parquet
df = pd.read_parquet("US30_Data.parquet")

# Convertir columna time
df["time"] = pd.to_datetime(df["time"])

# Ordenar y poner índice
df = df.sort_values("time")
df = df.set_index("time")

# Pedir rango al usuario
start = input("Ingresa la fecha/hora de inicio (2026-02-23 00:00:00): ")
end = input("Ingresa la fecha/hora de fin (2026-02-23 23:59:59): ")

# Filtrar
df_filtrado = df.loc[start:end]

# Mostrar
print("\nFilas encontradas:")
print(df_filtrado)

# Guardar resultado
df_filtrado.to_csv("rango_filtrado.csv")

print("\nArchivo guardado como: rango_filtrado.csv")