import os
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(_REPO_ROOT, "data", "backtesting")
os.makedirs(DATA_DIR, exist_ok=True)

# Leer archivo parquet
df = pd.read_parquet(os.path.join(DATA_DIR, "US30_Data.parquet"))

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
out_path = os.path.join(DATA_DIR, "rango_filtrado.csv")
df_filtrado.to_csv(out_path)

print(f"\nArchivo guardado como: {out_path}")