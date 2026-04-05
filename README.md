# Trading Lab

Sistema de trading algorítmico multi-estrategia para forex y commodities, con ejecución live via MetaTrader 5 y framework de backtesting estadístico.

---

## Estructura del Proyecto

```
Trading_Lab/
├── config/                    # Configuración global y por estrategia (YAML)
│   ├── global.yaml
│   ├── london_bot.yaml
│   ├── microscalping.yaml
│   ├── execution_audit.yaml
│   ├── mt5_accounts.yaml
│   ├── risk_limits.yaml
│   └── symbols.yaml
├── notebooks/                 # Análisis exploratorio
│   ├── london_review.ipynb
│   ├── slippage_analysis.ipynb
│   └── microstructure_notes.ipynb
├── src/
│   ├── london_bot/            # Estrategia London Range Breakout (LIVE)
│   ├── backtesting/           # Framework de backtesting (OPERATIVO)
│   ├── microscalping_lab/     # Lab de microescalping (EN DESARROLLO)
│   ├── execution_audit/       # Auditoría de ejecución live (PENDIENTE)
│   ├── shared/                # Utilidades comunes (PENDIENTE)
│   └── tools/                 # Pipeline de datos MT5
└── tests/
```

---

## Módulos

### London Bot (`src/london_bot/`)

Estrategia de breakout del rango de Londres en ejecución live contra MT5.

**Activos:** XAUUSD, US30, USTEC, US500, DE40

**Logica:**
- Identifica el rango de la sesion de Londres (09:00–15:00 GMT)
- Genera señales de breakout en la ventana 15:00–18:00 GMT
- Stop loss basado en porcentaje del precio (`sl_pct = 0.3%`)
- Trailing stop con multiplicador configurable por activo
- Maximo 2 operaciones por activo por día
- Persistencia de estado en JSON para recuperacion tras reinicios
- Audit de slippage en CSV

**Parametros por activo:**

| Activo | Trail Mult | LRR Min | Dias |
|--------|-----------|---------|------|
| XAUUSD | 1.0x | 1.5 | Mar–Vie |
| US30 | 3.0x | 1.0 | Lun–Vie |
| USTEC | 3.0x | 1.0 | Lun–Vie |
| US500 | 3.0x | 1.0 | Lun–Vie |
| DE40 | 3.0x | 1.0 | Lun–Vie |

**Archivos clave:**
- `main_london.py` — Motor principal con integracion MT5 live
- `london_levels.py` — Calculo del rango
- `trade_manager_london.py` — Ciclo de vida de operaciones
- `risk_manager_london.py` — Control de riesgo

---

### Backtesting (`src/backtesting/`)

Framework híbrido alineado con el bot live, diseñado para reproducir fielmente las condiciones de ejecucion real.

**Características:**
- `GLOBAL_RISK_PCT`: un solo parametro controla el riesgo de todos los activos
- Datos desde Parquet (sin limite de filas, 5x mas rapido que Excel) con fallback a `.xlsx`
- Especificaciones live desde MT5 o JSON fallback
- Comision con jerarquia: historial real de deals → `order_check()` → fallback manual
- Spread dinamico desde la vela historica
- Periodo de entrenamiento hasta `2025-01-01`, test desde `2025-01-01`
- Ventana maxima: 240 barras M2

**Analisis estadistico:**
- Bootstrap resampling (2,000 runs)
- Sign test para robustez de señales
- Monte Carlo simulation (2,000 runs)
- Rolling Sharpe ratio (ventana 20 barras)
- Heatmaps de sensibilidad de parametros
- Barrido de riesgo en 15 niveles (0.25% a 6%)

**Salidas:** Excel con metricas, CSV resumen, PNG charts

**Archivos clave:**
- `backtest_runner.py` — Backtester hibrido v2 Ultra Aligned
- `risk_sensitivity_runner.py` — Análisis de sensibilidad al riesgo

---

### Data Tools (`src/tools/`)

Pipeline de extraccion y transformacion de datos de mercado desde MT5.

**Activos soportados:** XAUUSD, EURUSD, USDJPY, US30, USTEC, US500, DE40, XAGUSD, GBPUSD

**Caracteristicas:**
- Extraccion de datos OHLCV en timeframe M2 desde 2018-01-01
- Salida en formato Parquet (primario), Excel specs+QC, JSON de especificaciones
- Filtrado por rango de fechas con `parquet_data_filter.py`
- ~1.3M filas por activo manejadas de forma nativa

**Archivos clave:**
- `mt5_data_pipeline.py` — Extractor MT5 v3
- `parquet_data_filter.py` — Filtro por fechas

---

### Microscalping Lab (`src/microscalping_lab/`)

Estructura para estrategias avanzadas de microescalping. Implementacion pendiente.

**Estrategias planificadas:**
- `micro_breakout.py` — Breakout de micro-rango
- `momentum_burst.py` — Rafaga de momentum
- `sweep_reaction.py` — Reaccion a barridos de liquidez

**Shadow mode** planeado para validacion en paper trading antes de ir live.

---

### Execution Audit (`src/execution_audit/`)

Modulo para auditar la calidad de ejecucion live. Estructura definida, implementacion pendiente.

---

## Stack Tecnologico

| Tecnologia | Uso |
|-----------|-----|
| Python 3 | Lenguaje principal |
| MetaTrader5 | Ejecucion live y datos historicos |
| Pandas / NumPy | Procesamiento de datos |
| SciPy | Analisis estadistico |
| Parquet (PyArrow) | Almacenamiento de datos historicos |
| openpyxl | Reportes en Excel |
| Matplotlib | Visualizacion de resultados |

---

## Instalacion

```bash
pip install pandas numpy scipy openpyxl pyarrow MetaTrader5 matplotlib
```

---

## Uso Rapido

**Ejecutar el London Bot:**
```bash
python src/london_bot/main_london.py
```

**Ejecutar backtesting:**
```bash
python src/backtesting/backtest_runner.py
```

**Extraer datos desde MT5:**
```bash
python src/tools/mt5_data_pipeline.py
```

---

## Parametros Globales Clave

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| `GLOBAL_RISK_PCT` | `0.02` | Riesgo por operacion (2%) |
| `INITIAL_CAPITAL` | `250.0` | Capital inicial por activo |
| `MAX_TRADES_PER_DAY_PER_ASSET` | `2` | Maximo de trades diarios por activo |
| `MT5_OFFSET_HOURS` | `1` | Correccion de hora MT5 |
| `MONTE_CARLO_RUNS` | `2000` | Runs de simulacion Monte Carlo |
| `TRAIN_END` | `2025-01-01` | Fin del periodo de entrenamiento |
