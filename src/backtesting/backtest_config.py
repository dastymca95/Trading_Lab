import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════

SEED = 42
np.random.seed(SEED)

INITIAL_PER_ASSET  = 250.0
GLOBAL_RISK_PCT    = 0.02      # ← CAMBIA SOLO ESTA LÍNEA
USE_LIVE_SPECS     = True      # ← True = MT5 live specs | False = JSON/base
AUTO_ENABLE_SYMBOL = True
COMMISSION_LOOKBACK_DAYS = 90

TRAIN_END          = pd.Timestamp('2025-01-01')
TEST_START         = pd.Timestamp('2025-01-01')
MAX_BARS           = 240
ROLLING_SHARPE_WIN = 20
MONTE_CARLO_RUNS   = 2000
BOOTSTRAP_RUNS     = 2000
TIMEFRAME_MINUTES  = 2

# Parámetros estratégicos base
ASSET_PARAMS_BASE = {
    'US30': {
        'sl_pct':0.003, 'trail_mult':3.0,  'risk_pct':GLOBAL_RISK_PCT, 'lrr_min':1.0,
        'hours':[15,18], 'dow':[0,1,2,3,4], 'atr_mult':1.5,
        'comm':0.00, 'cs':1,      'ml':0.1,  'step':0.1,  'sp':3.0,   'jpy':False, 'digits':2,
        'be_atr_mult':0.75,
    },
}

# XAUUSD excluido del portafolio activo (experimento: índices sin XAUUSD)
# Para restaurar: mover de vuelta a ASSET_PARAMS_BASE y añadir 'XAUUSD' a OPTIONAL_ASSETS si aplica
OPTIONAL_ASSETS = ['USTEC', 'US500', 'DE40']

OPTIONAL_PARAMS = {
    # ── USTEC — STATUS: LOCKED BASELINE — see BASELINE_USTEC.md ─────────────
    # Hipótesis: US_MID early (16-17h UTC = MT5 hora 18) + sesgo LONG
    # Ventana: hour 18 only — excluye slot London (hora 15) para evitar contaminación
    # Dirección: LONG forzado — edge validado en research como sesgo alcista US_MID
    # Filtro diario: LOW_VOL(p50,w90) + roll_mfe(N=20,shift1)>1.0
    # Engine: SL, trailing, sizing, LRR, vol filter — igual que todos los activos
    'USTEC': {'sl_pct':0.003,'trail_mult':3.0,'risk_pct':GLOBAL_RISK_PCT,'lrr_min':1.0,
              'hours':[18],'dow':[0,1,2,3,4],'atr_mult':1.5,
              'comm':0.00,'cs':1,'ml':0.1,'step':0.1,'sp':1.0,'jpy':False,'digits':2,
              'be_atr_mult':0.75,'force_direction':1,
              'low_vol_pct':50,'low_vol_win':90,'roll_mfe_n':20,'roll_mfe_min':1.0},
    'US500': {'sl_pct':0.003,'trail_mult':3.0,'risk_pct':GLOBAL_RISK_PCT,'lrr_min':1.0,
              'hours':[15,18],'dow':[0,1,2,3,4],'atr_mult':1.5,
              'comm':0.00,'cs':1,'ml':0.1,'step':0.1,'sp':0.5,'jpy':False,'digits':2,
              'be_atr_mult':0.75,
              # Refined hypothesis: LOW_VOL(p50,w90) + ATR<=1.45 guardrail
              'low_vol_pct':50,'low_vol_win':90,'atr_cap':1.45},
    'DE40':  {'sl_pct':0.003,'trail_mult':3.0,'risk_pct':GLOBAL_RISK_PCT,'lrr_min':1.0,
              'hours':[15,18],'dow':[0,1,2,3,4],'atr_mult':1.5,
              'comm':0.00,'cs':1,'ml':0.1,'step':0.1,'sp':1.0,'jpy':False,'digits':2,
              'be_atr_mult':0.75},
    'XAUUSD': {'sl_pct':0.003,'trail_mult':1.0,'risk_pct':GLOBAL_RISK_PCT,'lrr_min':1.5,
               'hours':[15,18],'dow':[1,2,3,4],'atr_mult':1.5,
               'comm':7.00,'cs':100,'ml':0.01,'step':0.01,'sp':0.30,'jpy':False,'digits':2,
               'be_atr_mult':0.5},
}

# Fallback manual tipo extractor
COMMISSION_MANUAL_RT = {
    'XAUUSD': 7.00,
    'US30':   0.00,
    'USTEC':  0.00,
    'US500':  0.00,
    'DE40':   0.00,
}

# ─── Experiment CLI override (solo backtesting) ──────────────────────────────
# Uso: python backtest_runner.py --be_atr_mult=0.40
import sys as _sys
for _arg in _sys.argv[1:]:
    if _arg.startswith('--be_atr_mult='):
        _val = float(_arg.split('=', 1)[1])
        _active = list(ASSET_PARAMS_BASE.values())
        _active += [OPTIONAL_PARAMS[k] for k in OPTIONAL_ASSETS if k in OPTIONAL_PARAMS]
        for _p in _active:
            _p['be_atr_mult'] = _val
        print(f"  [CLI override] be_atr_mult = {_val}")
        break
del _sys
