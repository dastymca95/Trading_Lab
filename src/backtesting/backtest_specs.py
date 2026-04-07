import os
import glob
import json
from datetime import datetime, timedelta

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

from backtest_config import (
    GLOBAL_RISK_PCT, COMMISSION_LOOKBACK_DAYS, AUTO_ENABLE_SYMBOL,
    USE_LIVE_SPECS, ASSET_PARAMS_BASE, OPTIONAL_ASSETS, OPTIONAL_PARAMS, COMMISSION_MANUAL_RT,
)


# ═══════════════════════════════════════════════════════════════════════
# UTILIDADES MT5 / SPECS
# ═══════════════════════════════════════════════════════════════════════

def connect_mt5() -> bool:
    if mt5 is None:
        print("  ⚠️  MetaTrader5 no está instalado.")
        return False
    if not mt5.initialize():
        print(f"  ❌ Error initialize(): {mt5.last_error()}")
        return False
    info = mt5.account_info()
    if info is None:
        print("  ❌ No se pudo leer account_info()")
        return False
    print(f"  ✓ Conectado MT5 | Cuenta: {info.login} | Broker: {info.company}")
    return True


def safe_symbol(symbol: str) -> bool:
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"  ⚠️  {symbol}: no existe en este broker")
        return False
    if info.visible:
        return True
    if AUTO_ENABLE_SYMBOL and mt5.symbol_select(symbol, True):
        print(f"  ✓ {symbol}: activado en Market Watch")
        return True
    print(f"  ⚠️  {symbol}: no visible y no se pudo activar")
    return False


def _comm_from_history_rt(symbol: str):
    try:
        deals = mt5.history_deals_get(
            datetime.now() - timedelta(days=COMMISSION_LOOKBACK_DAYS),
            datetime.now()
        )
        if deals is None or len(deals) == 0:
            return None
        hits = []
        for d in deals:
            if getattr(d, 'symbol', None) != symbol:
                continue
            comm = getattr(d, 'commission', 0)
            vol = getattr(d, 'volume', 0)
            if comm in (None, 0) or vol in (None, 0):
                continue
            hits.append(abs(comm) / vol)   # por lote, por lado
        if not hits:
            return None
        return round(float(sum(hits) / len(hits)) * 2, 4)
    except Exception:
        return None


def _comm_from_order_check_rt(symbol: str, volume_min: float):
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        check = mt5.order_check({
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       volume_min,
            "type":         mt5.ORDER_TYPE_BUY,
            "price":        tick.ask,
            "deviation":    50,
            "magic":        0,
            "comment":      "spec_check",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        })
        if check is None:
            return None
        commission = getattr(check, 'commission', None)
        if commission in (None, 0):
            return None
        side_per_lot = abs(float(commission)) / max(volume_min, 1e-8)
        return round(side_per_lot * 2, 4)
    except Exception:
        return None


def get_live_commission_rt(symbol: str, volume_min: float, fallback_rt: float):
    c = _comm_from_history_rt(symbol)
    if c is not None:
        return c, "history"
    c = _comm_from_order_check_rt(symbol, volume_min)
    if c is not None:
        return c, "order_check"
    return float(fallback_rt), "fallback"


def load_broker_specs_json(data_dir):
    json_files = sorted(glob.glob(os.path.join(data_dir, 'especificaciones_*.json')), reverse=True)
    if json_files:
        print(f"  ✓ Specs JSON: {os.path.basename(json_files[0])}")
        with open(json_files[0], 'r', encoding='utf-8') as f:
            return json.load(f)
    print("  ℹ️  Sin JSON de specs — usando parámetros base")
    return None


def apply_json_specs(specs, base_params):
    if specs is None:
        return {k: v.copy() for k, v in base_params.items()}
    updated = {}
    for asset, p in base_params.items():
        sp = specs.get(asset)
        if sp is None:
            updated[asset] = p.copy()
            continue
        q = p.copy()
        q['comm']   = float(sp.get('commission_round_trip', p['comm']) or p['comm'])
        q['cs']     = int(sp.get('contract_size', p['cs']) or p['cs'])
        q['ml']     = float(sp.get('volume_min', p['ml']) or p['ml'])
        q['step']   = float(sp.get('volume_step', p.get('step', p['ml'])) or p.get('step', p['ml']))
        q['digits'] = int(sp.get('digits', p['digits']) or p['digits'])
        q['jpy']    = bool(sp.get('is_jpy', p['jpy']))
        updated[asset] = q
    return updated


def apply_global_risk(params: dict, risk_pct: float) -> dict:
    out = {}
    for asset, p in params.items():
        q = p.copy()
        q['risk_pct'] = float(risk_pct)
        out[asset] = q
    return out


def apply_live_specs(base_params: dict):
    out = {}
    connected = connect_mt5()
    if not connected:
        print("  ⚠️  MT5 no disponible. Se usarán parámetros base/fallback.")
        return apply_global_risk(base_params, GLOBAL_RISK_PCT)

    for asset, p in base_params.items():
        if not safe_symbol(asset):
            q = p.copy()
            q['risk_pct'] = GLOBAL_RISK_PCT
            q['comm_source'] = 'fallback_no_symbol'
            out[asset] = q
            continue

        info = mt5.symbol_info(asset)
        q = p.copy()

        if info is not None:
            q['cs']     = int(info.trade_contract_size) if info.trade_contract_size else q['cs']
            q['ml']     = float(info.volume_min) if info.volume_min else q['ml']
            q['step']   = float(info.volume_step) if info.volume_step else q['step']
            q['digits'] = int(info.digits) if info.digits is not None else q['digits']
            q['jpy']    = bool(getattr(info, 'currency_profit', '') == 'JPY')

        comm_rt, source = get_live_commission_rt(
            asset,
            volume_min=q['ml'],
            fallback_rt=COMMISSION_MANUAL_RT.get(asset, q['comm'])
        )
        q['comm'] = float(comm_rt)
        q['comm_source'] = source
        q['risk_pct'] = GLOBAL_RISK_PCT
        out[asset] = q

    try:
        mt5.shutdown()
    except Exception:
        pass

    return out


def resolve_asset_params(data_dir):
    merged = {**ASSET_PARAMS_BASE}
    for k in OPTIONAL_ASSETS:
        if k in OPTIONAL_PARAMS:
            merged[k] = OPTIONAL_PARAMS[k].copy()

    if USE_LIVE_SPECS:
        print("1. Cargando specs live desde MT5...")
        params = apply_live_specs(merged)
    else:
        print("1. Buscando specs del broker en JSON...")
        specs = load_broker_specs_json(data_dir)
        params = apply_json_specs(specs, merged)

    params = apply_global_risk(params, GLOBAL_RISK_PCT)
    return params
