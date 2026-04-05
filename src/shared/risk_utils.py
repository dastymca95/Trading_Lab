from __future__ import annotations

from typing import Any, Dict


def calc_lots(
    capital: float,
    risk_pct: float,
    sl_dist: float,
    entry: float,
    p: Dict[str, Any],
) -> float:
    """
    Calcula el tamaño de la posición (lots) usando:
    - capital asignado
    - % de riesgo por trade
    - distancia al stop
    - parámetros del activo

    Espera que p contenga:
    - cs   : contract size
    - ml   : minimum lot
    - step : lot step
    - comm : comisión por lote
    - jpy  : si el activo liquida/expresa valor tipo JPY logic
    """
    if sl_dist <= 0:
        return 0.0

    contract_size = float(p["cs"])
    min_lot = float(p["ml"])
    lot_step = float(p["step"])
    commission = float(p["comm"])
    is_jpy = bool(p["jpy"])

    # Profit per unit approximation
    if is_jpy:
        price_per_unit = contract_size / max(entry, 1)
    else:
        price_per_unit = contract_size

    raw_lots = (capital * risk_pct) / max(price_per_unit * sl_dist + commission, 1e-8)

    # Ajustar al múltiplo del lot step
    rounded_lots = round(raw_lots / lot_step) * lot_step

    # Nunca bajar de minimum lot
    rounded_lots = max(min_lot, rounded_lots)

    # Tope conservador: no arriesgar más de ~20% del capital asignado por sizing extremo
    lots_cap = (capital * 0.20) / max(price_per_unit * sl_dist + commission, 0.001)
    lots_cap = round(lots_cap / lot_step) * lot_step

    final_lots = max(min_lot, min(rounded_lots, lots_cap))

    return round(final_lots, 4)