#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime,timedelta
from typing import Dict,Optional,Tuple,List
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os,time,json,logging, sys
from parity_logger_legacy import ensure_parity_csv, write_parity_row



PARITY_FILE = r"C:\Users\Dasty\PycharmProjects\Trading_Lab\reports\london_bot\parity_log_legacy.csv"
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
MT5_OFFSET_HOURS=1
INITIAL_CAPITAL=250.0
MAX_TRADES_PER_DAY_PER_ASSET=2
TRAILING_CHECK_SECONDS=10
BOT_MAGIC=20260402
STATE_PATH=os.path.join(BASE_DIR,"bot_state.json")
AUDIT_PATH=os.path.join(BASE_DIR,"slippage_audit.csv")
VOLUME_HISTORY_FROM=datetime(2018,1,1)
DEFAULT_RISK_PCT=0.02
FORCE_TEST_TRADE=False
FORCE_TEST_SYMBOL="US30"
PARITY_ONLY_MODE = False
FORCE_PARITY_SCAN = False
FORCE_TEST_DIRECTION=1
FORCE_TEST_RETURN_AFTER_OPEN=True
PREFER_PARQUET_FOR_VOLUME_MEAN=True
VOLUME_MEANS: Dict[str,float]={}
ASSET_PARAMS: Dict[str,dict]={}
AUDIT_COLUMNS=["datetime","asset","direction","lots","signal_price","real_entry_price","slippage_pts","spread_signal","spread_at_entry","execution_ms","pnl_real","pnl_backtest_approx","exit_reason"]
ASSET_STRATEGY={
"XAUUSD":{"sl_pct":0.003,"trail_mult":1.0,"risk_pct":DEFAULT_RISK_PCT,"lrr_min":1.5,"hours":[15,18],"dow":[1,2,3,4],"atr_mult":1.5,"fallback_comm":7.00,"fallback_cs":100,"fallback_ml":0.01,"fallback_step":0.01,"fallback_sp":0.30,"fallback_digits":2,"fallback_jpy":False},
"US30":{"sl_pct":0.003,"trail_mult":3.0,"risk_pct":DEFAULT_RISK_PCT,"lrr_min":1.0,"hours":[15,18],"dow":[0,1,2,3,4],"atr_mult":1.5,"fallback_comm":0.00,"fallback_cs":1,"fallback_ml":0.1,"fallback_step":0.1,"fallback_sp":3.0,"fallback_digits":2,"fallback_jpy":False},
"USTEC":{"sl_pct":0.003,"trail_mult":3.0,"risk_pct":DEFAULT_RISK_PCT,"lrr_min":1.0,"hours":[15,18],"dow":[0,1,2,3,4],"atr_mult":1.5,"fallback_comm":0.00,"fallback_cs":1,"fallback_ml":0.1,"fallback_step":0.1,"fallback_sp":1.0,"fallback_digits":2,"fallback_jpy":False},
"US500":{"sl_pct":0.003,"trail_mult":3.0,"risk_pct":DEFAULT_RISK_PCT,"lrr_min":1.0,"hours":[15,18],"dow":[0,1,2,3,4],"atr_mult":1.5,"fallback_comm":0.00,"fallback_cs":1,"fallback_ml":0.1,"fallback_step":0.1,"fallback_sp":0.5,"fallback_digits":2,"fallback_jpy":False},
"DE40":{"sl_pct":0.003,"trail_mult":3.0,"risk_pct":DEFAULT_RISK_PCT,"lrr_min":1.0,"hours":[15,18],"dow":[0,1,2,3,4],"atr_mult":1.5,"fallback_comm":0.00,"fallback_cs":1,"fallback_ml":0.1,"fallback_step":0.1,"fallback_sp":1.0,"fallback_digits":2,"fallback_jpy":False},
}
log_path=os.path.join(BASE_DIR,f"bot_log_{datetime.now().strftime('%Y%m%d')}.txt")
logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',handlers=[logging.FileHandler(log_path,encoding='utf-8'),logging.StreamHandler(sys.stdout)])
log=logging.getLogger(__name__)

def ensure_audit_csv():
    if not os.path.exists(AUDIT_PATH):
        pd.DataFrame(columns=AUDIT_COLUMNS).to_csv(AUDIT_PATH,index=False); return
    try:
        old=list(pd.read_csv(AUDIT_PATH,nrows=0).columns)
        if old!=AUDIT_COLUMNS:
            backup=os.path.join(BASE_DIR,f"slippage_audit_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            os.replace(AUDIT_PATH,backup)
            pd.DataFrame(columns=AUDIT_COLUMNS).to_csv(AUDIT_PATH,index=False)
            log.warning(f"AUDIT CSV incompatible. Respaldado en {os.path.basename(backup)} y recreado.")
    except Exception:
        pd.DataFrame(columns=AUDIT_COLUMNS).to_csv(AUDIT_PATH,index=False)

ensure_audit_csv()

def _serialize_pos(pos):
    out={}
    for k,v in pos.items():
        if isinstance(v,datetime): out[k]={"__datetime__":v.isoformat()}
        elif isinstance(v,(int,float,bool,str)): out[k]=v
        else: out[k]=str(v)
    return out
def _deserialize_pos(pos):
    out={}
    for k,v in pos.items():
        if isinstance(v,dict) and "__datetime__" in v: out[k]=datetime.fromisoformat(v["__datetime__"])
        elif k in ("ticket","direction","digits","cs","exec_ms"): out[k]=int(float(v))
        elif k in ("lots","real_entry_price","signal_price","backtest_price","slippage","spread","spread_signal","sl","atr","trail_mult","be_level","best_price","comm"): out[k]=float(v)
        elif k in ("breakeven_hit","jpy"): out[k]=bool(v) if isinstance(v,bool) else str(v).lower()=="true"
        else: out[k]=v
    return out
def save_state(open_positions,trades_today):
    try:
        state={"open_positions":{sym:_serialize_pos(pos) for sym,pos in open_positions.items()},"trades_today":trades_today,"saved_at":datetime.now().isoformat()}
        tmp=STATE_PATH+".tmp"
        with open(tmp,"w",encoding="utf-8") as f: json.dump(state,f,indent=2)
        os.replace(tmp,STATE_PATH)
    except Exception as e: log.error(f"Error guardando estado: {e}")
def load_state()->Tuple[dict,dict]:
    if not os.path.exists(STATE_PATH): return {},{}
    try:
        with open(STATE_PATH,"r",encoding="utf-8") as f: state=json.load(f)
        open_pos={sym:_deserialize_pos(pos) for sym,pos in state.get("open_positions",{}).items()}
        trades_hoy={k:int(v) for k,v in state.get("trades_today",{}).items()}
        log.info(f"Estado cargado (guardado: {state.get('saved_at','?')}) | Posiciones: {list(open_pos.keys())} | Trades hoy: {trades_hoy}")
        return open_pos,trades_hoy
    except Exception as e:
        log.error(f"Error cargando estado: {e}"); return {},{}

def connect():
    if not mt5.initialize():
        log.error(f"MT5 initialize() falló: {mt5.last_error()}"); return False
    info=mt5.account_info()
    if info is None:
        log.error("No se pudo leer account_info()"); return False
    log.info(f"Conectado | Cuenta: {info.login} | Balance: ${info.balance:.2f} | Demo: {info.trade_mode==0}")
    return True
def get_mt5_now(): return datetime.now()+timedelta(hours=MT5_OFFSET_HOURS)
def get_tick(symbol): return mt5.symbol_info_tick(symbol)
def get_spread_pts(symbol):
    tick=get_tick(symbol); info=mt5.symbol_info(symbol)
    return round(tick.ask-tick.bid,info.digits) if tick and info else 0.0
def get_live_spread_fallback(symbol,digits,fallback_sp):
    tick=mt5.symbol_info_tick(symbol)
    return float(fallback_sp) if tick is None else round(tick.ask-tick.bid,digits)
def build_live_asset_params():
    final={}
    for symbol,s in ASSET_STRATEGY.items():
        info=mt5.symbol_info(symbol)
        if info is None:
            final[symbol]={"sl_pct":s["sl_pct"],"trail_mult":s["trail_mult"],"risk_pct":s["risk_pct"],"lrr_min":s["lrr_min"],"hours":s["hours"],"dow":s["dow"],"atr_mult":s["atr_mult"],"comm":s["fallback_comm"],"cs":s["fallback_cs"],"ml":s["fallback_ml"],"step":s["fallback_step"],"sp":s["fallback_sp"],"digits":s["fallback_digits"],"jpy":s["fallback_jpy"]}
            continue
        digits=int(info.digits); point=float(info.point) if info.point else (10**(-digits) if digits>0 else 1.0)
        spread_now=get_live_spread_fallback(symbol,digits,s["fallback_sp"]); spread_fallback=max(spread_now,point)
        final[symbol]={"sl_pct":s["sl_pct"],"trail_mult":s["trail_mult"],"risk_pct":s["risk_pct"],"lrr_min":s["lrr_min"],"hours":s["hours"],"dow":s["dow"],"atr_mult":s["atr_mult"],"comm":float(s["fallback_comm"]),"cs":int(info.trade_contract_size) if info.trade_contract_size else int(s["fallback_cs"]),"ml":float(info.volume_min) if info.volume_min else float(s["fallback_ml"]),"step":float(info.volume_step) if info.volume_step else float(s["fallback_step"]),"sp":float(spread_fallback),"digits":digits,"jpy":bool(getattr(info,"currency_profit","")=="JPY")}
    return final
def get_candles(symbol,timeframe=mt5.TIMEFRAME_M2,n=1800):
    rates=mt5.copy_rates_from_pos(symbol,timeframe,0,n)
    if rates is None or len(rates)==0: return None
    df=pd.DataFrame(rates); df["time"]=pd.to_datetime(df["time"],unit="s"); return df
def engineer_rates_df(df,digits):
    df=df.copy().sort_values("time").reset_index(drop=True)
    df["date"]=df["time"].dt.date; df["dow"]=df["time"].dt.dayofweek; df["pc"]=df["close"].shift(1)
    df["tr"]=np.maximum(df["high"]-df["low"],np.maximum(abs(df["high"]-df["pc"]),abs(df["low"]-df["pc"])))
    df["atr14"]=df["tr"].rolling(14).mean()
    point=10**(-digits) if digits>0 else 1.0
    if "spread" in df.columns and df["spread"].notna().any():
        df["spread_px"]=pd.to_numeric(df["spread"],errors="coerce").ffill().bfill().fillna(0)*point
    else: df["spread_px"]=np.nan
    return df
def get_signal_context(symbol,signal_date,signal_hour,p):
    df_raw=get_candles(symbol,mt5.TIMEFRAME_M2,1800)
    if df_raw is None or len(df_raw)<100: return None
    df=engineer_rates_df(df_raw,p["digits"]); today=df[df["date"]==signal_date].copy()
    if len(today)<20: return None
    sig=today[(today["time"].dt.hour==signal_hour)&(today["time"].dt.minute==0)].copy()
    if len(sig)==0: return None
    sig=sig.iloc[-1]
    london=today[(today["time"].dt.hour>=9)&(today["time"].dt.hour<15)].copy()
    if len(london)==0: return None
    lh=float(london["high"].max()); ll=float(london["low"].min()); lam=float(london["atr14"].mean()) if london["atr14"].notna().any() else 0.0
    lrr=(lh-ll)/lam if lam>0 else 0.0
    spread_signal=sig.get("spread_px",np.nan)
    if pd.isna(spread_signal) or spread_signal<=0: spread_signal=float(p["sp"])
    return {"signal_row":sig,"lh":lh,"ll":ll,"lam":lam,"lrr":lrr,"spread_signal":float(spread_signal)}
def load_parquet_volume_mean(symbol):
    path=os.path.join(BASE_DIR,f"{symbol}_Data.parquet")
    if not os.path.exists(path): return None
    try:
        df=pd.read_parquet(path,engine="pyarrow",columns=["tick_volume"])
        if len(df)==0: return None
        vm=float(pd.to_numeric(df["tick_volume"],errors="coerce").dropna().mean())
        return vm if np.isfinite(vm) else None
    except Exception as e:
        log.warning(f"{symbol}: no se pudo leer parquet para vm_global ({e})"); return None
def load_mt5_volume_mean(symbol):
    try:
        rates=mt5.copy_rates_range(symbol,mt5.TIMEFRAME_M2,VOLUME_HISTORY_FROM,datetime.now())
        if rates is None or len(rates)==0: return None
        vm=float(pd.DataFrame(rates)["tick_volume"].mean())
        return vm if np.isfinite(vm) else None
    except Exception as e:
        log.warning(f"{symbol}: no se pudo calcular vm_global desde MT5 ({e})"); return None
def validate_symbol(symbol,p):
    info=mt5.symbol_info(symbol)
    if info is None:
        log.error(f"{symbol}: símbolo no encontrado en MT5"); return False
    if not info.visible: mt5.symbol_select(symbol,True)
    log.info(f"{symbol}: trade_mode={getattr(info,'trade_mode',None)} | trade_exemode={getattr(info,'trade_exemode',None)} | filling_mode={getattr(info,'filling_mode',None)} | volume_min={getattr(info,'volume_min',None)} | volume_step={getattr(info,'volume_step',None)} | stops_level={getattr(info,'trade_stops_level',None)} | freeze_level={getattr(info,'trade_freeze_level',None)}")
    return True
def init_volume_means():
    log.info("Calculando medias de volumen histórico...")
    for symbol in ASSET_PARAMS:
        vm=None; source="disabled"
        if PREFER_PARQUET_FOR_VOLUME_MEAN:
            vm=load_parquet_volume_mean(symbol); source="parquet" if vm is not None else "mt5_history"
            if vm is None: vm=load_mt5_volume_mean(symbol)
        else:
            vm=load_mt5_volume_mean(symbol); source="mt5_history" if vm is not None else "parquet"
            if vm is None: vm=load_parquet_volume_mean(symbol)
        if vm is None: vm=0.0; source="disabled"
        VOLUME_MEANS[symbol]=float(vm); log.info(f"  {symbol}: vm_global={vm:.1f} | source={source}")
def calc_lots(capital,risk_pct,sl_dist,entry,p):
    if sl_dist<=0: return 0.0
    ppu=p["cs"]/max(entry,1) if p["jpy"] else p["cs"]; step=p["step"]
    lots=(capital*risk_pct)/max(ppu*sl_dist+p["comm"],1e-8)
    lots=round(lots/step)*step; lots=max(p["ml"],lots)
    lots_max=(capital*0.20)/max(ppu*sl_dist+p["comm"],0.001)
    lots_max=round(lots_max/step)*step; lots=max(p["ml"],min(lots,lots_max))
    return round(lots,4)
def get_filling_candidates(symbol):
    info=mt5.symbol_info(symbol)
    if info is None: return [mt5.ORDER_FILLING_IOC,mt5.ORDER_FILLING_FOK,mt5.ORDER_FILLING_RETURN]
    preferred=[]; reported=info.filling_mode; modes=[mt5.ORDER_FILLING_IOC,mt5.ORDER_FILLING_FOK,mt5.ORDER_FILLING_RETURN]
    for mode in modes:
        try:
            if reported==mode or (reported & mode)==mode: preferred.append(mode)
        except TypeError:
            if reported==mode: preferred.append(mode)
    for mode in modes:
        if mode not in preferred: preferred.append(mode)
    return preferred
def find_exit_deals(ticket,lookback_hours=168):
    try:
        deals=mt5.history_deals_get(datetime.now()-timedelta(hours=lookback_hours),datetime.now())
        if not deals: return []
        return sorted([d for d in deals if d.magic==BOT_MAGIC and d.position_id==ticket and d.entry==1],key=lambda d:d.time)
    except Exception as e:
        log.error(f"Error buscando exit deals ticket={ticket}: {e}"); return []
def audit_closed_position(pos,exit_reason):
    symbol=pos["symbol"]; ticket=pos["ticket"]; direction=pos["direction"]; lots=pos["lots"]
    try:
        exit_deals=find_exit_deals(ticket)
        if not exit_deals:
            log.warning(f"{symbol} ticket={ticket}: no se encontraron exit deals para auditoría."); return
        pnl_real=float(sum(getattr(d,"profit",0.0) for d in exit_deals)); close_price=float(exit_deals[-1].price)
        bp=pos["signal_price"]; spread_signal=pos.get("spread_signal",pos.get("spread",0.0))
        if pos["jpy"]:
            avg=(bp+close_price)/2; raw_bt=(close_price-bp)*direction*lots*pos["cs"]/max(avg,1)
        else:
            raw_bt=(close_price-bp)*direction*lots*pos["cs"]
        pnl_bt_approx=raw_bt-lots*pos["comm"]-spread_signal*lots*pos["cs"]
        row={"datetime":datetime.now().isoformat(),"asset":symbol,"direction":"BUY" if direction==1 else "SELL","lots":lots,"signal_price":pos["signal_price"],"real_entry_price":pos["real_entry_price"],"slippage_pts":pos["slippage"],"spread_signal":spread_signal,"spread_at_entry":pos["spread"],"execution_ms":pos["exec_ms"],"pnl_real":round(pnl_real,2),"pnl_backtest_approx":round(pnl_bt_approx,2),"exit_reason":exit_reason}
        pd.DataFrame([row],columns=AUDIT_COLUMNS).to_csv(AUDIT_PATH,mode="a",header=False,index=False)
        log.info(f"📋 Auditado: {symbol} | pnl_real=${pnl_real:+.2f} | pnl_bt≈${pnl_bt_approx:+.2f} | razón={exit_reason}")
    except Exception as e:
        log.error(f"Error auditando cierre de {symbol}: {e}")
def open_position(symbol,direction,lots,sl,p,signal_price,spread_signal,stype):
    tick=get_tick(symbol)
    if tick is None: log.error(f"{symbol}: no se pudo obtener tick"); return None
    info=mt5.symbol_info(symbol)
    if info is None: log.error(f"{symbol}: symbol_info() devolvió None"); return None
    price=tick.ask if direction==1 else tick.bid; spread_entry=get_spread_pts(symbol); order_type=mt5.ORDER_TYPE_BUY if direction==1 else mt5.ORDER_TYPE_SELL
    filling_candidates=get_filling_candidates(symbol); last_result=None
    for filling_mode in filling_candidates:
        request={"action":mt5.TRADE_ACTION_DEAL,"symbol":symbol,"volume":lots,"type":order_type,"price":price,"sl":round(sl,p["digits"]),"deviation":30,"magic":BOT_MAGIC,"comment":"bot_demo_LRB_ultra_v7","type_time":mt5.ORDER_TIME_GTC,"type_filling":filling_mode}
        check=mt5.order_check(request)
        if check is None:
            log.warning(f"{symbol}: order_check devolvió None | last_error={mt5.last_error()} | request={request}")
        else:
            log.info(f"{symbol}: order_check filling_mode={filling_mode} | retcode={getattr(check,'retcode',None)} | comment={getattr(check,'comment','')}")
        t_before=time.time(); log.info(f"{symbol}: intentando orden con filling_mode={filling_mode}")
        result=mt5.order_send(request); exec_ms=int((time.time()-t_before)*1000); last_result=result
        if result is not None and result.retcode==mt5.TRADE_RETCODE_DONE:
            real_price=result.price; slippage=round((real_price-signal_price)*direction,p["digits"]); time.sleep(0.3); real_ticket=int(result.order)
            open_pos_mt5=mt5.positions_get(symbol=symbol)
            if open_pos_mt5:
                for p_mt5 in open_pos_mt5:
                    if p_mt5.magic==BOT_MAGIC and abs(p_mt5.volume-lots)<0.001 and p_mt5.type==(0 if direction==1 else 1):
                        real_ticket=int(p_mt5.ticket); break
            be_level=real_price+spread_entry if direction==1 else real_price-spread_entry
            log.info(f"✅ {symbol} {'BUY' if direction==1 else 'SELL'} {lots}L @ {real_price} | signal={signal_price} | slip={slippage:+.{p['digits']}f} | spread_entry={spread_entry} | spread_signal={spread_signal} | {exec_ms}ms | order_ticket={result.order} | pos_ticket={real_ticket} | {stype}")
            return {"ticket":real_ticket,"symbol":symbol,"direction":int(direction),"lots":float(lots),"real_entry_price":float(real_price),"signal_price":float(signal_price),"backtest_price":float(signal_price),"slippage":float(slippage),"spread":float(spread_entry),"spread_signal":float(spread_signal),"exec_ms":int(exec_ms),"sl":float(sl),"atr":0.0,"trail_mult":float(p["trail_mult"]),"be_level":float(be_level),"best_price":float(real_price),"breakeven_hit":False,"open_time":datetime.now(),"digits":int(p["digits"]),"cs":int(p["cs"]),"jpy":bool(p["jpy"]),"comm":float(p["comm"]),"stype":stype}
        if result is None:
            log.warning(f"{symbol}: intento con filling_mode={filling_mode} devolvió None | last_error={mt5.last_error()} | request={request}")
        else:
            log.warning(f"{symbol}: intento con filling_mode={filling_mode} rechazado | retcode={result.retcode} | comment={getattr(result,'comment','')}")
    if last_result is None: log.error(f"{symbol}: orden rechazada definitivamente | result=None | last_error={mt5.last_error()}")
    else: log.error(f"{symbol}: orden rechazada definitivamente | retcode={last_result.retcode} | comment={getattr(last_result,'comment','')}")
    return None
def sl_is_valid_for_broker(symbol,direction,new_sl,digits):
    info=mt5.symbol_info(symbol); tick=mt5.symbol_info_tick(symbol)
    if info is None or tick is None: return False,"symbol_info/tick unavailable"
    point=float(info.point) if info.point else (10**(-digits) if digits>0 else 1.0)
    stops_level_px=float(getattr(info,"trade_stops_level",0) or 0)*point
    freeze_level_px=float(getattr(info,"trade_freeze_level",0) or 0)*point
    min_distance=max(stops_level_px,freeze_level_px,point)
    current_ref=tick.bid if direction==1 else tick.ask; dist=abs(current_ref-new_sl)
    if dist<min_distance: return False,f"SL demasiado cerca del precio actual | dist={dist:.8f} < min_distance={min_distance:.8f}"
    return True,"ok"
def modify_sl(ticket,symbol,direction,new_sl,digits):
    valid,reason=sl_is_valid_for_broker(symbol,direction,new_sl,digits)
    if not valid:
        log.info(f"{symbol} ticket={ticket}: modify_sl omitido | {reason}"); return False
    request={"action":mt5.TRADE_ACTION_SLTP,"position":ticket,"symbol":symbol,"sl":round(new_sl,digits)}
    check=mt5.order_check(request)
    if check is not None: log.info(f"{symbol} ticket={ticket}: order_check modify_sl | retcode={getattr(check,'retcode',None)} | comment={getattr(check,'comment','')}")
    result=mt5.order_send(request)
    if result is None or result.retcode!=mt5.TRADE_RETCODE_DONE:
        code=result.retcode if result else "None"; log.warning(f"{symbol} ticket={ticket}: modify_sl falló retcode={code} | last_error={mt5.last_error()}"); return False
    return True
def force_demo_trade(symbol,direction=1):
    if symbol not in ASSET_PARAMS: log.error(f"Símbolo de prueba no soportado: {symbol}"); return None
    p=ASSET_PARAMS[symbol]; tick=get_tick(symbol)
    if tick is None: log.error(f"{symbol}: no se pudo obtener tick para prueba"); return None
    entry=tick.ask if direction==1 else tick.bid; spread_signal=float(p["sp"]); spread_entry=get_spread_pts(symbol)
    min_buffer=10**(-p["digits"])*50; sl_buffer=max(entry*p["sl_pct"],spread_entry*3,min_buffer)
    sl=entry-sl_buffer if direction==1 else entry+sl_buffer; lots=p["ml"]
    pos=open_position(symbol,direction,lots,sl,p,entry,spread_signal,"FORCE_TEST_TRADE")
    if pos: pos["atr"]=1.0; log.info(f"🧪 PRUEBA EXITOSA {symbol} | pos_ticket={pos['ticket']}")
    else: log.error(f"🧪 PRUEBA FALLÓ {symbol}")
    return pos
def check_signal(symbol,p,trades_today,signal_hour):
    mt5_now=get_mt5_now()
    if not FORCE_PARITY_SCAN:
        if mt5_now.weekday() not in p["dow"] or trades_today >= MAX_TRADES_PER_DAY_PER_ASSET:
            return None
    ctx=get_signal_context(symbol,mt5_now.date(),signal_hour,p)
    if not ctx: return None
    sig=ctx["signal_row"]; ep=float(sig["close"]); av=float(sig["atr14"]) if pd.notna(sig["atr14"]) else 0.0; cr=float(sig["high"]-sig["low"]); tv=float(sig["tick_volume"]); lh=ctx["lh"]; ll=ctx["ll"]; lrr=ctx["lrr"]; sp_signal=ctx["spread_signal"]
    if av<=0: return None
    vm=VOLUME_MEANS.get(symbol,0.0)
    if vm>0 and tv<vm: return None
    if not np.isfinite(lrr) or lrr<=p["lrr_min"]: return None
    direction=None; stype=""
    if ep>lh: direction=1; stype=f"Breakout ALCISTA (London High={lh:.{p['digits']}f})"
    elif ep<ll: direction=-1; stype=f"Breakout BAJISTA (London Low={ll:.{p['digits']}f})"
    elif cr>p["atr_mult"]*av: direction=-1 if sig["close"]>sig["open"] else 1; stype=f"Vela grande ({cr:.{p['digits']}f} > {p['atr_mult']}xATR)"
    if direction is None: return None
    sl=ep*(1-p["sl_pct"]) if direction==1 else ep*(1+p["sl_pct"]); sl=max(sl,ll) if direction==1 else min(sl,lh)
    sl_dist=max(abs(ep-sl)+sp_signal,ep*0.0015)
    if sl_dist<1e-8: return None
    lots=calc_lots(INITIAL_CAPITAL,p["risk_pct"],sl_dist,ep,p)
    if lots<=0: return None
    return {"symbol":symbol,"direction":direction,"stype":stype,"ep":ep,"sl":sl,"sl_dist":sl_dist,"lots":lots,"atr":av,"lh":lh,"ll":ll,"lrr":lrr,"spread_signal":sp_signal,"signal_time":sig["time"],"signal_tick_volume":tv}
def update_trailing(pos):
    symbol=pos["symbol"]; direction=pos["direction"]; ticket=pos["ticket"]; atr=pos["atr"]; digits=pos["digits"]
    tick=get_tick(symbol)
    if tick is None: return "error"
    current=tick.bid if direction==1 else tick.ask; positions=mt5.positions_get(ticket=ticket)
    if not positions: return "stopped"
    cur_sl=float(positions[0].sl)
    pos["best_price"]=max(pos["best_price"],current) if direction==1 else min(pos["best_price"],current)
    be_level=pos["be_level"]
    pos["breakeven_hit"]=pos["breakeven_hit"] or ((pos["best_price"]>=be_level) if direction==1 else (pos["best_price"]<=be_level))
    if not pos["breakeven_hit"]: return "ok"
    new_sl=max(pos["best_price"]-atr*pos["trail_mult"],be_level,cur_sl) if direction==1 else min(pos["best_price"]+atr*pos["trail_mult"],be_level,cur_sl)
    new_sl=round(new_sl,digits)
    if abs(new_sl-cur_sl)>10**(-digits):
        if modify_sl(ticket,symbol,direction,new_sl,digits): pos["sl"]=new_sl
    return "ok"
def reconcile_with_mt5(open_positions,trades_today):
    for sym in list(open_positions.keys()):
        pos=open_positions[sym]; ticket=pos["ticket"]; mt5_pos=mt5.positions_get(ticket=ticket)
        if mt5_pos:
            current_sl=float(mt5_pos[0].sl)
            if current_sl!=pos["sl"]: log.info(f"{sym}: SL actualizado durante reinicio {pos['sl']} → {current_sl}"); pos["sl"]=current_sl
            log.info(f"✅ {sym} ticket={ticket}: activa en MT5")
        else:
            log.info(f"⚠️ {sym} ticket={ticket}: cerrada mientras bot estaba inactivo"); audit_closed_position(pos,"SL_MIENTRAS_BOT_APAGADO"); open_positions.pop(sym)
    return open_positions,trades_today
def main():

    global ASSET_PARAMS
    log.info("╔════════════════════════════════════════════════════════════╗")
    log.info("║ BOT DEMO v7 ULTRA CORREGIDO — London Range Breakout      ║")
    log.info("╚════════════════════════════════════════════════════════════╝")
    if not connect(): log.error("No se pudo conectar a MT5."); return
    ensure_parity_csv(PARITY_FILE)
    ASSET_PARAMS=build_live_asset_params()
    log.info(f"risk_pct={DEFAULT_RISK_PCT:.2%} | capital/activo=${INITIAL_CAPITAL} | MT5_offset={MT5_OFFSET_HOURS}h | prefer_parquet_vm={PREFER_PARQUET_FOR_VOLUME_MEAN}")
    for sym,p in ASSET_PARAMS.items(): validate_symbol(sym,p)
    init_volume_means()
    if FORCE_TEST_TRADE:
        pos=force_demo_trade(FORCE_TEST_SYMBOL,FORCE_TEST_DIRECTION)
        if pos and FORCE_TEST_RETURN_AFTER_OPEN: log.info("🧪 Prueba manual completada. Script finalizado por configuración."); mt5.shutdown(); return
    open_positions,trades_today=load_state(); open_positions,trades_today=reconcile_with_mt5(open_positions,trades_today); save_state(open_positions,trades_today)
    last_signal_check={}; last_date=None
    log.info("Bot corriendo. Señal base: vela 15:00 / 18:00; escaneo a 15:02-15:03 / 18:02-18:03 MT5...\n")
    while True:
        try:
            now=datetime.now(); mt5_now=now+timedelta(hours=MT5_OFFSET_HOURS)
            if last_date!=mt5_now.date():
                last_date=mt5_now.date(); trades_today={s:0 for s in ASSET_PARAMS}; last_signal_check.clear(); save_state(open_positions,trades_today); log.info(f"── Nuevo día MT5: {mt5_now.date()} ──")
            state_changed=False
            for symbol,pos in list(open_positions.items()):
                digits=pos.get("digits",2); sl_before=pos.get("sl",0.0); be_before=pos.get("breakeven_hit",False); bp_before=pos.get("best_price",0.0); status=update_trailing(pos)
                if status=="stopped":
                    audit_closed_position(pos,"SL_MT5"); open_positions.pop(symbol); state_changed=True
                elif pos.get("sl",0.0)!=sl_before or pos.get("breakeven_hit",False)!=be_before or abs(pos.get("best_price",0.0)-bp_before)>10**(-digits):
                    state_changed=True
            if state_changed: save_state(open_positions,trades_today)
            current_hour=mt5_now.hour; current_minute=mt5_now.minute

            if FORCE_PARITY_SCAN or (current_minute in [2,3] and current_hour in [15,18]):
                for symbol,p in ASSET_PARAMS.items():
                    if not FORCE_PARITY_SCAN and current_hour not in p["hours"]:
                        continue
                    key=f"{symbol}_{mt5_now.date()}_{current_hour}"
                    if key in last_signal_check:
                        continue

                    if symbol in open_positions:
                        write_parity_row(
                            parity_file=PARITY_FILE,
                            bot_version="legacy",
                            symbol=symbol,
                            signal_hour=current_hour,
                            mt5_time=mt5_now,
                            can_eval=False,
                            eval_reason="ya existe posición abierta en este símbolo",
                            signal_found=False,
                            signal_payload=None,
                            exec_allowed=False,
                            exec_reason="not_applicable",
                        )
                        last_signal_check[key] = True
                        continue

                    log.info(
                        f"🔍 Revisando vela {current_hour}:00 MT5 (scan {mt5_now.strftime('%H:%M')}) | capital=${INITIAL_CAPITAL} | symbol={symbol}")

                    sig = check_signal(symbol, p, trades_today.get(symbol, 0), current_hour)

                    if sig:
                        log.info(
                            f"⚡ {sig['stype']} | lots={sig['lots']} | sl={sig['sl']:.{p['digits']}f} | lrr={sig['lrr']:.2f} | spread_signal={sig['spread_signal']:.{p['digits']}f} | tv={sig['signal_tick_volume']:.1f} vs vm={VOLUME_MEANS.get(symbol, 0.0):.1f}")

                        write_parity_row(
                            parity_file=PARITY_FILE,
                            bot_version="legacy",
                            symbol=symbol,
                            signal_hour=current_hour,
                            mt5_time=mt5_now,
                            can_eval=True,
                            eval_reason="ok",
                            signal_found=True,
                            signal_payload=sig,
                            exec_allowed=False,
                            exec_reason="comparison_mode",
                        )

                        if PARITY_ONLY_MODE:
                            last_signal_check[key] = True
                            continue

                        pos = open_position(symbol, sig["direction"], sig["lots"], sig["sl"], p, sig["ep"],
                                            sig["spread_signal"], sig["stype"])
                        if pos:
                            pos["atr"] = sig["atr"]
                            open_positions[symbol] = pos
                            trades_today[symbol] = trades_today.get(symbol, 0) + 1
                            save_state(open_positions, trades_today)

                    else:
                        log.info("— Sin señal")
                        write_parity_row(
                            parity_file=PARITY_FILE,
                            bot_version="legacy",
                            symbol=symbol,
                            signal_hour=current_hour,
                            mt5_time=mt5_now,
                            can_eval=True,
                            eval_reason="ok",
                            signal_found=False,
                            signal_payload=None,
                            exec_allowed=False,
                            exec_reason="no_signal",
                        )

                    last_signal_check[key] = True
            if current_minute==0 and now.second<30:
                info=mt5.account_info(); bal=info.balance if info else 0.0
                log.info(f"[{mt5_now.strftime('%H:%M')} MT5] Balance=${bal:.2f} | Posiciones: {len(open_positions)} | Trades hoy: {dict(trades_today)}")
            time.sleep(TRAILING_CHECK_SECONDS)
        except KeyboardInterrupt:
            log.info("Bot detenido manualmente."); save_state(open_positions,trades_today); break
        except Exception as e:
            log.error(f"Error inesperado: {e}",exc_info=True); time.sleep(30)
            if not mt5.terminal_info(): log.warning("MT5 desconectado. Reconectando..."); connect()
    mt5.shutdown(); log.info("Bot finalizado()")
if __name__=="__main__": main()
