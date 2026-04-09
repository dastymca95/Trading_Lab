import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

try:
    from .backtest_config import BOOTSTRAP_RUNS, MONTE_CARLO_RUNS, ROLLING_SHARPE_WIN
except ImportError:
    from backtest_config import BOOTSTRAP_RUNS, MONTE_CARLO_RUNS, ROLLING_SHARPE_WIN


# ═══════════════════════════════════════════════════════════════════════
# ESTADÍSTICAS
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_mean_ci(x, runs=BOOTSTRAP_RUNS, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if len(x) == 0: return 0.0, 0.0
    means = np.array([np.random.choice(x, len(x), replace=True).mean() for _ in range(runs)])
    return float(np.quantile(means, alpha/2)), float(np.quantile(means, 1-alpha/2))


def sign_test_pvalue(x):
    x = np.asarray(x, dtype=float); x = x[x != 0]
    n = len(x)
    if n == 0: return 1.0
    return float(scipy_stats.binomtest(int((x>0).sum()), n=n, p=0.5, alternative='two-sided').pvalue)


def monte_carlo_dd(pnl_arr, cap0, runs=MONTE_CARLO_RUNS):
    pnl_arr = np.asarray(pnl_arr, dtype=float)
    if len(pnl_arr) == 0:
        return 0.0, 0.0, 0.0, cap0

    dds = np.empty(runs)
    finals = np.empty(runs)

    for i in range(runs):
        shuffled = np.random.permutation(pnl_arr)
        cap = cap0
        eq = [cap]
        for p in shuffled:
            cap = max(cap + p, 0.01)
            eq.append(cap)
        eq = np.array(eq)
        pk = np.maximum.accumulate(eq)
        dds[i] = ((eq - pk) / pk * 100).min()
        finals[i] = eq[-1]

    return (
        float(np.quantile(dds, 0.05)),
        float(np.median(dds)),
        float(np.quantile(dds, 0.95)),
        float(np.median(finals)),
    )


def _resolve_sample_bounds(t, sample_start=None, sample_end=None):
    trade_dates = pd.to_datetime(t['Date']).dt.normalize()
    start = sample_start or t.attrs.get('sample_start') or trade_dates.min()
    end = sample_end or t.attrs.get('sample_end') or trade_dates.max()
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        end = start
    return start, end


def _business_day_index(sample_start, sample_end):
    return pd.bdate_range(sample_start, sample_end)


def daily_equity(t, cap0, sample_start=None, sample_end=None):
    if len(t) == 0:
        return pd.DataFrame(columns=['Date','DailyPnL','Equity','ReturnPct','DrawdownPct'])
    sample_start, sample_end = _resolve_sample_bounds(t, sample_start, sample_end)
    idx = _business_day_index(sample_start, sample_end)
    daily_pnl = (
        t.groupby(pd.to_datetime(t['Date']).dt.normalize())['PnL Neto USD']
        .sum()
        .reindex(idx, fill_value=0.0)
    )
    d = daily_pnl.rename_axis('Date').reset_index(name='DailyPnL')
    d['Date'] = d['Date'].dt.strftime('%Y-%m-%d')
    d['Equity']     = cap0 + d['DailyPnL'].cumsum()
    d['ReturnPct']  = d['DailyPnL'] / cap0 * 100.0
    pk = d['Equity'].cummax()
    d['DrawdownPct'] = (d['Equity'] - pk) / pk * 100.0
    return d


def rolling_sharpe(t, cap0, window=ROLLING_SHARPE_WIN, sample_start=None, sample_end=None):
    d = daily_equity(t, cap0, sample_start=sample_start, sample_end=sample_end)
    if len(d) == 0: return pd.DataFrame()
    r = d['ReturnPct'] / 100.0
    d['RollingSharpe'] = (r.rolling(window).mean() / r.rolling(window).std(ddof=1)) * np.sqrt(252)
    return d[['Date','ReturnPct','RollingSharpe']]


def monthly_heatmap(t):
    if len(t) == 0: return pd.DataFrame()
    m = t.copy()
    m['Year'] = pd.to_datetime(m['Date']).dt.year
    m['MonthNum'] = pd.to_datetime(m['Date']).dt.month
    heat = m.pivot_table(index='Year', columns='MonthNum',
                         values='PnL Neto USD', aggfunc='sum', fill_value=0.0)
    names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
             7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    return heat.rename(columns=names).reset_index()


def return_distribution(t, cap0, sample_start=None, sample_end=None):
    d = daily_equity(t, cap0, sample_start=sample_start, sample_end=sample_end)
    if len(d) == 0: return pd.DataFrame(columns=['Metric','Value'])
    x = d['ReturnPct'].values
    rows = [
        ('CountDays',   len(x)),
        ('MeanDailyPct',   np.mean(x)),
        ('MedianDailyPct', np.median(x)),
        ('StdDailyPct',    np.std(x,ddof=1) if len(x)>1 else 0.0),
        ('SkewDaily',      scipy_stats.skew(x,bias=False) if len(x)>2 else 0.0),
        ('KurtosisDaily',  scipy_stats.kurtosis(x,fisher=True,bias=False) if len(x)>3 else 0.0),
        ('Pct05',  np.quantile(x,.05)), ('Pct25', np.quantile(x,.25)),
        ('Pct50',  np.quantile(x,.50)), ('Pct75', np.quantile(x,.75)),
        ('Pct95',  np.quantile(x,.95)),
        ('WorstDayPct', np.min(x)),     ('BestDayPct',  np.max(x)),
    ]
    return pd.DataFrame(rows, columns=['Metric','Value'])


def calc_metrics(t, e, cap0, sample_start=None, sample_end=None):
    if len(t) == 0:
        return {}

    pk = np.maximum.accumulate(e)
    mdd = ((e - pk) / pk * 100).min()

    w  = t[t['Resultado'] == 'WIN']
    lo = t[t['Resultado'] == 'LOSS']
    be = t[t['Resultado'] == 'BE']

    ret = (e[-1] - e[0]) / e[0] * 100
    wr  = len(w) / len(t) * 100

    gp = w['PnL Neto USD'].sum() if len(w) > 0 else 0
    gl = abs(lo['PnL Neto USD'].sum()) if len(lo) > 0 else 0.001
    pf = gp / gl

    sample_start, sample_end = _resolve_sample_bounds(t, sample_start, sample_end)
    day_index = _business_day_index(sample_start, sample_end)
    days = len(day_index)
    aw = w['PnL Neto USD'].mean() if len(w) > 0 else 0
    al = lo['PnL Neto USD'].mean() if len(lo) > 0 else 0
    exp = t['PnL Neto USD'].mean()

    month_index = pd.period_range(sample_start, sample_end, freq='M')
    monthly = (
        t.groupby(pd.to_datetime(t['Date']).dt.to_period('M'))['PnL Neto USD']
        .sum()
        .reindex(month_index, fill_value=0.0)
    )
    monthly.index = monthly.index.astype(str)
    mret = monthly / cap0 * 100

    mret_std = mret.std(ddof=1)
    sharpe = (
        mret.mean() / mret_std * np.sqrt(12)
        if len(mret) > 1 and pd.notna(mret_std) and mret_std > 0
        else np.nan
    )

    downside = np.minimum(mret, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2)) if len(mret) > 0 else np.nan
    sortino = (
        mret.mean() / downside_dev * np.sqrt(12)
        if pd.notna(downside_dev) and downside_dev > 0
        else np.nan
    )

    ann = (
        ((e[-1] / e[0]) ** (252 / days) - 1) * 100
        if days > 0 and e[0] > 0 and e[-1] > 0
        else 0
    )
    calmar = ann / abs(mdd) if mdd != 0 else 0

    daily_pnl = (
        t.groupby(pd.to_datetime(t['Date']).dt.normalize())['PnL Neto USD']
        .sum()
        .reindex(day_index, fill_value=0.0)
        .sort_index()
    )
    t_p = scipy_stats.ttest_1samp(daily_pnl, 0).pvalue if len(daily_pnl) > 1 else 1.0
    sign_p = sign_test_pvalue(daily_pnl.values)
    ci_lo, ci_hi = bootstrap_mean_ci(daily_pnl.values)

    mc_dd_p5, mc_dd_p50, mc_dd_p95, mc_final = monte_carlo_dd(
        t['PnL Neto USD'].values, cap0
    )

    worst_day_usd = round(float(daily_pnl.min()), 2) if len(daily_pnl) > 0 else 0.0
    best_day_usd  = round(float(daily_pnl.max()), 2) if len(daily_pnl) > 0 else 0.0

    ms = 0
    cur = 0
    for r in t['Resultado']:
        if r == 'LOSS':
            cur += 1
            ms = max(ms, cur)
        else:
            cur = 0

    return {
        'n': len(t), 'n_win': len(w), 'n_loss': len(lo), 'n_be': len(be),
        'wr': round(wr, 1), 'ret': round(ret, 2), 'final': round(e[-1], 2),
        'mdd': round(mdd, 2), 'pf': round(pf, 2),
        'sharpe': round(float(sharpe), 3) if pd.notna(sharpe) else np.nan,
        'sortino': round(float(sortino), 3) if pd.notna(sortino) else np.nan,
        'calmar': round(calmar, 3), 'aw': round(aw, 2), 'al': round(al, 2),
        'rr': round(abs(aw / al), 3) if al != 0 else 0,
        'exp': round(exp, 4), 'p_val': round(float(t_p), 6),
        'sign_p': round(float(sign_p), 6),
        'boot_lo': round(float(ci_lo), 4), 'boot_hi': round(float(ci_hi), 4),
        'mc_dd_p5': round(float(mc_dd_p5), 2),
        'mc_dd_p50': round(float(mc_dd_p50), 2),
        'mc_dd_p95': round(float(mc_dd_p95), 2),
        'mc_final_p50': round(float(mc_final), 2),
        'ms': ms, 'days': days, 'avg_day': round(ret / days, 3) if days > 0 else 0,
        'worst_day_usd': worst_day_usd,
        'best_day_usd': best_day_usd,
        'total_comm': round(t['Comisión USD'].sum(), 2),
        'total_raw': round(t['PnL Bruto USD'].sum(), 2),
        'monthly': monthly, 'monthly_ret': mret, 'equity': e,
    }
