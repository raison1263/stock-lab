"""
stock_lab.py — 개인용 주식 분석 엔진
====================================================
데이터 로딩 · 기술적 지표 · 전략 · 백테스트(손절/익절) · 멀티종목 · 스크리닝 · 머신러닝.
화면(대시보드)은 app.py 에서 이 엔진을 불러 씁니다.

[설치]
    pip install finance-datareader pandas numpy scikit-learn

[티커]
    한국: '005930'(삼성전자) · 미국: 'AAPL'

[주의]
    분석/학습용. 백테스트 성과는 미래를 보장하지 않으며,
    과최적화·거래비용·데이터 누수를 항상 의심해야 합니다.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

try:
    import FinanceDataReader as fdr
except ImportError:
    fdr = None


# ======================================================================
# 1. 데이터
# ======================================================================
def load_data(ticker: str, start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    if fdr is None:
        raise ImportError("pip install finance-datareader 를 먼저 실행하세요.")
    df = fdr.DataReader(ticker, start, end).rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna()
    df.index = pd.to_datetime(df.index)
    return df


# ======================================================================
# 2. 지표
# ======================================================================
def sma(s, w): return s.rolling(w).mean()
def ema(s, span): return s.ewm(span=span, adjust=False).mean()


def rsi(s, period=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(s, fast=12, slow=26, signal=9):
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(s, w=20, k=2.0):
    mid = sma(s, w); std = s.rolling(w).std()
    return mid + k*std, mid, mid - k*std


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(); c = out["Close"]
    out["SMA20"], out["SMA60"] = sma(c, 20), sma(c, 60)
    out["RSI14"] = rsi(c, 14)
    out["MACD"], out["MACD_signal"], out["MACD_hist"] = macd(c)
    out["BB_up"], out["BB_mid"], out["BB_low"] = bollinger(c)
    return out


# ======================================================================
# 3. 전략 — 1=보유 의도, 0=현금. 모두 '당일 종가까지'의 정보만 사용(인과적).
# ======================================================================
def strat_sma_cross(df, fast=20, slow=60):
    return (sma(df["Close"], fast) > sma(df["Close"], slow)).astype(int)


def strat_rsi(df, low=30, high=70):
    r = rsi(df["Close"], 14)
    s = pd.Series(np.nan, index=df.index)
    s[r < low] = 1; s[r > high] = 0
    return s.ffill().fillna(0).astype(int)


def strat_macd(df):
    line, sig, _ = macd(df["Close"])
    return (line > sig).astype(int)


def strat_bollinger(df, low=30):
    """가격이 하단밴드 아래로 이탈 후 중심선 회복까지 보유(평균회귀)."""
    up, mid, lo = bollinger(df["Close"])
    c = df["Close"]
    s = pd.Series(np.nan, index=df.index)
    s[c < lo] = 1; s[c > mid] = 0
    return s.ffill().fillna(0).astype(int)


STRATEGIES = {
    "SMA 20-60 교차": strat_sma_cross,
    "RSI 과매도/과매수": strat_rsi,
    "MACD 교차": strat_macd,
    "볼린저 평균회귀": strat_bollinger,
}


def combine_signals(df, names, mode="AND"):
    """여러 전략 신호를 AND(모두 충족) 또는 OR(하나라도 충족)로 결합."""
    sigs = [STRATEGIES[n](df) for n in names]
    stacked = pd.concat(sigs, axis=1).fillna(0)
    return (stacked.min(axis=1) if mode == "AND" else stacked.max(axis=1)).astype(int)


# ======================================================================
# 4. 백테스트 (손절/익절 지원, 경로의존적 → 일별 루프)
# ======================================================================
def backtest(df, signal, fee=0.0015, stop_loss=None, take_profit=None):
    """
    signal : 1=보유 의도. look-ahead 방지로 '다음 날'부터 진입/청산.
    fee    : 포지션 변경 시 편도 거래비용(기본 0.15%).
    stop_loss/take_profit : 진입가 대비 비율(예: 0.05 = -5% 손절 / +5% 익절). None이면 미사용.
    """
    df = df.copy()
    close = df["Close"].values
    high = df["High"].values if "High" in df else close
    low = df["Low"].values if "Low" in df else close
    sig = signal.reindex(df.index).fillna(0).astype(int).values

    n = len(df)
    pos = np.zeros(n)          # 당일 실제 포지션(0/1)
    strat_ret = np.zeros(n)
    in_pos = False
    entry = 0.0

    for t in range(1, n):
        prev_close = close[t-1]
        if in_pos:
            day_ret = close[t] / prev_close - 1
            exited = False
            # 손절/익절은 당일 고저가로 체크(보수적으로 손절 우선)
            if stop_loss is not None and low[t] <= entry * (1 - stop_loss):
                day_ret = (entry * (1 - stop_loss)) / prev_close - 1
                in_pos, exited = False, True
            elif take_profit is not None and high[t] >= entry * (1 + take_profit):
                day_ret = (entry * (1 + take_profit)) / prev_close - 1
                in_pos, exited = False, True
            elif sig[t-1] == 0:        # 전략 청산 신호
                in_pos, exited = False, True
            strat_ret[t] = day_ret - (fee if exited else 0)
            pos[t] = 0 if exited else 1
        else:
            if sig[t-1] == 1:          # 진입
                in_pos, entry = True, close[t-1]
                strat_ret[t] = (close[t] / prev_close - 1) - fee
                pos[t] = 1

    df["pos"] = pos
    df["strat_ret"] = strat_ret
    df["ret"] = df["Close"].pct_change().fillna(0)
    df["equity"] = (1 + df["strat_ret"]).cumprod()
    df["buyhold"] = (1 + df["ret"]).cumprod()
    return df


def performance(bt) -> dict:
    eq, r = bt["equity"], bt["strat_ret"]
    years = max(len(bt) / 252, 1e-9)
    cagr = eq.iloc[-1] ** (1/years) - 1
    sharpe = r.mean()/r.std()*np.sqrt(252) if r.std() > 0 else np.nan
    mdd = (eq/eq.cummax() - 1).min()
    trades = int((pd.Series(bt["pos"]).diff() == 1).sum())
    nz = bt.loc[bt["strat_ret"] != 0, "strat_ret"]
    wins = (nz > 0).mean() if len(nz) else np.nan
    return {
        "전략 누적수익률": eq.iloc[-1] - 1,
        "단순보유 누적수익률": bt["buyhold"].iloc[-1] - 1,
        "CAGR": cagr,
        "샤프지수": sharpe,
        "최대낙폭(MDD)": mdd,
        "거래횟수": trades,
        "승률(거래일 기준)": wins,
    }


# ======================================================================
# 5. 멀티 종목 백테스트
# ======================================================================
def multi_backtest(tickers, strategy_fn, start="2021-01-01", **bt_kwargs):
    """여러 종목에 같은 전략을 적용해 성과표를 반환."""
    rows = {}
    for tk in tickers:
        try:
            df = add_indicators(load_data(tk, start=start))
            bt = backtest(df, strategy_fn(df), **bt_kwargs)
            rows[tk] = performance(bt)
        except Exception as e:
            rows[tk] = {"오류": str(e)}
    return pd.DataFrame(rows).T


# ======================================================================
# 6. 스크리닝 — 오늘 시점의 조건 충족 종목 찾기
# ======================================================================
def screen(tickers, start="2024-01-01"):
    """관심 종목들의 현재 지표 상태와 시그널 발생 여부를 표로 반환."""
    out = []
    for tk in tickers:
        try:
            df = add_indicators(load_data(tk, start=start))
            last, prev = df.iloc[-1], df.iloc[-2]
            golden = prev["SMA20"] <= prev["SMA60"] and last["SMA20"] > last["SMA60"]
            out.append({
                "종목": tk,
                "종가": round(float(last["Close"]), 2),
                "RSI14": round(float(last["RSI14"]), 1),
                "과매도(<30)": bool(last["RSI14"] < 30),
                "과매수(>70)": bool(last["RSI14"] > 70),
                "골든크로스(오늘)": bool(golden),
                "MACD>시그널": bool(last["MACD"] > last["MACD_signal"]),
                "밴드하단이탈": bool(last["Close"] < last["BB_low"]),
            })
        except Exception as e:
            out.append({"종목": tk, "종가": f"오류: {e}"})
    return pd.DataFrame(out)


# ======================================================================
# 7. 머신러닝 — 다음 날 상승/하락 확률 (데이터 누수 방지 포함)
# ======================================================================
def build_features(df: pd.DataFrame, horizon: int = 1):
    """지표 기반 피처. 모두 '당일까지'의 정보만 사용하므로 인과적(미래정보 없음).
    horizon: 며칠 뒤 상승 여부를 맞힐지(1=다음 날, 63≈3개월)."""
    d = add_indicators(df).copy()
    c = d["Close"]
    feat = pd.DataFrame(index=d.index)
    feat["ret1"] = c.pct_change()
    feat["ret5"] = c.pct_change(5)
    feat["ret20"] = c.pct_change(20)
    feat["rsi"] = d["RSI14"]
    feat["macd_hist"] = d["MACD_hist"]
    feat["bb_pos"] = (c - d["BB_low"]) / (d["BB_up"] - d["BB_low"])   # 밴드 내 위치 0~1
    feat["sma_gap"] = d["SMA20"] / d["SMA60"] - 1
    feat["vol_chg"] = d["Volume"].pct_change() if "Volume" in d else 0
    # 타깃: horizon일 뒤 종가가 오르면 1 (shift(-h) → 미래는 '정답'으로만 사용)
    target = (c.shift(-horizon) > c).astype(int)
    data = feat.join(target.rename("target")).replace([np.inf, -np.inf], np.nan).dropna()
    return data.drop(columns="target"), data["target"]


def train_predict(df, model_name="RandomForest", test_ratio=0.3, horizon=1):
    """
    시계열 순서를 지켜 앞부분으로 학습, 뒷부분으로 검증(셔플 금지=누수 방지).
    스케일러도 학습 구간에만 fit. 반환: 성과 dict + 최신일 상승확률.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X, y = build_features(df, horizon=horizon)
    n = len(X)
    if n < 200:
        raise ValueError("데이터가 너무 적습니다(200일 이상 권장).")
    split = int(n * (1 - test_ratio))
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler().fit(Xtr)          # 학습 구간에만 fit
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    if model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=4,
                                       min_samples_leaf=20, random_state=0)
    model.fit(Xtr_s, ytr)

    pred = model.predict(Xte_s)
    acc = accuracy_score(yte, pred)
    baseline = max(yte.mean(), 1 - yte.mean())   # 다수 클래스 항상 찍기
    latest_prob = float(model.predict_proba(scaler.transform(X.iloc[[-1]]))[0, 1])

    return {
        "모델": model_name,
        "예측기간(일)": horizon,
        "검증 정확도": acc,
        "기준선(다수클래스)": baseline,
        "정확도-기준선": acc - baseline,
        "학습/검증 표본": f"{split} / {n - split}",
        "상승확률": latest_prob,
    }


def project_cone(df, horizon=63, n_paths=3000, lookback=756, seed=0, drift="keep"):
    """
    과거 일간 수익률을 부트스트랩 재표집해 horizon일 뒤까지의 가격 분포(부채꼴)를 추정.
    - 정규분포를 가정하지 않고 '실제 과거 수익률'을 섞어 굴리므로 꼬리위험도 일부 반영.
    - drift="keep": 과거 추세를 그대로 연장(장기엔 비현실적일 수 있음).
      drift="zero": 추세를 제거하고 변동성만 반영(중앙값이 평평) → 장기 예측에 더 안전.
    - lookback: 사용할 과거 일수. None이면 가능한 전체 기간 사용(여러 국면 포함).
    반환: 미래 영업일 인덱스의 p5/p25/p50/p75/p95 밴드, 시작가 S0, horizon 시점 상승확률.
    """
    logret = np.log(df["Close"]).diff().dropna()
    if lookback:
        logret = logret.tail(lookback)
    logret = logret.values
    if len(logret) < 60:
        raise ValueError("데이터가 부족합니다(최소 60일).")
    if drift == "zero":
        logret = logret - logret.mean()          # 추세 제거 → 변동성만
    S0 = float(df["Close"].iloc[-1])
    rng = np.random.default_rng(seed)
    draws = rng.choice(logret, size=(n_paths, horizon), replace=True)
    paths = S0 * np.exp(np.cumsum(draws, axis=1))           # (n_paths, horizon)
    pct = {f"p{p}": np.percentile(paths, p, axis=0) for p in (5, 25, 50, 75, 95)}
    future_idx = pd.bdate_range(df.index[-1], periods=horizon + 1)[1:]
    band = pd.DataFrame(pct, index=future_idx)
    prob_up = float((paths[:, -1] > S0).mean())             # horizon 시점 상승 확률
    return band, S0, prob_up


# ======================================================================
# 8. 종목 검색 (이름 ↔ 코드)
# ======================================================================
def _norm_listing(df, market):
    cols = {c.lower(): c for c in df.columns}
    code_col = cols.get("symbol") or cols.get("code")
    name_col = cols.get("name")
    out = df[[code_col, name_col]].copy()
    out.columns = ["ticker", "name"]
    out["market"] = market
    return out


def load_listings() -> pd.DataFrame:
    """한국(KRX) + 미국(NASDAQ/NYSE) 전체 종목 목록(코드·이름)을 합쳐 반환."""
    if fdr is None:
        raise ImportError("pip install finance-datareader 를 먼저 실행하세요.")
    frames = []
    for fetch, market in [("KRX", "KR"), ("NASDAQ", "US"), ("NYSE", "US")]:
        try:
            frames.append(_norm_listing(fdr.StockListing(fetch), market))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(columns=["ticker", "name", "market"])
    df = pd.concat(frames, ignore_index=True).dropna(subset=["ticker", "name"])
    df["ticker"] = df["ticker"].astype(str)
    return df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def search_ticker(listings: pd.DataFrame, query: str, limit: int = 15) -> pd.DataFrame:
    """이름 또는 코드로 종목 검색. 부분 일치 결과를 표로 반환."""
    if listings is None or listings.empty:
        return pd.DataFrame(columns=["ticker", "name", "market"])
    q = str(query).strip().lower()
    if not q:
        return listings.iloc[0:0]
    name_hit = listings["name"].astype(str).str.lower().str.contains(q, na=False)
    code_hit = listings["ticker"].astype(str).str.lower().str.contains(q, na=False)
    return listings[name_hit | code_hit].head(limit)


def get_name(listings: pd.DataFrame, ticker: str) -> str:
    """코드로 종목명을 찾음. 없으면 코드 그대로 반환."""
    if listings is None or listings.empty:
        return ticker
    hit = listings[listings["ticker"].astype(str) == str(ticker)]
    return str(hit["name"].iloc[0]) if len(hit) else ticker


def market_of(listings: pd.DataFrame, ticker: str) -> str:
    if listings is None or listings.empty:
        return "US" if str(ticker)[:1].isalpha() else "KR"
    hit = listings[listings["ticker"].astype(str) == str(ticker)]
    return str(hit["market"].iloc[0]) if len(hit) else ("US" if str(ticker)[:1].isalpha() else "KR")


# ======================================================================
# 9. CLI 데모
# ======================================================================
def main():
    for tk, nm in [("005930", "삼성전자"), ("AAPL", "Apple")]:
        df = add_indicators(load_data(tk, start="2021-01-01"))
        bt = backtest(df, strat_sma_cross(df), stop_loss=0.07, take_profit=0.15)
        print(f"\n=== {nm} / SMA교차 +손절7%/익절15% ===")
        for k, v in performance(bt).items():
            print(f"  {k:<16}: {v:.3f}" if isinstance(v, float) else f"  {k:<16}: {v}")


if __name__ == "__main__":
    main()
