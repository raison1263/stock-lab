"""
app.py — 주식 분석 대시보드 (토스 증권 스타일 · 모바일 최적화)
====================================================
[로컬 실행]
    pip install -r requirements.txt
    streamlit run app.py

[폰에서 어디서나 보기]
    README.md 의 '폰에서 보기' 안내(Streamlit Cloud 무료 배포)를 참고하세요.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import stock_lab as sl

st.set_page_config(page_title="내 증권", page_icon="📈",
                   layout="centered", initial_sidebar_state="collapsed")

# 한국식 색상: 상승=빨강, 하락=파랑 (토스와 동일)
UP, DOWN, INK, SUB, BLUE = "#F04452", "#3182F6", "#191F28", "#8B95A1", "#3182F6"

st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
html, body, [class*="css"], .stMarkdown, .stButton button { font-family:'Pretendard',-apple-system,sans-serif; }
#MainMenu, header, footer { visibility:hidden; }
.block-container { padding:0.8rem 1rem 3rem; max-width:460px; }
.toss-card { background:#fff; border:1px solid #F2F4F6; border-radius:18px;
             padding:20px; box-shadow:0 1px 6px rgba(0,0,0,0.05); margin-bottom:14px; }
.t-name { font-size:19px; font-weight:700; color:#191F28; }
.t-code { font-size:13px; color:#8B95A1; margin-top:3px; }
.t-price { font-size:34px; font-weight:800; color:#191F28; margin-top:16px; letter-spacing:-0.5px; }
.t-change { font-size:15px; font-weight:600; margin-top:5px; }
.t-row { display:flex; justify-content:space-between; padding:9px 0;
         border-bottom:1px solid #F2F4F6; font-size:14px; }
.t-row .k { color:#8B95A1; } .t-row .v { color:#191F28; font-weight:600; }
.sig-on { color:#fff; background:#F04452; border-radius:6px; padding:2px 8px; font-size:12px; font-weight:600; }
.sig-off { color:#B0B8C1; font-size:13px; }
.stButton button { border-radius:12px; }
div[data-baseweb="tab-list"] { gap:4px; }
button[data-baseweb="tab"] { font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ===== (선택) 비밀번호 잠금 — secrets에 app_password 설정 시에만 작동 =====
def check_password():
    try:
        required = st.secrets.get("app_password", None)
    except Exception:
        required = None
    if not required:
        return True
    if st.session_state.get("auth_ok"):
        return True
    st.markdown("### 🔒 잠금")
    pw = st.text_input("비밀번호", type="password", label_visibility="collapsed",
                       placeholder="비밀번호 입력")
    if pw == required:
        st.session_state.auth_ok = True
        st.rerun()
    elif pw:
        st.error("비밀번호가 틀렸습니다.")
    return False


if not check_password():
    st.stop()


# ===== 데이터 =====
@st.cache_data(ttl=86400, show_spinner="종목 목록 불러오는 중…")
def listings():
    return sl.load_listings()


@st.cache_data(ttl=1800, show_spinner=False)
def get_data(ticker):
    return sl.add_indicators(sl.load_data(ticker, start="2016-01-01"))


def fmt_price(p, market):
    return f"{p:,.0f}원" if market == "KR" else f"${p:,.2f}"


def fmt_pct(x):
    return f"{x:.1%}" if isinstance(x, (int, float, np.floating)) and not pd.isna(x) else x


LST = listings()
if "ticker" not in st.session_state:
    st.session_state.ticker = "005930"

# ===== 검색 =====
q = st.text_input("종목 검색", placeholder="🔍 삼성전자, AAPL, 카카오…",
                  label_visibility="collapsed")
if q:
    res = sl.search_ticker(LST, q, limit=8)
    if len(res):
        for _, r in res.iterrows():
            flag = "🇰🇷" if r["market"] == "KR" else "🇺🇸"
            if st.button(f"{flag}  {r['name']}  ·  {r['ticker']}",
                         key=f"pick_{r['ticker']}", width='stretch'):
                st.session_state.ticker = r["ticker"]
                st.rerun()
    else:
        st.caption("검색 결과가 없어요. 코드(예: 005930)나 영문 심볼(예: AAPL)로도 검색해 보세요.")

ticker = st.session_state.ticker
name = sl.get_name(LST, ticker)
market = sl.market_of(LST, ticker)

# ===== 현재가 카드 =====
try:
    df = get_data(ticker)
    last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
    chg = last - prev
    chg_pct = chg / prev * 100
    color = UP if chg >= 0 else DOWN
    sign = "+" if chg >= 0 else ""
    arrow = "▲" if chg >= 0 else "▼"
    st.markdown(f"""
    <div class="toss-card">
      <div class="t-name">{name}</div>
      <div class="t-code">{ticker} · {'코스피/코스닥' if market=='KR' else '미국'}</div>
      <div class="t-price">{fmt_price(last, market)}</div>
      <div class="t-change" style="color:{color}">{arrow} {sign}{fmt_price(abs(chg), market)} ({sign}{chg_pct:.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()


tab_chart, tab_bt, tab_screen, tab_ml = st.tabs(["📈 차트", "🧪 백테스트", "🔍 신호", "🤖 예측"])


# ----- 차트 (토스풍 라인차트 + 기간 버튼) -----
with tab_chart:
    period = st.radio("기간", ["1주", "1달", "3달", "1년", "5년"],
                      index=2, horizontal=True, label_visibility="collapsed")
    days = {"1주": 5, "1달": 21, "3달": 63, "1년": 252, "5년": 1260}[period]
    d = df.tail(days)
    up = d["Close"].iloc[-1] >= d["Close"].iloc[0]
    line_c = UP if up else DOWN
    pr = (d["Close"].iloc[-1] / d["Close"].iloc[0] - 1) * 100
    lo, hi = d["Close"].min(), d["Close"].max()
    pad = (hi - lo) * 0.15 or hi * 0.05

    fig = go.Figure(go.Scatter(
        x=d.index, y=d["Close"], mode="lines",
        line=dict(color=line_c, width=2.4),
        fill="tozeroy", fillcolor=f"rgba({'240,68,82' if up else '49,130,246'},0.10)",
        hovertemplate="%{y:,.2f}<extra></extra>"))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                      plot_bgcolor="white", paper_bgcolor="white",
                      xaxis=dict(showgrid=False, showticklabels=True, color=SUB),
                      yaxis=dict(range=[lo - pad, hi + pad], showgrid=False,
                                 side="right", color=SUB))
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    st.markdown(f"<div style='text-align:center;color:{line_c};font-weight:700;font-size:15px'>"
                f"{period} 수익률 {'+' if pr>=0 else ''}{pr:.2f}%</div>", unsafe_allow_html=True)

    st.markdown('<div class="toss-card">', unsafe_allow_html=True)
    rows = [("RSI(14)", f"{df['RSI14'].iloc[-1]:.1f}"),
            ("20일 이평", fmt_price(df["SMA20"].iloc[-1], market)),
            ("60일 이평", fmt_price(df["SMA60"].iloc[-1], market)),
            ("52주 최고", fmt_price(df["Close"].tail(252).max(), market)),
            ("52주 최저", fmt_price(df["Close"].tail(252).min(), market))]
    for k, v in rows:
        st.markdown(f'<div class="t-row"><span class="k">{k}</span><span class="v">{v}</span></div>',
                    unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ----- 백테스트 -----
with tab_bt:
    chosen = st.multiselect("전략 (여러 개 = 결합)", list(sl.STRATEGIES.keys()),
                            default=["SMA 20-60 교차"])
    mode = st.radio("결합", ["AND", "OR"], horizontal=True)
    c1, c2 = st.columns(2)
    sl_pct = c1.slider("손절 %", 0, 30, 7)
    tp_pct = c2.slider("익절 %", 0, 50, 0)
    if chosen and st.button("백테스트 실행", width='stretch', type="primary"):
        sig = sl.combine_signals(df, chosen, mode)
        bt = sl.backtest(df, sig, stop_loss=sl_pct/100 or None, take_profit=tp_pct/100 or None)
        p = sl.performance(bt)
        excess = p["전략 누적수익률"] - p["단순보유 누적수익률"]
        st.markdown(f"""<div class="toss-card">
          <div class="t-row"><span class="k">전략 누적수익</span><span class="v" style="color:{UP if p['전략 누적수익률']>=0 else DOWN}">{fmt_pct(p['전략 누적수익률'])}</span></div>
          <div class="t-row"><span class="k">단순보유 누적수익</span><span class="v">{fmt_pct(p['단순보유 누적수익률'])}</span></div>
          <div class="t-row"><span class="k">단순보유 대비</span><span class="v" style="color:{UP if excess>=0 else DOWN}">{'+' if excess>=0 else ''}{fmt_pct(excess)}</span></div>
          <div class="t-row"><span class="k">최대낙폭(MDD)</span><span class="v" style="color:{DOWN}">{fmt_pct(p['최대낙폭(MDD)'])}</span></div>
          <div class="t-row"><span class="k">샤프지수</span><span class="v">{p['샤프지수']:.2f}</span></div>
          <div class="t-row"><span class="k">거래 횟수</span><span class="v">{p['거래횟수']}회</span></div>
        </div>""", unsafe_allow_html=True)
        buys = bt.index[(pd.Series(bt["pos"]).diff() == 1).values]
        sells = bt.index[(pd.Series(bt["pos"]).diff() == -1).values]
        f = go.Figure()
        f.add_trace(go.Scatter(x=bt.index, y=bt["Close"], name="가격",
                               line=dict(color=INK, width=1.4)))
        f.add_trace(go.Scatter(x=buys, y=bt.loc[buys, "Close"], mode="markers", name="매수",
                               marker=dict(symbol="triangle-up", size=10, color=UP)))
        f.add_trace(go.Scatter(x=sells, y=bt.loc[sells, "Close"], mode="markers", name="매도",
                               marker=dict(symbol="triangle-down", size=10, color=DOWN)))
        f.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="white", legend=dict(orientation="h"),
                        xaxis=dict(showgrid=False, color=SUB),
                        yaxis=dict(showgrid=False, side="right", color=SUB))
        st.plotly_chart(f, width='stretch', config={"displayModeBar": False})
        st.caption("⚠️ 과거 성과는 미래를 보장하지 않습니다. 단순보유를 못 이기는 전략이 대부분이에요.")


# ----- 스크리닝 -----
with tab_screen:
    st.caption("관심 종목에 오늘 어떤 신호가 떴는지 확인해요.")
    wl_raw = st.text_area("관심 종목 (쉼표/줄바꿈)",
                          value="005930, 000660, 035720, AAPL, NVDA")
    if st.button("신호 확인", width='stretch', type="primary"):
        wl = [t.strip() for t in wl_raw.replace("\n", ",").split(",") if t.strip()]
        with st.spinner("확인 중…"):
            scr = sl.screen(wl, start="2024-01-01")
        for _, r in scr.iterrows():
            nm = sl.get_name(LST, r["종목"])
            badges = []
            for label, key in [("과매도", "과매도(<30)"), ("골든크로스", "골든크로스(오늘)"),
                               ("MACD강세", "MACD>시그널"), ("밴드이탈", "밴드하단이탈"),
                               ("과매수", "과매수(>70)")]:
                if key in r and r[key] is True:
                    badges.append(f'<span class="sig-on">{label}</span>')
            badge_html = " ".join(badges) if badges else '<span class="sig-off">신호 없음</span>'
            price = r["종가"] if isinstance(r["종가"], str) else f"{r['종가']:,.2f}"
            st.markdown(f"""<div class="toss-card" style="padding:14px 18px">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div><span class="t-name" style="font-size:16px">{nm}</span>
                <span class="t-code">{r['종목']}</span></div>
                <div class="v" style="font-weight:700">{price}</div>
              </div>
              <div style="margin-top:8px">{badge_html}</div>
            </div>""", unsafe_allow_html=True)


# ----- 예측 (기간 선택 + 확률 부채꼴) -----
with tab_ml:
    st.markdown(f"""<div class="toss-card" style="background:#FFF6E9;border-color:#FFE2B8">
      <div style="font-weight:700;color:#C77700">⚠️ 예측의 한계 — 꼭 읽어주세요</div>
      <div style="font-size:13px;color:#7A5A1E;margin-top:6px;line-height:1.5">
      미래는 <b>한 점(가격)으로 맞힐 수 없어</b> '범위(부채꼴)'로 보여드립니다.
      기간이 길수록 띠가 넓어지는 게 정상이에요. '방향 확률'의 정확도는 동전 던지기
      수준이고, 특히 <b>뉴스·실적 충격은 전혀 반영하지 못합니다.</b>
      참고용일 뿐, 이 숫자만으로 매매하지 마세요.</div>
    </div>""", unsafe_allow_html=True)

    hsel = st.selectbox("예측 기간", ["1주", "1달", "3달", "6달", "1년", "3년", "5년"],
                        index=2)
    H = {"1주": 5, "1달": 21, "3달": 63, "6달": 126,
         "1년": 252, "3년": 756, "5년": 1260}[hsel]
    long_h = H >= 252
    zero_drift = st.checkbox("변동성만 보기 (추세 제거)", value=long_h,
                             help="추세를 그대로 5년 곱하면 비현실적이라, 장기엔 추세를 빼고 "
                                  "변동성만 보는 게 더 안전해요.")
    if H <= 63:
        model_name = st.selectbox("방향 예측 모델", ["RandomForest", "LogisticRegression"])

    if st.button("예측 실행", width='stretch', type="primary"):
        try:
            # 1) 확률 부채꼴 (몬테카를로) — 장기엔 전체 기간 사용
            band, S0, mc_prob = sl.project_cone(
                df, horizon=H, lookback=(None if long_h else 756),
                drift=("zero" if zero_drift else "keep"))
            hist = df["Close"].tail(min(120, max(40, H // 4)))
            x_band = [hist.index[-1]] + list(band.index)
            def w0(col): return [S0] + list(band[col])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist, name="실제",
                                     line=dict(color=INK, width=1.8)))
            fig.add_trace(go.Scatter(x=x_band, y=w0("p95"), line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x_band, y=w0("p5"), fill="tonexty",
                                     fillcolor="rgba(49,130,246,0.10)", line=dict(width=0),
                                     name="90% 범위", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x_band, y=w0("p75"), line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x_band, y=w0("p25"), fill="tonexty",
                                     fillcolor="rgba(49,130,246,0.22)", line=dict(width=0),
                                     name="50% 범위", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x_band, y=w0("p50"), name="중앙값",
                                     line=dict(color=BLUE, width=2, dash="dot")))
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                              plot_bgcolor="white", legend=dict(orientation="h"),
                              xaxis=dict(showgrid=False, color=SUB),
                              yaxis=dict(showgrid=False, side="right", color=SUB))
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

            lo, mid, hi = band["p5"].iloc[-1], band["p50"].iloc[-1], band["p95"].iloc[-1]
            mid_label = "중앙값(변동성만, 추세 0)" if zero_drift else "중앙값(추세 연장 가정)"
            st.markdown(f"""<div class="toss-card">
              <div style="font-size:14px;font-weight:700;color:#191F28">{hsel} 뒤 예상 범위</div>
              <div class="t-row"><span class="k">{mid_label}</span><span class="v">{fmt_price(mid, market)} ({(mid/S0-1)*100:+.1f}%)</span></div>
              <div class="t-row"><span class="k">90% 확률 범위</span><span class="v">{fmt_price(lo, market)} ~ {fmt_price(hi, market)}</span></div>
              <div class="t-row"><span class="k">{hsel} 뒤 상승확률(시뮬레이션)</span>
                <span class="v" style="color:{UP if mc_prob>=0.5 else DOWN}">{mc_prob*100:.0f}%</span></div>
            </div>""", unsafe_allow_html=True)
            if long_h:
                st.caption(f"⚠️ {hsel} 부채꼴은 '범위'만 참고하세요. 그 정도 미래의 중앙값·"
                           "방향은 누구도 신뢰성 있게 못 맞힙니다. 띠의 넓이가 핵심 정보예요.")

            # 2) 방향 예측 모델 — 3개월 이하에서만 (그 이상은 검증 불가)
            if H <= 63:
                res = sl.train_predict(df, model_name, test_ratio=0.3, horizon=H)
                edge = res["정확도-기준선"]
                st.markdown(f"""<div class="toss-card">
                  <div style="font-size:14px;font-weight:700;color:#191F28">방향 예측 모델 성적</div>
                  <div class="t-row"><span class="k">검증 정확도</span><span class="v">{fmt_pct(res['검증 정확도'])}</span></div>
                  <div class="t-row"><span class="k">기준선(그냥 찍기)</span><span class="v">{fmt_pct(res['기준선(다수클래스)'])}</span></div>
                  <div class="t-row"><span class="k">예측력(정확도-기준선)</span>
                    <span class="v" style="color:{UP if edge>0.005 else DOWN}">{'+' if edge>0 else ''}{fmt_pct(edge)}</span></div>
                </div>""", unsafe_allow_html=True)
                if edge <= 0.005:
                    st.warning(f"이 종목·{hsel} 기준에선 모델의 예측력이 사실상 없어요"
                               "(그냥 찍기와 비슷). 부채꼴 '범위'만 참고하세요.")
            else:
                st.info("📌 방향 예측 모델은 3개월 이하에서만 제공해요. 그보다 먼 미래는 "
                        "검증할 독립 표본이 거의 없어 정확도 자체가 의미 없어서, 부채꼴 범위만 보여드립니다.")
        except Exception as e:
            st.error(f"오류: {e}")

st.markdown(f"<div style='text-align:center;color:{SUB};font-size:12px;margin-top:20px'>"
            "분석·학습용 도구 · 투자 판단의 책임은 본인에게 있습니다</div>", unsafe_allow_html=True)
