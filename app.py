"""
app.py — 내 증권 (토스 스타일 다크 · 모바일)
====================================================
홈: 관심 종목 목록(오늘 등락률 순) → 종목 탭하면 상세(차트·백테스트·예측).
[실행] pip install -r requirements.txt → streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import stock_lab as sl

st.set_page_config(page_title="내 증권", page_icon="📈",
                   layout="centered", initial_sidebar_state="collapsed")

# 한국식 색상(상승 빨강·하락 파랑) + 다크 팔레트
UP, DOWN, INK, SUB = "#F0616D", "#4D9EFF", "#F2F4F6", "#8B95A1"
CARD, BG, DIV, BLUE = "#1E2127", "#17171C", "#2A2E37", "#4D9EFF"
DEFAULT_WATCH = ["005930", "000660", "035420", "035720", "005380",
                 "AAPL", "NVDA", "MSFT", "TSLA", "GOOGL"]

st.markdown(f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
html, body, [class*="css"], .stMarkdown, .stButton button {{ font-family:'Pretendard',-apple-system,sans-serif; }}
.stApp {{ background:{BG}; }}
#MainMenu, header, footer {{ visibility:hidden; }}
.block-container {{ padding:0.8rem 1rem 3rem; max-width:460px; }}
.card {{ background:{CARD}; border-radius:18px; padding:18px 20px; margin-bottom:12px; }}
.t-name {{ font-size:19px; font-weight:700; color:{INK}; }}
.t-code {{ font-size:13px; color:{SUB}; margin-top:3px; }}
.t-price {{ font-size:34px; font-weight:800; color:{INK}; margin-top:14px; letter-spacing:-0.5px; }}
.t-change {{ font-size:14px; font-weight:600; margin-top:6px; }}
.t-row {{ display:flex; justify-content:space-between; padding:11px 0; border-bottom:1px solid {DIV}; font-size:14px; }}
.t-row .k {{ color:{SUB}; }} .t-row .v {{ color:{INK}; font-weight:600; }}
.li {{ display:flex; align-items:center; padding:13px 2px; border-bottom:1px solid {DIV}; }}
.li-rank {{ width:26px; font-size:15px; font-weight:700; color:{SUB}; }}
.li-mid {{ flex:1; }}
.li-name {{ font-size:16px; font-weight:700; color:{INK}; }}
.li-code {{ font-size:12px; color:{SUB}; margin-top:2px; }}
.li-price {{ font-size:15px; font-weight:700; color:{INK}; text-align:right; }}
.li-chg {{ font-size:13px; font-weight:600; margin-top:2px; text-align:right; }}
.stButton button {{ border-radius:12px; background:{CARD}; color:{INK}; border:1px solid {DIV}; }}
.stButton button:hover {{ border-color:{SUB}; color:{INK}; }}
div[data-baseweb="tab-list"] {{ gap:6px; }} button[data-baseweb="tab"] {{ font-weight:600; }}
h1,h2,h3,h4 {{ color:{INK}; }}
</style>
""", unsafe_allow_html=True)


def check_password():
    try:
        required = st.secrets.get("app_password", None)
    except Exception:
        required = None
    if not required or st.session_state.get("auth_ok"):
        return True
    st.markdown("### 🔒 잠금")
    pw = st.text_input("비밀번호", type="password", label_visibility="collapsed",
                       placeholder="비밀번호 입력")
    if pw == required:
        st.session_state.auth_ok = True; st.rerun()
    elif pw:
        st.error("비밀번호가 틀렸습니다.")
    return False


if not check_password():
    st.stop()


@st.cache_data(ttl=86400, show_spinner="종목 목록 불러오는 중…")
def listings():
    return sl.load_listings()


@st.cache_data(ttl=1800, show_spinner=False)
def get_data(ticker):
    return sl.add_indicators(sl.load_data(ticker, start="2016-01-01"))


@st.cache_data(ttl=600, show_spinner=False)
def get_quote(ticker):
    return sl.quote(ticker)


@st.cache_data(ttl=600, show_spinner=False)
def get_indices():
    return sl.index_quotes()


@st.cache_data(ttl=600, show_spinner="시장 순위 불러오는 중…")
def get_rankings_cached(scope):
    markets = {"국내": {"KR"}, "해외": {"US"}}.get(scope)
    return sl.market_rankings(get_quote, LST, markets=markets)


def fmt_price(p, market):
    return f"{p:,.0f}원" if market == "KR" else f"${p:,.2f}"


def fmt_pct(x):
    return f"{x:.1%}" if isinstance(x, (int, float, np.floating)) and not pd.isna(x) else x


def fmt_value(v, market):
    if v is None:
        return "-"
    if market == "KR":
        if v >= 1e12: return f"{v/1e12:.1f}조원"
        if v >= 1e8: return f"{v/1e8:.0f}억원"
        return f"{v:,.0f}원"
    if v >= 1e9: return f"${v/1e9:.1f}B"
    if v >= 1e6: return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"


def nav():
    c1, c2 = st.columns(2)
    if c1.button("⭐ 관심", width='stretch',
                 type=("primary" if ss.view == "home" else "secondary")):
        ss.view = "home"; st.rerun()
    if c2.button("🔍 발견", width='stretch',
                 type=("primary" if ss.view == "discover" else "secondary")):
        ss.view = "discover"; st.rerun()


def render_index_bar():
    idx = get_indices()
    if not idx:
        return
    cells = ""
    for it in idx:
        c = UP if it["pct"] >= 0 else DOWN
        sg = "+" if it["pct"] >= 0 else ""
        val = f"{it['last']:,.1f}" if it["name"] == "환율" else f"{it['last']:,.2f}"
        cells += (f'<div style="flex:0 0 auto;min-width:88px">'
                  f'<div style="font-size:11px;color:{SUB}">{it["name"]}</div>'
                  f'<div style="font-size:13px;font-weight:700;color:{INK}">{val}</div>'
                  f'<div style="font-size:11px;font-weight:600;color:{c}">{sg}{it["pct"]:.1f}%</div></div>')
    st.markdown(f'<div style="display:flex;gap:14px;overflow-x:auto;margin:2px 0 14px;'
                f'padding-bottom:4px">{cells}</div>', unsafe_allow_html=True)


def render_rank_row(i, r, prefix=""):
    c1, c2 = st.columns([0.78, 0.22])
    with c1:
        color = UP if r["pct"] >= 0 else DOWN
        sg = "+" if r["pct"] >= 0 else ""
        st.markdown(f"""<div class="li">
          <div class="li-rank">{i}</div>
          <div class="li-mid"><div class="li-name">{r['name']}</div><div class="li-code">{r['ticker']}</div></div>
          <div><div class="li-price">{fmt_price(r['last'], r['market'])}</div>
          <div class="li-chg" style="color:{color}">{sg}{r['pct']:.2f}%</div></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.write("")
        if st.button("보기", key=f"rk_{prefix}_{r['ticker']}_{i}", width='stretch'):
            open_detail(r["ticker"])


def render_discover():
    nav()
    st.markdown(f"<div style='font-size:24px;font-weight:800;color:{INK};margin:2px 0 12px'>발견</div>",
                unsafe_allow_html=True)
    render_index_bar()
    scope = st.radio("범위", ["국내", "해외", "전체"], horizontal=True,
                     label_visibility="collapsed")
    st.markdown(f"<div style='font-size:16px;font-weight:700;color:{INK};margin:6px 2px'>실시간 차트</div>",
                unsafe_allow_html=True)
    ranks = get_rankings_cached(scope)
    tabs = st.tabs(["💰 거래대금", "📊 거래량", "📈 급상승", "📉 급하락"])
    for tab, key in zip(tabs, ["거래대금", "거래량", "급상승", "급하락"]):
        with tab:
            d = ranks.get(key)
            if d is None or d.empty:
                st.caption("순위를 불러오지 못했어요.")
                continue
            for i, (_, r) in enumerate(d.iterrows(), 1):
                render_rank_row(i, r, prefix=f"{scope}_{key}")


def dark_axes(fig, date_fmt=None):
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=SUB), dragmode=False)
    fig.update_xaxes(fixedrange=True, showgrid=False, color=SUB,
                     zeroline=False, **({"tickformat": date_fmt} if date_fmt else {}))
    fig.update_yaxes(fixedrange=True, showgrid=False, color=SUB, zeroline=False, side="right")
    return fig


LST = listings()
ss = st.session_state
ss.setdefault("view", "home")
ss.setdefault("ticker", "005930")
ss.setdefault("watch", list(DEFAULT_WATCH))


def open_detail(tk):
    ss.ticker = tk; ss.view = "detail"; st.rerun()


# ======================================================================
# 홈 — 관심 종목 목록
# ======================================================================
def render_home():
    nav()
    st.markdown(f"<div style='font-size:24px;font-weight:800;color:{INK};margin:2px 0 14px'>내 증권</div>",
                unsafe_allow_html=True)

    q = st.text_input("종목 검색", placeholder="🔍 삼성전자, AAPL, 카카오…",
                      label_visibility="collapsed")
    if q:
        res = sl.search_ticker(LST, q, limit=8)
        if len(res):
            for _, r in res.iterrows():
                flag = "🇰🇷" if r["market"] == "KR" else "🇺🇸"
                if st.button(f"{flag}  {r['name']}  ·  {r['ticker']}",
                             key=f"s_{r['ticker']}", width='stretch'):
                    open_detail(r["ticker"])
        else:
            st.caption("결과가 없어요. 코드(005930)나 영문 심볼(AAPL)로도 검색해 보세요.")
        st.divider()

    st.markdown(f"<div style='font-size:14px;font-weight:700;color:{SUB};margin:4px 2px 2px'>"
                "관심 종목 · 오늘 등락률 순</div>", unsafe_allow_html=True)

    rows = []
    for tk in ss.watch:
        try:
            qd = get_quote(tk)
            rows.append((tk, sl.get_name(LST, tk), sl.market_of(LST, tk), qd))
        except Exception:
            rows.append((tk, sl.get_name(LST, tk), sl.market_of(LST, tk), None))
    rows.sort(key=lambda x: (x[3]["pct"] if x[3] else -999), reverse=True)

    for i, (tk, name, market, qd) in enumerate(rows, 1):
        c1, c2 = st.columns([0.76, 0.24])
        with c1:
            if qd:
                color = UP if qd["chg"] >= 0 else DOWN
                sign = "+" if qd["chg"] >= 0 else ""
                st.markdown(f"""<div class="li">
                  <div class="li-rank">{i}</div>
                  <div class="li-mid"><div class="li-name">{name}</div><div class="li-code">{tk}</div></div>
                  <div><div class="li-price">{fmt_price(qd['last'], market)}</div>
                  <div class="li-chg" style="color:{color}">{sign}{qd['pct']:.2f}%</div></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="li"><div class="li-rank">{i}</div>
                  <div class="li-mid"><div class="li-name">{name}</div>
                  <div class="li-code">{tk} · 시세 못 불러옴</div></div></div>""", unsafe_allow_html=True)
        with c2:
            st.write("")
            if st.button("보기", key=f"o_{tk}", width='stretch'):
                open_detail(tk)

    with st.expander("관심 종목 편집"):
        keep = st.multiselect("목록에 둘 종목", ss.watch, default=ss.watch,
                              format_func=lambda t: f"{sl.get_name(LST, t)} ({t})")
        if st.button("적용", width='stretch'):
            ss.watch = keep; st.rerun()
        st.caption("종목 추가는 위 검색 → 종목 화면의 ☆ 버튼으로. "
                   "목록은 앱 재시작 시 기본값으로 돌아가요(영구 저장은 코드의 DEFAULT_WATCH 수정).")


# ======================================================================
# 상세
# ======================================================================
def candle_volume_chart(d, market):
    o, h, l, c = d["Open"], d["High"], d["Low"], d["Close"]
    v = d["Volume"] if "Volume" in d else None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.76, 0.24], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(
        x=d.index, open=o, high=h, low=l, close=c, name="",
        increasing_line_color=UP, decreasing_line_color=DOWN,
        increasing_fillcolor=UP, decreasing_fillcolor=DOWN), 1, 1)
    if v is not None:
        vc = [UP if cc >= oo else DOWN for oo, cc in zip(o, c)]
        fig.add_trace(go.Bar(x=d.index, y=v, marker_color=vc, marker_line_width=0,
                             opacity=0.6), 2, 1)
    hi_i, lo_i = c.idxmax(), c.idxmin()
    fig.add_annotation(x=hi_i, y=c.loc[hi_i], text=f"최고 {fmt_price(c.loc[hi_i], market)}",
                       showarrow=False, yshift=12, font=dict(size=10, color=SUB), row=1, col=1)
    fig.add_annotation(x=lo_i, y=c.loc[lo_i], text=f"최저 {fmt_price(c.loc[lo_i], market)}",
                       showarrow=False, yshift=-12, font=dict(size=10, color=SUB), row=1, col=1)
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=14, b=0), showlegend=False,
                      xaxis_rangeslider_visible=False)
    dark_axes(fig)
    fig.update_xaxes(tickformat="%Y-%m-%d", row=2, col=1)
    return fig


def render_detail():
    tk = ss.ticker
    name = sl.get_name(LST, tk)
    market = sl.market_of(LST, tk)

    cb, cs = st.columns([0.5, 0.5])
    if cb.button("← 목록", width='stretch'):
        ss.view = "home"; st.rerun()
    starred = tk in ss.watch
    if cs.button("★ 관심 해제" if starred else "☆ 관심 추가", width='stretch'):
        ss.watch = [x for x in ss.watch if x != tk] if starred else ss.watch + [tk]
        st.rerun()

    try:
        df = get_data(tk)
    except Exception as e:
        st.error(f"데이터를 불러오지 못했어요: {e}"); return

    last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
    pdate = df.index[-2]
    chg, pct = last - prev, (last / prev - 1) * 100
    color = UP if chg >= 0 else DOWN
    sign = "+" if chg >= 0 else ""
    st.markdown(f"""<div class="card">
      <div class="t-name">{name}</div>
      <div class="t-code">{tk} · {'코스피/코스닥' if market=='KR' else '미국'}</div>
      <div class="t-price">{fmt_price(last, market)}</div>
      <div class="t-change" style="color:{color}">{pdate.month}월 {pdate.day}일보다 {sign}{fmt_price(abs(chg), market)} ({sign}{pct:.2f}%)</div>
    </div>""", unsafe_allow_html=True)

    t_chart, t_info, t_bt, t_ml = st.tabs(["📈 차트", "📋 종목정보", "🧪 백테스트", "🤖 예측"])

    with t_chart:
        period = st.radio("기간", ["1주", "1달", "3달", "1년", "5년"], index=2,
                          horizontal=True, label_visibility="collapsed")
        days = {"1주": 5, "1달": 21, "3달": 63, "1년": 252, "5년": 1260}[period]
        d = df.tail(days)
        pr = (d["Close"].iloc[-1] / d["Close"].iloc[0] - 1) * 100
        st.plotly_chart(candle_volume_chart(d, market), width='stretch',
                        config={"displayModeBar": False, "scrollZoom": False})
        st.markdown(f"<div style='text-align:center;color:{UP if pr>=0 else DOWN};"
                    f"font-weight:700;font-size:15px;margin-top:-4px'>"
                    f"{period} 수익률 {'+' if pr>=0 else ''}{pr:.2f}%</div>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for k, val in [("RSI(14)", f"{df['RSI14'].iloc[-1]:.1f}"),
                       ("20일 이평", fmt_price(df["SMA20"].iloc[-1], market)),
                       ("60일 이평", fmt_price(df["SMA60"].iloc[-1], market)),
                       ("52주 최고", fmt_price(df["Close"].tail(252).max(), market)),
                       ("52주 최저", fmt_price(df["Close"].tail(252).min(), market))]:
            st.markdown(f'<div class="t-row"><span class="k">{k}</span><span class="v">{val}</span></div>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with t_info:
        info = sl.info_stats(df)

        def pos_bar(lab_lo, lo, lab_hi, hi, cur):
            pct = max(2, min(98, (cur - lo) / (hi - lo) * 100 if hi > lo else 50))
            return (f'<div style="margin:10px 0 18px">'
                    f'<div style="position:relative;height:6px;background:{DIV};border-radius:3px">'
                    f'<div style="position:absolute;left:{pct}%;top:50%;transform:translate(-50%,-50%);'
                    f'width:12px;height:12px;border-radius:50%;background:#22C55E;border:2px solid {CARD}"></div></div>'
                    f'<div style="display:flex;justify-content:space-between;margin-top:8px;font-size:12px;color:{SUB}">'
                    f'<div>{lab_lo}<br><span style="color:{INK};font-weight:600">{fmt_price(lo, market)}</span></div>'
                    f'<div style="text-align:right">{lab_hi}<br><span style="color:{INK};font-weight:600">{fmt_price(hi, market)}</span></div>'
                    f'</div></div>')

        vol = f"{info['거래량']:,.0f}주" if info["거래량"] else "-"
        st.markdown(f"""<div class="card">
          <div style="font-size:16px;font-weight:700;color:{INK}">시세</div>
          {pos_bar('1일 최저', info['1일최저'], '1일 최고', info['1일최고'], info['현재가'])}
          {pos_bar('1년 최저', info['1년최저'], '1년 최고', info['1년최고'], info['현재가'])}
          <div class="t-row"><span class="k">시작가</span><span class="v">{fmt_price(info['시작가'], market)}</span></div>
          <div class="t-row"><span class="k">종가</span><span class="v">{fmt_price(info['종가'], market)}</span></div>
          <div class="t-row"><span class="k">거래량</span><span class="v">{vol}</span></div>
          <div class="t-row"><span class="k">거래대금</span><span class="v">{fmt_value(info['거래대금'], market)}</span></div>
        </div>""", unsafe_allow_html=True)

        tbl = sl.daily_table(df, 30)
        head = (f'<div style="display:flex;color:{SUB};font-size:12px;font-weight:600;'
                f'padding:6px 2px;border-bottom:1px solid {DIV}">'
                f'<div style="flex:1.1">날짜</div><div style="flex:1.4;text-align:right">종가</div>'
                f'<div style="flex:1;text-align:right">등락률</div>'
                f'<div style="flex:1.4;text-align:right">거래량</div></div>')
        body = ""
        for date, r in tbl.iterrows():
            c = UP if r["등락률"] >= 0 else DOWN
            sg = "+" if r["등락률"] >= 0 else ""
            volr = f"{int(r['거래량']):,}" if "거래량" in tbl.columns else "-"
            body += (f'<div style="display:flex;font-size:13px;padding:9px 2px;border-bottom:1px solid {DIV}">'
                     f'<div style="flex:1.1;color:{SUB}">{date}</div>'
                     f'<div style="flex:1.4;text-align:right;color:{INK};font-weight:600">{fmt_price(r["종가"], market)}</div>'
                     f'<div style="flex:1;text-align:right;color:{c};font-weight:600">{sg}{r["등락률"]:.2f}%</div>'
                     f'<div style="flex:1.4;text-align:right;color:{SUB}">{volr}</div></div>')
        st.markdown(f'<div class="card"><div style="font-size:16px;font-weight:700;color:{INK};'
                    f'margin-bottom:8px">일별 시세</div>{head}{body}</div>', unsafe_allow_html=True)

    with t_bt:
        chosen = st.multiselect("전략 (여러 개 = 결합)", list(sl.STRATEGIES.keys()),
                                default=["SMA 20-60 교차"])
        mode = st.radio("결합", ["AND", "OR"], horizontal=True)
        cc1, cc2 = st.columns(2)
        slp = cc1.slider("손절 %", 0, 30, 7); tpp = cc2.slider("익절 %", 0, 50, 0)
        if chosen and st.button("백테스트 실행", width='stretch', type="primary"):
            sig = sl.combine_signals(df, chosen, mode)
            bt = sl.backtest(df, sig, stop_loss=slp/100 or None, take_profit=tpp/100 or None)
            p = sl.performance(bt); ex = p["전략 누적수익률"] - p["단순보유 누적수익률"]
            st.markdown(f"""<div class="card">
              <div class="t-row"><span class="k">전략 누적수익</span><span class="v" style="color:{UP if p['전략 누적수익률']>=0 else DOWN}">{fmt_pct(p['전략 누적수익률'])}</span></div>
              <div class="t-row"><span class="k">단순보유 누적수익</span><span class="v">{fmt_pct(p['단순보유 누적수익률'])}</span></div>
              <div class="t-row"><span class="k">단순보유 대비</span><span class="v" style="color:{UP if ex>=0 else DOWN}">{'+' if ex>=0 else ''}{fmt_pct(ex)}</span></div>
              <div class="t-row"><span class="k">최대낙폭(MDD)</span><span class="v" style="color:{DOWN}">{fmt_pct(p['최대낙폭(MDD)'])}</span></div>
              <div class="t-row"><span class="k">거래 횟수</span><span class="v">{p['거래횟수']}회</span></div>
            </div>""", unsafe_allow_html=True)
            st.caption("⚠️ 과거 성과는 미래를 보장하지 않아요. 단순보유를 못 이기는 전략이 대부분입니다.")

    with t_ml:
        st.markdown(f"""<div class="card" style="background:#2A2310;border:1px solid #5A4A14">
          <div style="font-weight:700;color:#F0C24B">⚠️ 예측의 한계</div>
          <div style="font-size:13px;color:#D9C68A;margin-top:6px;line-height:1.5">
          미래는 한 점으로 못 맞혀서 '확률 범위(부채꼴)'로 보여드려요. 기간이 길수록 띠가
          넓어지는 게 정상이고, 뉴스·실적 충격은 반영하지 못합니다. 참고용으로만 보세요.</div>
        </div>""", unsafe_allow_html=True)
        hsel = st.selectbox("예측 기간", ["1주", "1달", "3달", "6달", "1년", "3년", "5년"], index=2)
        H = {"1주": 5, "1달": 21, "3달": 63, "6달": 126, "1년": 252, "3년": 756, "5년": 1260}[hsel]
        long_h = H >= 252
        zero_drift = st.checkbox("변동성만 보기 (추세 제거)", value=long_h)
        if st.button("예측 실행", width='stretch', type="primary"):
            try:
                band, S0, mc = sl.project_cone(df, horizon=H,
                                               lookback=(None if long_h else 756),
                                               drift=("zero" if zero_drift else "keep"))
                hist = df["Close"].tail(min(120, max(40, H // 4)))
                xb = [hist.index[-1]] + list(band.index)
                def w0(col): return [S0] + list(band[col])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist, name="실제", line=dict(color=INK, width=1.8)))
                fig.add_trace(go.Scatter(x=xb, y=w0("p95"), line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=xb, y=w0("p5"), fill="tonexty", fillcolor="rgba(77,158,255,0.12)",
                                         line=dict(width=0), name="90% 범위", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=xb, y=w0("p75"), line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=xb, y=w0("p25"), fill="tonexty", fillcolor="rgba(77,158,255,0.28)",
                                         line=dict(width=0), name="50% 범위", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=xb, y=w0("p50"), name="중앙값", line=dict(color=BLUE, width=2, dash="dot")))
                fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                                  legend=dict(orientation="h", font=dict(color=SUB)))
                dark_axes(fig, date_fmt="%Y-%m-%d")
                st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
                lo, mid, hi = band["p5"].iloc[-1], band["p50"].iloc[-1], band["p95"].iloc[-1]
                mlab = "중앙값(변동성만)" if zero_drift else "중앙값(추세 연장)"
                st.markdown(f"""<div class="card">
                  <div class="t-row"><span class="k">{mlab}</span><span class="v">{fmt_price(mid, market)} ({(mid/S0-1)*100:+.1f}%)</span></div>
                  <div class="t-row"><span class="k">90% 확률 범위</span><span class="v">{fmt_price(lo, market)} ~ {fmt_price(hi, market)}</span></div>
                  <div class="t-row"><span class="k">{hsel} 뒤 상승확률</span><span class="v" style="color:{UP if mc>=0.5 else DOWN}">{mc*100:.0f}%</span></div>
                </div>""", unsafe_allow_html=True)
                if H <= 63:
                    res = sl.train_predict(df, "RandomForest", horizon=H)
                    if res["정확도-기준선"] <= 0.005:
                        st.warning(f"이 종목·{hsel} 기준 모델 예측력은 사실상 없어요. 부채꼴 범위만 참고하세요.")
                else:
                    st.info("📌 방향 예측 모델은 3개월 이하만 제공해요(그 이상은 검증 불가). 부채꼴 범위만 봐주세요.")
            except Exception as e:
                st.error(f"오류: {e}")


if ss.view == "home":
    render_home()
elif ss.view == "discover":
    render_discover()
else:
    render_detail()

st.markdown(f"<div style='text-align:center;color:{SUB};font-size:12px;margin-top:18px'>"
            "분석·학습용 · 투자 판단의 책임은 본인에게 있습니다</div>", unsafe_allow_html=True)
