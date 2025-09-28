# ------------------------------------------------------------
# app.py – 公開用（メモ機能なし）
# ------------------------------------------------------------
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- ページ設定 ----------
st.set_page_config(page_title="Stock Viewer", layout="wide")
st.title("マルチ銘柄チャート")

# ---------- サイドバー ----------
st.sidebar.header("表示設定")

raw_codes = st.sidebar.text_input(
    "銘柄コード（カンマ区切り）",
    value="7203, AAPL, 7060, 9556"
)

interval_jp = st.sidebar.selectbox("足種", ["日足", "週足", "月足"])
interval_map = {"日足": "1d", "週足": "1wk", "月足": "1mo"}
interval = interval_map[interval_jp]

cols_num = st.sidebar.selectbox("チャートの列数を選んでね", [1, 2, 3, 4], index=1)

indicators = st.sidebar.multiselect(
    "インジケーター（複数選択可）",
    ["SMA 20", "EMA 50", "Bollinger Bands", "RSI 14"],
    default=[]
)

# ---------- コード整形 (.T 補完) ----------
tickers = []
for code in raw_codes.split(","):
    code = code.strip().upper()
    if not code:
        continue
    if "." not in code and code.isdigit():
        code += ".T"
    tickers.append(code)

if not tickers:
    st.warning("銘柄コードを入力してください")
    st.stop()

# ---------- 期間設定 ----------
period = {"1d": "2y", "1wk": "5y", "1mo": "10y"}[interval]

# ---------- データ取得 ----------
data = yf.download(
    tickers=" ".join(tickers),
    period=period,
    interval=interval,
    group_by="ticker",
    auto_adjust=True,
    progress=False,
    threads=False
)

# ---------- インジケーター計算 ----------
def add_indicators(df: pd.DataFrame, inds: list) -> pd.DataFrame:
    if "SMA 20" in inds:
        df["SMA20"] = df["Close"].rolling(20).mean()
    if "EMA 50" in inds:
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    if "Bollinger Bands" in inds:
        ma = df["Close"].rolling(20).mean()
        std = df["Close"].rolling(20).std()
        df["BB_upper"] = ma + 2 * std
        df["BB_lower"] = ma - 2 * std
    if "RSI 14" in inds:
        delta = df["Close"].diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        ma_up = up.ewm(alpha=1/14, adjust=False).mean()
        ma_down = down.ewm(alpha=1/14, adjust=False).mean()
        rs = ma_up / ma_down
        df["RSI14"] = 100 - (100 / (1 + rs))
    return df

# ---------- チャート作成 ----------
def make_chart(df: pd.DataFrame, symbol: str, inds: list, h=350):
    df = add_indicators(df.copy(), inds)
    rsi_on = "RSI 14" in inds
    rows, heights = (3, [0.6, 0.25, 0.15]) if rsi_on else (2, [0.75, 0.25])

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.02
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], whiskerwidth=0.12,
        increasing=dict(line=dict(color="#26a69a", width=1)),
        decreasing=dict(line=dict(color="#ef5350", width=1)),
        name="Price"), row=1, col=1)
    if "SMA 20" in inds:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], line=dict(width=1, color="#f5c542"), name="SMA20"), row=1, col=1)
    if "EMA 50" in inds:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], line=dict(width=1, color="#42c5f5"), name="EMA50"), row=1, col=1)
    if "Bollinger Bands" in inds:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], line=dict(width=1, color="#888"), name="BB up", hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], line=dict(width=1, color="#888"), name="BB lo", fill='tonexty', fillcolor="rgba(136,136,136,0.1)", hoverinfo="skip"), row=1, col=1)

    vol_colors = ["#26a69a" if o < c else "#ef5350" for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, opacity=0.5, name="Volume"), row=2, col=1)

    if rsi_on:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], line=dict(color="#e0e64c", width=1), name="RSI14"), row=3, col=1)
        for y in [70, 30]:
            fig.add_hline(y=y, line=dict(color="#666", width=0.5, dash="dot"), row=3, col=1)

    fig.update_layout(height=h, template="plotly_dark", plot_bgcolor="#131722", paper_bgcolor="#131722", margin=dict(l=10, r=10, t=25, b=0), hovermode="x unified", dragmode="pan", xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#FFF")))
    fig.update_yaxes(gridcolor="#333")
    return fig

plot_cfg = dict(scrollZoom=True, displaylogo=False, modeBarButtonsToRemove=["lasso2d", "select2d"])

# ---------- 動的列レイアウト ----------
cols = st.columns(cols_num)
for idx, symbol in enumerate(tickers):
    if len(tickers) == 1:
        df_stock = data.dropna()
    else:
        if symbol not in data:
            st.error(f"{symbol} のデータがありません")
            continue
        df_stock = data[symbol].dropna()

    if df_stock.empty:
        st.error(f"{symbol} のデータがありません")
        continue

    fig = make_chart(df_stock, symbol, indicators)

    with cols[idx % cols_num]:
        st.caption(symbol)
        # --- ★keyを追加してエラー回避 ---
        st.plotly_chart(fig, use_container_width=True, config=plot_cfg, key=f"chart_{symbol}_{idx}")

