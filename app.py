"""
BIST100 Hisse Tarama ve Puanlama — Mobil Uyumlu Web Uygulaması
=================================================================
Bu, masaüstü (Tkinter) uygulamasının web sürümüdür. Tarayıcı üzerinden
çalıştığı için telefondan da erişilebilir.

Yerel çalıştırma:
    pip install -r requirements.txt
    streamlit run app.py

Telefondan erişim (aynı WiFi ağında):
    streamlit run app.py --server.address 0.0.0.0
    Sonra telefonun tarayıcısından: http://<BİLGİSAYARIN_YEREL_IP'Sİ>:8501

Her yerden telefondan erişim (ücretsiz bulut deploy):
    README.md'deki "Telefondan her yerden erişim" bölümüne bakın.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

import scanner_core as core

st.set_page_config(
    page_title="BIST100 Tarama",
    page_icon="📊",
    layout="centered",  # mobilde "wide" yerine "centered" daha iyi görünür
)

# ------------------------------------------------------------------
# Mobil ekranlarda daha rahat dokunma alanı için hafif CSS ayarı
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    div.stButton > button { width: 100%; padding: 0.75em; font-size: 1.05em; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 BIST100 Hisse Tarama")
st.caption("Teknik + temel kriterlere göre puanlanmış en iyi hisseler")

# ------------------------------------------------------------------
# Ayarlar — mobilde dikey sıralanır (columns yerine tek sütun)
# ------------------------------------------------------------------
with st.expander("⚙️ Ayarlar", expanded=False):
    top_n = st.slider("Kaç hisse listelensin", min_value=5, max_value=10, value=core.DEFAULT_TOP_N)
    min_vol = st.number_input(
        "Minimum günlük hacim (TL)",
        min_value=0,
        value=core.DEFAULT_MIN_AVG_VOLUME_TRY,
        step=500_000,
    )

run_clicked = st.button("▶ Taramayı Başlat", type="primary")

# ------------------------------------------------------------------
# Tarama
# ------------------------------------------------------------------
if run_clicked:
    try:
        tickers = core.load_tickers()
    except Exception as e:
        st.error(f"Ticker listesi yüklenemedi: {e}")
        st.stop()

    progress_bar = st.progress(0, text="Başlatılıyor...")
    status_placeholder = st.empty()

    def progress_cb(i, total, ticker):
        progress_bar.progress(i / total, text=f"Çekiliyor: {ticker} ({i}/{total})")

    try:
        with st.spinner("Veriler çekiliyor, bu 1-2 dakika sürebilir..."):
            scored = core.run_scan(
                top_n=top_n,
                min_avg_volume_try=min_vol,
                tickers=tickers,
                progress_callback=progress_cb,
            )
        progress_bar.empty()
        st.session_state["scored_df"] = scored
        st.session_state["scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state["top_n"] = top_n
    except Exception as e:
        progress_bar.empty()
        st.error(f"Tarama başarısız oldu: {e}")
        st.stop()

# ------------------------------------------------------------------
# Sonuçları göster (session_state'te varsa — sayfa yeniden çizilse de kalır)
# ------------------------------------------------------------------
if "scored_df" in st.session_state:
    scored = st.session_state["scored_df"]
    top_n = st.session_state["top_n"]
    top = scored.head(top_n)

    st.success(f"Tamamlandı ({st.session_state['scan_time']}) — {len(scored)} hisse tarandı")

    # Mobilde geniş tablo yerine kart görünümü daha okunaklı
    for rank, (ticker, row) in enumerate(top.iterrows(), 1):
        def fmt(v, decimals=1, suffix=""):
            try:
                if v is None or pd.isna(v):
                    return "—"
                return f"{v:.{decimals}f}{suffix}"
            except Exception:
                return "—"

        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{rank}. {ticker.replace('.IS', '')}**")
            with c2:
                st.markdown(f"**{fmt(row.get('final_score'))} / 100**")

            m1, m2, m3 = st.columns(3)
            m1.metric("Fiyat", fmt(row.get("last_price"), 2, " TL"))
            m2.metric("F/K", fmt(row.get("pe_ratio")))
            m3.metric("ROE", fmt(row.get("roe"), 1, "%"))

            m4, m5, m6 = st.columns(3)
            m4.metric("PD/DD", fmt(row.get("pb_ratio"), 2))
            m5.metric("3A Momentum", fmt(row.get("momentum_3m"), 1, "%"))
            m6.metric("RSI", fmt(row.get("rsi")))

    st.divider()

    # Tam tabloyu isteyenler için (yatay kaydırmalı)
    with st.expander("Tüm taranan hisseler (tam tablo)"):
        st.dataframe(scored, use_container_width=True)

    csv = scored.to_csv(encoding="utf-8-sig")
    st.download_button(
        "⬇ CSV Olarak İndir",
        data=csv,
        file_name=f"bist_tarama_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
else:
    st.info("Taramayı başlatmak için yukarıdaki butona basın.")
