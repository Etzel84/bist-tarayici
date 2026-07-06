import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier

# 1. Web Sitesi Sayfa Ayarları
st.set_page_config(
    page_title="BIST Hacim Destekli Yapay Zeka Tarayıcı", 
    page_icon="📈", 
    layout="wide"
)

# Stil ve Özelleştirmeler
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stButton>button { background-color: #10b981; color: white; border-radius: 8px; width: 100%; font-weight: bold; font-size: 16px; }
    .stButton>button:hover { background-color: #059669; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 BIST Hacim & Yapay Zeka Destekli Akıllı Tarayıcı")
st.subheader("Büyük Oyuncuların Girdiği Hacimli Hisseleri ve Yapay Zeka Sinyallerini Yakala")

# BIST'teki Tüm Güncel Hisseleri İnternetten Otomatik Çeken Fonksiyon
@st.cache_data(ttl=86400)
def bist_tum_hisseleri_getir():
    try:
        url = "https://tr.wikipedia.org/wiki/Borsa_İstanbul%27da_işlem_gören_şirketler"
        tablolar = pd.read_html(url)
        df_hisse = tablolar[0]
        
        kod_sutunu = [col for col in df_hisse.columns if "Kod" in col or "Sembol" in col][0]
        hisseler = df_hisse[kod_sutunu].dropna().astype(str).tolist()
        
        bist_listesi = [f"{hisse.strip().upper()}.IS" for hisse in hisseler if len(hisse.strip()) <= 5]
        return sorted(list(set(bist_listesi)))
    except:
        return ["THYAO.IS", "ASELS.IS", "EREGL.IS", "TUPRS.IS", "ISCTR.IS", "BIMAS.IS", "KCHOL.IS", "SISE.IS", "GARAN.IS", "AKBNK.IS"]

# Veri Çekme Fonksiyonu
@st.cache_data(ttl=3600)
def veri_getir(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, timeout=5)
        if df is None or len(df) < 80: return None
        return df
    except:
        return None

# Makine Öğrenmesi, Teknik Analiz ve Hacim Kontrol Özellikleri
def teknik_ve_yz_analiz(df):
    kapanis_serisi = df['Close'].values.flatten()
    hacim_serisi = df['Volume'].values.flatten()
    
    df_temp = pd.DataFrame(index=df.index)
    df_temp['Close'] = kapanis_serisi
    df_temp['Volume'] = hacim_serisi
    
    # Hacim Filtresi Hesaplaması
    son_3_gun_hacim = df_temp['Volume'].iloc[-3:].mean()
    son_20_gun_hacim = df_temp['Volume'].iloc[-20:].mean()
    hacim_artisi = son_3_gun_hacim / (son_20_gun_hacim + 1)
    
    # İndikatörler
    df_temp['RSI'] = RSIIndicator(close=df_temp['Close']).rsi()
    macd = MACD(close=df_temp['Close'])
    df_temp['MACD'] = macd.macd()
    df_temp['MACD_Signal'] = macd.macd_signal()
    df_temp['SMA_20'] = df_temp['Close'].rolling(window=20).mean()
    df_temp['SMA_50'] = df_temp['Close'].rolling(window=50).mean()
    df_temp['Hedef'] = np.where(df_temp['Close'].shift(-1) > df_temp['Close'], 1, 0)
    df_temp.dropna(inplace=True)
    
    if len(df_temp) < 10:
        return 0, 50, kapanis_serisi[-1], 1.0
        
    X = df_temp[['RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_50']]
    y = df_temp['Hedef']
    
    model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
    model.fit(X.iloc[:-1], y.iloc[:-1])
    
    olasilik = model.predict_proba(X.iloc[[-1]])[0][1]
    return olasilik, float(df_temp['RSI'].iloc[-1]), float(df_temp['Close'].iloc[-1]), hacim_artisi

# Canlı Listeyi Yükle
butun_bist_hisseleri = bist_tum_hisseleri_getir()

st.sidebar.metric(label="🚀 Taranacak Toplam Hisse Sayısı", value=len(butun_bist_hisseleri))
st.sidebar.markdown("""
### Filtre Kriterleri:
* **YZ Güven Skoru:** > %60
* **RSI Seviyesi:** < 65
* **Hacim Patlaması:** > %20
""")

# Web Sitesi Butonu ve Çalıştırma Bölümü
if st.button("🔍 TÜM BIST HİSSELERİNİ (HACİM ODAKLI) TARAMAYA BAŞLAT"):
    bilgi_kutusu = st.info(f"Yapay zeka modelleri eğitiliyor ve kurumsal para girişleri taranıyor. Lütfen bekleyin...")
    
    ilerleme_bar = st.progress(0)
    durum_yazisi = st.empty()
    sonuclar = []
    
    for idx, ticker in enumerate(butun_bist_hisseleri):
        hisse_yalniz_ad = ticker.replace(".IS", "")
        durum_yazisi.text(f"Para girişi ve teknik yapı analiz ediliyor: {hisse_yalniz_ad} ({idx+1}/{len(butun_bist_hisseleri)})")
        
        df = veri_getir(ticker)
        if df is not None and len(df) > 80:
            try:
                # Tek seferde tüm verileri alarak hızı 2 katına çıkarıyoruz
                yz_skoru, rsi_degeri, son_fiyat, hacim_artisi = teknik_ve_yz_analiz(df.copy())
                
                if yz_skoru > 0.60 and rsi_degeri < 65 and hacim_artisi >= 1.20:
                    stop_loss = son_fiyat * 0.95
                    hedef_fiyat = son_fiyat * 1.08
                    
                    sonuclar.append({
                        "Hisse Adı": hisse_yalniz_ad,
                        "Mevcut Fiyat (TL)": f"{son_fiyat:.2f}",
                        "Teknik Hedef (TL)": f"{hedef_fiyat:.2f}",
                        "Stop-Loss (TL)": f"{stop_loss:.2f}",
                        "RSI": f"{rsi_degeri:.1f}",
                        "Hacim Artış Katı": f"{hacim_artisi:.2f}x",
                        "YZ Güven Skoru": f"%{yz_skoru*100:.1f}"
                    })
            except:
                continue
                
        ilerleme_bar.progress((idx + 1) / len(butun_bist_hisseleri))
    
    # İşlem bittiğinde geçici yazıları tamamen temizle
    durum_yazisi.empty()
    bilgi_kutusu.empty()
    
    # Sonuç Paneli
    if sonuclar:
        st.success(f"Analiz Bitti! Hacim patlaması yaşayan ve yapay zekanın onay verdiği {len(sonuclar)} hisse bulundu:")
        sonuc_df = pd.DataFrame(sonuclar)
        sonuc_df = sonuc_df.sort_values(by="Hacim Artış Katı", ascending=False)
        st.dataframe(sonuc_df, use_container_width=True)
    else:
        st.warning("Bütün BIST tarandı. Filtrelere uyan uygun bir hisse bulunamadı.")