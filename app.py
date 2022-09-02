import streamlit as st
import numpy as np
import pandas as pd
import datetime as datetime
import pandas_datareader
import datetime
import plotly.graph_objects as go
from PIL import Image
import sklearn
import sklearn.linear_model
import sklearn.model_selection
import yfinance as yf

st.write("""
# Crypto Currency Dashboard Application
Visually show data on crypto (BTC-JPY, DOGE-JPY, ETH-JPY, XTZ-JPY, XRP-JPY, LTC-JPY, XEM-JPY)
""")

image = Image.open("crypto_image3.PNG")
st.image(image, use_column_width=True)

st.sidebar.header("User Input")

#csvデータをgetする
def crypto_get():
    start_days = datetime.datetime(2020, 1, 1)
    end_days = datetime.datetime.now()
    # BTCをゲット
    df_btc = pandas_datareader.DataReader("BTC-JPY", "yahoo", start_days, end_days)
    df_btc.to_csv("BTC.csv")
    # ETHをゲット
    df_btc = pandas_datareader.DataReader("ETH-JPY", "yahoo", start_days, end_days)
    df_btc.to_csv("ETH.csv")
    # DOGEをゲット
    df_btc = pandas_datareader.DataReader("DOGE-JPY", "yahoo", start_days, end_days)
    df_btc.to_csv("DOGE.csv")
    # Tezosをゲット
    df_btc = pandas_datareader.DataReader("XTZ-JPY", "yahoo", start_days, end_days)
    df_btc.to_csv("XTZ.csv")
    # XRPをゲット
    df_btc = pandas_datareader.DataReader("XRP-JPY", "yahoo", start_days, end_days)
    df_btc.to_csv("XRP.csv")
    # LTCをゲット
    df_btc = pandas_datareader.DataReader("LTC-JPY", "yahoo", start_days, end_days)
    df_btc.to_csv("LTC.csv")
    # NEMをゲット
    df_btc = pandas_datareader.DataReader("XEM-JPY", "yahoo", start_days, end_days)
    df_btc.to_csv("XEM.csv")

crypto_get()

def get_input():
    dt_now = datetime.datetime.now()
    end_days = dt_now.date()

    start_date = st.sidebar.text_input("Start Date", "2021-01-01")
    end_date = st.sidebar.text_input("End Date", end_days)
    st.sidebar.write("Enter the CryptoCurrency ticker symbol")
    st.sidebar.write("BTC, DOGE, ETH, XTZ, XRP, LTC, XEM")
    crypto_symbol = st.sidebar.text_input("Crypto Symbol", "BTC")
    return start_date, end_date, crypto_symbol

def get_crypto_name(symbol):
    symbol = symbol.upper()
    if symbol == "BTC":
        return "Bitcoin"
    elif symbol == "ETH":
        return  "Etherium"
    elif symbol == "DOGE":
        return  "Dogecoin"
    elif symbol == "XTZ":
        return  "TEZOS"
    elif symbol == "XRP":
        return  "RIPPLE"
    elif symbol == "LTC":
        return  "Litecoin"
    elif symbol == "XEM":
        return  "NEM"
    else:
        return "None"

def get_data(symbol, start, end):
    symbol = symbol.upper()
    if symbol == "BTC":
        df = pd.read_csv("BTC.csv")
    elif symbol == "ETH":
        df = pd.read_csv("ETH.csv")
    elif symbol == "DOGE":
        df = pd.read_csv("DOGE.csv")
    elif symbol == "XTZ":
        df = pd.read_csv("XTZ.csv")
    elif symbol == "XRP":
        df = pd.read_csv("XRP.csv")
    elif symbol == "LTC":
        df = pd.read_csv("LTC.csv")
    elif symbol == "XEM":
        df = pd.read_csv("XEM.csv")
    else:
        df = pd.DataFrame(columns=["Date", "Close", "Open", "Volume", "Adj Close"])
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    return df.loc[start:end]

start, end, symbol = get_input()
df = get_data(symbol, start, end)
crypto_name = get_crypto_name(symbol)

#表示するCandle Stick の設定
fig = go.Figure(
    data = [go.Candlestick(
        x = df.index,
        open = df["Open"],
        high = df["High"],
        low = df["Low"],
        close = df["Close"],
        increasing_line_color = "green",
        decreasing_line_color = "red"
        )
    ]
)

st.header(crypto_name + "Data")

st.write(df)

st.header(crypto_name + "Data Statistics")
st.write(df.describe())

st.header(crypto_name + "Close Price")
st.line_chart(df["Close"])

st.header(crypto_name + "Volume")
st.bar_chart(df["Volume"])

st.header(crypto_name + "Candle Stick")
st.plotly_chart(fig)

df["label"] = df["Close"].shift(-30)

st.header(crypto_name + "Predict one month later.")

def predict_crypto():
    #機械学習
    X = np.array(df.drop(["label"], axis=1))
    X = sklearn.preprocessing.scale(X)
    predict_data = X[-30:]
    X = X[:-30]
    y = np.array(df["label"])
    y = y[:-30]
    #データ分割
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size= 0.2)

    #訓練データを使って学習
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    # 小数点第一位で四捨五入
    st.write(f"正答率は{round((accuracy) * 100, 1)}%です。")

    # accuracyより信頼度を表示
    if accuracy > 0.75:
        st.write("信頼度：高")
    elif accuracy > 0.5:
        st.write("信頼度：中")
    else:
        st.write("信頼度：低")
    st.write("オレンジの線（Predict）が予想値です。")

    # 検証データを使って検証する
    predicted_data = model.predict(predict_data)
    df["Predict"] = np.nan
    last_date = df.iloc[-1].name
    one_day = 86400
    next_unix = last_date.timestamp() + one_day

    for data in predicted_data:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += one_day
        df.loc[next_date] = np.append([np.nan] * (len(df.columns) - 1), data)

    df["Close"].plot(figsize=(15, 6), color="green")
    df["Predict"].plot(figsize=(15, 6), color="orange")

    df_stock3 = df[["Close", "Predict"]]
    st.line_chart(df_stock3)
    # ボタンを押すとstock_predict()が発動


if st.button("予測する"):
    predict_crypto()

st.write("Copyright © 2021 Tomouyki Yoshikawa. ALL Rights Reserved.")
