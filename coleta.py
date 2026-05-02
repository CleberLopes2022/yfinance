import streamlit as st
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import joblib
import altair as alt

# Configurações da página
st.set_page_config(page_title="Análise de Ações", layout="wide")

st.write("---")
st.title("Preço de Ativo")
st.write("---")

# Sidebar
with st.sidebar:
    st.image("shutterstock_349461494.jpg")
    st.header("Ações")
    tickerSimbolo = st.selectbox(
        "Escolha o Ativo", 
        ("PETR4.SA", "BBAS3.SA", "VALE3.SA", "COGN3.SA"),
        index=0
    )
    default_start_date = datetime.now().date() - relativedelta(years=10)
    inicio = st.date_input("Escolha a data de início", value=default_start_date)
    final = st.date_input("Escolha a data final", value=datetime.now().date())

if tickerSimbolo and inicio and final:
    tickerData = yf.Ticker(tickerSimbolo)
    tickerDF = tickerData.history(start=inicio, end=final)

    if not tickerDF.empty:
        # Gráficos interativos
        col1, col2 = st.columns(2)

        with col1:
            st.header("Gráfico de Fechamento")
            chart_close = alt.Chart(tickerDF.reset_index()).mark_line(color="blue").encode(
                x="Date:T", y="Close:Q", tooltip=["Date:T", "Close:Q"]
            ).interactive()
            st.altair_chart(chart_close, use_container_width=True)

        with col2:
            st.header("Gráfico de Volume")
            chart_volume = alt.Chart(tickerDF.reset_index()).mark_area(color="orange", opacity=0.6).encode(
                x="Date:T", y="Volume:Q", tooltip=["Date:T", "Volume:Q"]
            ).interactive()
            st.altair_chart(chart_volume, use_container_width=True)

        st.write("---")
        st.title("Previsão de Ações")
        st.write("---")

        # Features
        tickerDF["Price_Change"] = tickerDF["Close"] - tickerDF["Open"]
        tickerDF["SMA_10"] = tickerDF["Close"].rolling(window=10).mean()
        tickerDF = tickerDF.dropna()

        x = tickerDF[["Open", "High", "Low", "Volume", "Price_Change", "SMA_10"]].to_numpy()
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        modelo_path = "modelo_random_forest.joblib"
        try:
            modelo_carregado = joblib.load(modelo_path)
            st.success("Modelo carregado com sucesso.")
        except FileNotFoundError:
            st.error("O arquivo do modelo salvo não foi encontrado.")
            st.stop()

        # Previsão para múltiplos dias
        dias_futuros = 7
        ultimos_valores = x[-1].reshape(1, -1)
        previsoes = []

        for i in range(dias_futuros):
            pred = modelo_carregado.predict(ultimos_valores)[0]
            previsoes.append(pred)

            # Atualiza entrada simulando próximo dia
            nova_linha = ultimos_valores.copy()
            nova_linha[0, 0] = pred  # Open ~ previsão
            nova_linha[0, 1] = pred * 1.01  # High (estimativa simples)
            nova_linha[0, 2] = pred * 0.99  # Low (estimativa simples)
            nova_linha[0, 4] = pred - nova_linha[0, 0]  # Price_Change
            nova_linha[0, 5] = pred  # SMA_10 aproximada
            ultimos_valores = nova_linha

        # Datas futuras
        datas_futuras = [tickerDF.index[-1] + timedelta(days=i+1) for i in range(dias_futuros)]
        df_previsoes = pd.DataFrame({"Date": datas_futuras, "Previsao": previsoes})

        # Gráfico comparativo
        chart_compare = alt.Chart(tickerDF.reset_index()).mark_line(color="blue").encode(
            x="Date:T", y="Close:Q"
        ) + alt.Chart(df_previsoes).mark_line(color="red").encode(
            x="Date:T", y="Previsao:Q"
        )

        st.header("Comparativo Real vs Previsões Futuras (7 dias)")
        st.altair_chart(chart_compare.interactive(), use_container_width=True)

        # Tabela de previsões
        st.subheader("Tabela de Previsões")
        st.dataframe(df_previsoes.set_index("Date"))

    else:
        st.warning("Não há dados disponíveis para o período selecionado.")
else:
    st.info("Por favor, selecione um ativo e o período de datas.")


