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

# Título e seção
st.write("---")
st.title("Preço de Ativo")
st.write("---")

# Sidebar para seleção de ações
with st.sidebar:
    st.image("shutterstock_349461494.jpg")
    st.header("Ações")
    tickerSimbolo = st.selectbox(
        "Escolha o Ativo", 
        ("PETR4.SA", "BBAS3.SA", "VALE3.SA", "COGN3.SA"),
        index=0
    )
    
    # Data inicial (10 anos atrás a partir de hoje)
    default_start_date = datetime.now().date() - relativedelta(years=10)
    inicio = st.date_input("Escolha a data de início", value=default_start_date)
    
    # Data final (data atual)
    final = st.date_input("Escolha a data final", value=datetime.now().date())

# Verificação de entradas
if tickerSimbolo and inicio and final:
    # Obtenção dos dados da ação (sem period)
    tickerData = yf.Ticker(tickerSimbolo)
    tickerDF = tickerData.history(start=inicio, end=final)

    if not tickerDF.empty:
        # Colunas para gráficos
        col1, col2 = st.columns(2)

        with col1:
            st.header("Gráfico de Fechamento")
            chart_close = alt.Chart(tickerDF.reset_index()).mark_line(color="blue").encode(
                x="Date:T",
                y="Close:Q",
                tooltip=["Date:T", "Close:Q"]
            ).interactive()
            st.altair_chart(chart_close, use_container_width=True)

        with col2:
            st.header("Gráfico de Volume")
            chart_volume = alt.Chart(tickerDF.reset_index()).mark_area(color="orange", opacity=0.6).encode(
                x="Date:T",
                y="Volume:Q",
                tooltip=["Date:T", "Volume:Q"]
            ).interactive()
            st.altair_chart(chart_volume, use_container_width=True)

        st.write("---")
        st.title("Previsão de Ações")
        st.write("---")

        # Preparação dos dados
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
            st.error("O arquivo do modelo salvo não foi encontrado. Certifique-se de que 'modelo_random_forest.joblib' está no diretório correto.")
            st.stop()

        # Previsão
        ultimos_valores = x[-1].reshape(1, -1)
        previsao_futura = modelo_carregado.predict(ultimos_valores)

        with st.sidebar:
            st.write("Previsão para Data Futura")
            data_futura = st.date_input("Escolha uma data futura", value=datetime.now().date() + timedelta(days=5))
        
        st.subheader(f"**Previsão do preço de fechamento para {data_futura}: R$ {previsao_futura[0]:.2f}**")

        # Comparativo real vs previsão
        tickerDF["Previsao"] = np.nan
        tickerDF.iloc[-1, tickerDF.columns.get_loc("Previsao")] = previsao_futura[0]

        chart_compare = alt.Chart(tickerDF.reset_index()).mark_line().encode(
            x="Date:T",
            y=alt.Y("value:Q", title="Preço"),
            color="variable:N"
        ).transform_fold(
            ["Close", "Previsao"]
        ).interactive()

        st.header("Comparativo Real vs Previsão")
        st.altair_chart(chart_compare, use_container_width=True)

    else:
        st.warning("Não há dados disponíveis para o período selecionado.")
else:
    st.info("Por favor, selecione um ativo e o período de datas para exibir os gráficos e a previsão.")

