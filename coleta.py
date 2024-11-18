import streamlit as st
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import joblib

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
    # Obtenção dos dados da ação
    tickerData = yf.Ticker(tickerSimbolo)
    tickerDF = tickerData.history(period="1d", start=inicio, end=final)

    # Verificação para garantir que existem dados no período selecionado
    if not tickerDF.empty:
        # Colunas para gráficos de fechamento e volume
        col1, col2 = st.columns(2)

        with col1:
            st.header("Gráfico de Fechamento")
            st.line_chart(tickerDF["Close"])

        with col2:
            st.header("Gráfico de Volume")
            st.line_chart(tickerDF["Volume"])

        st.write("---")
        st.title("Previsão de Ações")
        st.write("---")

        # Preparação dos dados para o modelo
        tickerDF["Price_Change"] = tickerDF["Close"] - tickerDF["Open"]
        tickerDF["SMA_10"] = tickerDF["Close"].rolling(window=10).mean()  # Média móvel de 10 dias
        tickerDF = tickerDF.dropna()  # Remover linhas com valores NaN

        # Seleção de features
        x = tickerDF[["Open", "High", "Low", "Volume", "Price_Change", "SMA_10"]].to_numpy()

        # Escalonamento dos dados
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        # Caminho para o arquivo do modelo salvo
        modelo_path = "modelo_random_forest.joblib"

        # Carregar o modelo salvo
        try:
            modelo_carregado = joblib.load(modelo_path)
            st.write("Modelo carregado com sucesso.")
        except FileNotFoundError:
            st.error("O arquivo do modelo salvo não foi encontrado. Certifique-se de que 'modelo_random_forest.joblib' está no diretório correto.")
            st.stop()

        # Previsão para uma data futura usando o modelo carregado
        ultimos_valores = x[-1].reshape(1, -1)  # Últimos valores escalonados
        previsao_futura = modelo_carregado.predict(ultimos_valores)

        # Exibição da previsão
        with st.sidebar:
            st.write("Previsão para Data Futura")
            data_futura = st.date_input("Escolha uma data futura", value=datetime.now().date() + timedelta(days=5))
        
        st.subheader(f"**Previsão do preço de fechamento para {data_futura}: R$ {previsao_futura[0]:.2f}**")

    else:
        st.warning("Não há dados disponíveis para o período selecionado.")
else:
    st.info("Por favor, selecione um ativo e o período de datas para exibir os gráficos e a previsão.")


