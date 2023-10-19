import yfinance as yf # datos
import datetime as dt # ventana de tiempo
from dateutil.relativedelta import relativedelta
import plotly.express as px # graficos
import streamlit as st
from scipy import stats
from streamlit_plotly_events import plotly_events
import numpy as np
import pandas as pd
import random as rd

# Funciones

def getData(optn,start,end):

    stake = {'Google': 'GOOGL',
             'Amazon': 'AMZN',
             'Netflix': 'NFLX',
             'Apple': 'AAPL',
             'Facebook': 'META',
             'TNX': '^TNX'}

    data = yf.download(stake[optn],start=start,end=end,interval='1d') 
    return pd.DataFrame(data)

def getExpirationDate(date,time):

    optn = {'1 mes': 1, '3 meses': 3,
            '6 meses': 6, '1 año': 12}
    
    return date.today() + relativedelta(months=+optn[time])

def calculateTrayectories(df,start,end,k):
    
    fechas = pd.date_range(start+relativedelta(days=1),end)
    trayectories = pd.DataFrame(index=fechas)
    for i in range(200):
        trayectories[i] = MBG(df, len(fechas), k)

    return trayectories

def MBG(df, tiempo, k):

    #t= len(df)

    Z= np.random.normal(size=tiempo) # Movimiento browniano
    h=1/tiempo
    mu= (df["Rendimiento"].mean()/h) + (df["Rendimiento"].std()**2/(2*h))
    sigma= df["Rendimiento"].std()/(h**0.5)
    X=[]
    for i in Z:
        X.append(mu*h-(sigma**2/2)*h+sigma*(h**0.5)*i)
    X=np.array(X)
    S=[]
    for i in range(tiempo):
        p1= df["Close"].iloc[-1]
        suma= (X[0:i+1]).sum()
        exponencial= np.exp(suma)
        sol= p1*exponencial
        S.append(sol)
    S= np.array(S)
    return S

def modeloBlackSholes(precio_accion,precio_ejercicio,tasa,tiempo,volatilidad):
    '''
    Inputs:
    precio_accion (float): Precio actual de la acción
    precio_ejercicio (float): Precio de ejercicio de la acción
    tasa (float): Tasa de interés libre de riesgo (anualizada)
    tiempo (float): Tiempo hasta la expiración de la opción (en años)
    volatilidad (float): Volatilidad del precio de la acción (anualizada)

    Outputs:
    call: Precio de la opción de compra
    put: Precio de la opción de venta
    '''

    t = {'1 mes': 1/12, '3 meses': 3/12,
         '6 meses': 6/12, '1 año': 1}
    
    # Cálculo de d1 y d2
    d1 = (np.log(precio_accion / precio_ejercicio) + (tasa + volatilidad**2 / 2) * t[tiempo]) / (volatilidad * np.sqrt(t[tiempo]))
    d2 = d1 - volatilidad * np.sqrt(t[tiempo])

    # Cálculo de los precios de opción de compra y la opción de venta
    call = stats.norm.cdf(d1) * precio_accion  - stats.norm.cdf(d2) * precio_ejercicio * np.exp(-tasa * t[tiempo])
    put = stats.norm.cdf(-d2) * precio_ejercicio * np.exp(-tasa * t[tiempo]) - stats.norm.cdf(-d1) * precio_accion

    return call,put

# Página

st.set_page_config(
    page_title='Opciones financieras',
    layout='wide',
)

st.title('Opciones Europeas')

a1,a2,a3,a4,a5 = st.columns([2,1,1,2,2])

stake = a1.selectbox('Activo subyacente',('Facebook','Amazon','Apple','Netflix','Google')) # Confirmar que no tengan dividendos
start_date = a2.date_input('Fecha de inicio',dt.date(2023, 1, 1))
end_date = a3.date_input('Fecha de fin',max_value=dt.date.today())
expiration_date = a4.selectbox('Fecha de vencimiento',('1 mes','3 meses','6 meses','1 año'),
                               help='Fecha de vencimiento a partir de la fecha de fin seleccionada.')
a5.info(f'Fecha de vencimiento seleccionada: {getExpirationDate(end_date,expiration_date)}')

st.header('Analisis de la acción')

df = getData(stake,start_date,end_date)
df['Media movil'] = df['Adj Close'].rolling(5).mean()
df['Rendimiento'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))

fig_stake = px.line(df,x=df.index,y=['Adj Close','Media movil'],title=f'Valor de la acción {stake}',
                    color_discrete_sequence=px.colors.qualitative.Plotly)
fig_stake.update_layout(title_x=0)
plotly_events(fig_stake)

c1, c2 = st.columns(2)

# Primera columna

with c1:
    fig_return1 = px.line(df,x=df.index,y=['Rendimiento'], title='Rendimiento Logarítmico',
                          color_discrete_sequence=px.colors.qualitative.Plotly)
    fig_return1.update_layout(showlegend=False,title_x=0)
    plotly_events(fig_return1)

# Segunda columna

with c2:
    fig_return2 = px.histogram(df['Rendimiento'],title='Distribución del rendimiento',
                               color_discrete_sequence=px.colors.qualitative.Plotly)
    fig_return2.update_layout(showlegend=False,title_x=0)
    plotly_events(fig_return2)

st.header('Movimiento Browniano Geométrico')

d1, d2 = st.columns(2)

with d1:
    trayectorias_MBG = calculateTrayectories(df,end_date,getExpirationDate(end_date,expiration_date),
                                             df['Media movil'][-1])
    fig_MBG = px.line(trayectorias_MBG,x=trayectorias_MBG.index,y=[x for x in range(200)],
                    color_discrete_sequence=px.colors.qualitative.Plotly)
    fig_MBG.update_layout(title='Simulaciones (200) del precio subyacente',title_x=0)
    plotly_events(fig_MBG)

with d2:
    trayectoria_media = pd.DataFrame(trayectorias_MBG.mean(axis=1))
    fig_tMedia = px.line(trayectoria_media,color_discrete_sequence=px.colors.qualitative.Plotly)
    fig_tMedia.update_layout(title='Trayectoria promedio del precio subyacente',title_x=0,
                             showlegend=False)
    plotly_events(fig_tMedia)

df['Tipo'] = 'Valor real'
trayectoria_media['Tipo'] = 'Predicción'
trayectoria_media.columns = ['Adj Close','Tipo']
predict = pd.concat([df[['Adj Close','Tipo']],trayectoria_media],axis=0)
fig_proy = px.line(predict,x=predict.index,y='Adj Close',color='Tipo',
                   color_discrete_sequence=px.colors.qualitative.Plotly)
plotly_events(fig_proy)

st.header('Modelo Black-Sholes')

valor_medio = df['Rendimiento'].mean()
sigma = df['Rendimiento'].std()
volatilidad = sigma * np.sqrt(252)

interes = getData('TNX',start_date,end_date)['Adj Close'] / 100

col1, col2 = st.columns([1,2])

with col1.form('black-sholes'):
    S = st.number_input('Precio de la acción', value = df['Adj Close'][-1])
    K = st.number_input('Precio de ejercicio', min_value=1)
    vol = st.number_input('Volatilidad', value = volatilidad)
    tasa = st.number_input('Tasa de interés',value = interes[-1],format='%f')

    submitted = st.form_submit_button("Submit")
    if submitted:
        compra, venta = modeloBlackSholes(S,K,tasa,expiration_date,vol)

        with col2:
            st.write(f'Precio de la acción: {S}')
            st.write(f'Precio de ejercicio: {K}')
            st.write(f'Volatilidad {round(vol * 100,2)} %')
            st.write(f'Tasa de interés libre de riesgo: {tasa}')
            st.write(f'Tiempo hasta la fecha de ejercicio: {expiration_date}')

            st.info(f'Precio de la opción de call: {round(compra,2)}')
            st.info(f'Precio de la opción de put: {round(venta,2)}')