import streamlit as st
import pandas as pd
import joblib

# Cargar los archivos que subiste a GitHub
modelo = joblib.load('modelo_vitalfit.pkl')
scaler = joblib.load('scaler_vitalfit.pkl')

# EL TRUCO MAESTRO: Leer la mente del modelo para ver qué columnas exactas espera
columnas_modelo = modelo.feature_names_in_

st.title('Sistema de Prediccion de Churn - VitalFit S.A.')

opcion = st.sidebar.selectbox('Seleccione Tipo de Prediccion', ['Individual', 'Masiva (Cargar CSV)'])

if opcion == 'Individual':
    st.subheader('Prediccion Individual de Socio')
    recencia = st.number_input('Recencia (Dias desde la ultima visita)', min_value=0, value=30)
    frecuencia = st.number_input('Frecuencia (Total de visitas)', min_value=1, value=10)
    valor_monetario = st.number_input('Valor Monetario Total ($)', min_value=0, value=50000)
    
    ticket_promedio = valor_monetario / frecuencia if frecuencia > 0 else 0
    
    if st.button('Evaluar Riesgo'):
        # Crear una tabla vacía que tenga EXACTAMENTE las columnas que el modelo pidió
        datos = pd.DataFrame(0, index=[0], columns=columnas_modelo)
        
        # Llenar las columnas que siempre existen
        datos['recencia'] = recencia
        datos['frecuencia'] = frecuencia
        datos['valor_monetario'] = valor_monetario
        datos['ticket_promedio'] = ticket_promedio
        
        # Llenar las columnas de tipo de cliente SOLO si el modelo las está pidiendo
        if 'tipo_cliente_Esporadico' in columnas_modelo:
            datos['tipo_cliente_Esporadico'] = 1 if frecuencia <= 4 else 0
        if 'tipo_cliente_Regular' in columnas_modelo:
            datos['tipo_cliente_Regular'] = 1 if 5 <= frecuencia <= 12 else 0
        if 'tipo_cliente_Frecuente' in columnas_modelo:
            datos['tipo_cliente_Frecuente'] = 1 if frecuencia > 12 else 0
            
        # Normalizar los datos matemáticos
        cols_num = ['recencia', 'frecuencia', 'valor_monetario', 'ticket_promedio']
        datos[cols_num] = scaler.transform(datos[cols_num])
        
        # Predecir
        prediccion = modelo.predict(datos)[0]
        probabilidad = modelo.predict_proba(datos)[0][1]
        
        if prediccion == 1:
            st.error(f'Alerta: Alto riesgo de fuga. Probabilidad: {probabilidad * 100:.2f}%')
        else:
            st.success(f'Socio Estable. Probabilidad de fuga: {probabilidad * 100:.2f}%')

else:
    st.subheader('Prediccion Masiva de Socios')
    archivo = st.file_uploader('Cargue el archivo CSV con las metricas de los socios', type=['csv'])
    
    if archivo is not None:
        df = pd.read_csv(archivo)
        
        if st.button('Procesar Lista'):
            # Misma lógica dinámica pero para múltiples clientes
            df_procesado = pd.DataFrame(0, index=df.index, columns=columnas_modelo)
            
            df_procesado['recencia'] = df['recencia']
            df_procesado['frecuencia'] = df['frecuencia']
            df_procesado['valor_monetario'] = df['valor_monetario']
            df_procesado['ticket_promedio'] = df['valor_monetario'] / df['frecuencia']
            
            if 'tipo_cliente_Esporadico' in columnas_modelo:
                df_procesado['tipo_cliente_Esporadico'] = df['frecuencia'].apply(lambda x: 1 if x <= 4 else 0)
            if 'tipo_cliente_Regular' in columnas_modelo:
                df_procesado['tipo_cliente_Regular'] = df['frecuencia'].apply(lambda x: 1 if 5 <= x <= 12 else 0)
            if 'tipo_cliente_Frecuente' in columnas_modelo:
                df_procesado['tipo_cliente_Frecuente'] = df['frecuencia'].apply(lambda x: 1 if x > 12 else 0)
            
            cols_num = ['recencia', 'frecuencia', 'valor_monetario', 'ticket_promedio']
            df_procesado[cols_num] = scaler.transform(df_procesado[cols_num])
            
            df['Prediccion_Churn'] = modelo.predict(df_procesado)
            df['Probabilidad_Fuga_%'] = (modelo.predict_proba(df_procesado)[:, 1] * 100).round(2)
            
            st.write('Resultados del Analisis:')
            st.dataframe(df)
