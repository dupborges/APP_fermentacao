#########################################################################################################################
#########################################################################################################################

# %%
#CARREGA AS BIBLIOTECAS
from pydoc import describe
from threading import local
from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
import matplotlib
#import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import matplotlib.text as text
import matplotlib.dates as mdates

import matplotlib.font_manager as fm
from matplotlib import rcParams

import os
import seaborn as sns
from scipy import stats

pd.set_option('display.max_columns', None)

import base64
import io

import warnings
warnings.simplefilter(action='ignore')

#################### define o estilo dos gráficos feitos pelo SNS
global estilo_sns
#estilo_sns = "dark"
estilo_sns ="white"


export = None #'C:\\Users\\Public\\Documents'
pasta_local = os.getcwd()
os.chdir(pasta_local)


########################################################################################################################
########################################################################################################################
####################################################  INICIO ###########################################################
########################################################################################################################
########################################################################################################################

# >>>>>>>>>>>>>>> INICIA O STREAMLIT
import streamlit as st
st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

st.title('APP ENSAIO DE FERMENTAÇÃO')
# Importa o logo do GAOA

############################### SIDE CONFIGURAÇÕES #######################################

st.sidebar.write("Configurações")

uploaded_file = st.sidebar.file_uploader("Escolha o Arquivo")

if uploaded_file is not None:
    arq = uploaded_file#.name
    st.write('UPLOAD')
    st.write(arq)
    st.write ('Nome do aqruivo:',uploaded_file.name)

    arquivo = arq

    ensaio  = st.sidebar.text_input('Nome do ensaio: ', "Ensaio")
    st.title('Análise de dados: '+ensaio)

    significancia = st.sidebar.number_input('Significancia', value=0.05, step=0.05)


    import carrega_xlsx_app_fermentacao
    df = carrega_xlsx_app_fermentacao.carrega_xlsx(arquivo=arquivo
                                    ,INDEXAR = True
                                    ,mantem_OBJECTS = True
                                    ,IMPUTA_NAN = False
                                    ,INTERPOLATE = False
                                    ,Taxa_Elementos_min = 0.3
                                    )



    st.title('Informações do ensaio')
    st.write ('Número total de fermentações:', df.shape[0])
    st.write ('Número total de tratamentos:', df['tratamento'].nunique())
    st.write ('Número total de ciclos:', df['ciclo'].nunique())
    st.write ('Número total de repetições:', df['repetição'].nunique())
    st.write ('Tratamentos:', df['tratamento'].unique())
    
    with st.expander('RESUMO GERAL DOS DADOS'):
        df_dados = df.drop(['repetição'], axis = 1)
        st.dataframe(df_dados.describe().round(2))

    with st.expander ('AGRUPADO POR TRATAMENTO'):
        df_tratamento=df_dados.groupby(by=['tratamento']).mean()
        st.dataframe(df_tratamento.round(2).drop(['ciclo'], axis = 1))

    with st.expander ('AGRUPADO POR CICLO'):
        df_tratamento=df_dados.groupby(by=['ciclo']).mean()
        st.dataframe(df_tratamento.round(2))

    with st.expander ('AGRUPADO POR TRATAMENTO E CICLO'):
        df_tratamento=df_dados.groupby(by=['tratamento','ciclo']).mean()
        st.dataframe(df_tratamento.round(2))

    lista_features = df.columns[3:].tolist()
    features = st.multiselect("SELECIONE AS VARIÁVEIS PARA SEREM ANALISADAS:",lista_features, lista_features)

    usar_primeiro_ciclo  = st.sidebar.selectbox ('USAR o primeiro ciclo nas análises estatísticas: ', ['Não','Sim'])
    if usar_primeiro_ciclo == 'Não':
        df= df[df['ciclo'] > 1]

    col_wrap = st.sidebar.number_input('Quantidade de colunas no gráfico de comparação', value = df['tratamento'].nunique())

   ##################################################################################################
   # ########### ROTNA PARA SELECIONAR PASTA PARA EXPORTAÇÃO DOS DADOS                              #
   #                                                                                                #
   # import libraries                                                                               #
    import streamlit as st                                                                          #
    import tkinter as tk                                                                            #
    from tkinter import filedialog                                                                  #
                                                                                                    #
    # Set up tkinter                                                                                #
    root = tk.Tk()                                                                                  #
    root.withdraw()                                                                                 #
                                                                                                    #
    # Make folder picker dialog appear on top of other windows                                      #
    root.wm_attributes('-topmost', 1)                                                               #
                                                                                                    #
    # Folder picker button                                                                          #
    #st.title('Folder Picker')                                                                      #
    #st.write('Please select a folder:')                                                            #
    clicked = st.button('Selecione a pasta para exportação dos dados para continuar as análises')   #
    if clicked:                                                                                     #
        export = st.text_input('pasta para exportação dos dados:', filedialog.askdirectory(master=root))            #
    #################################################################################################


    import graficos
    import anova_tukey
    import pca

    if export is not None:
        # FAZ ANOVA E TESTE DE TUKEY
        # st.write(features)
        
        anova_tukey.anova_tukey(df,features,export,significancia,show=False)

        #APRESENTA GRÁFICOS COM RESUMO DOS DADOS
        
        graficos.graficos(df,features,export,col_wrap)

        # COLOCAR O PCA
        
        pca.fatorial_pca(df=df,
                        features=features,
                        export=export, 
                        PC = 0, 
                        Ylabel = 'tratamento', 
                        mostra_labels= True,
                        r2_limite=0.6,significancia = significancia)

        # COLOCAR ALGUMA TÉCNICA DE AGRUPAMENTO

os.chdir(pasta_local)    








