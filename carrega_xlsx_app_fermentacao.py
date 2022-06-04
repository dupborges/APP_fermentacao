def carrega_xlsx (arquivo,INDEXAR,mantem_OBJECTS,IMPUTA_NAN,INTERPOLATE,Taxa_Elementos_min):
    import pandas as pd
    import streamlit as st
    #abre o arquivo
    print ('Arquivo analisado:',arquivo)
    xls = pd.ExcelFile(arquivo)
   
    TIPO=xls.sheet_names[0]
    st.write ('pasta a ser analizada ==>',TIPO)

    #CARREGA INFORMAÇÕES DA ABA dados
    dados = xls.parse(TIPO)

    ##############################################################################################################################


    #DEFINE A COLUNA QUE VAI SER INDEX e A VARIAVEL DEPENDENTE
    COL=dados.columns[0]
    print('********************')

    colunas_originais=dados.columns.size
    st.write('Shape df original:' + str(dados.shape))
    #st.write(dados.columns)
    #####################################################################################################################    
    ##################################################################################################################
    print ('ELIMINA AS COLUNAS QUE TEM MENOS QUE ', Taxa_Elementos_min*100,'% DE ELEMENTOS NA COLUNA' )
    taxa_minima_valores_nao_nulos = Taxa_Elementos_min
    limite_valores_validos = dados.index.size * taxa_minima_valores_nao_nulos
    dados = dados.dropna(thresh=limite_valores_validos, axis=1)
        
    #################################################################################################################   
    if INDEXAR == True: #DEFINE A COLUNA QUE VAI SER INDEX
        print('Coluna que vai ser INDEX:', COL)
        dados.set_index(dados[COL],inplace=True)
        dados=dados.drop(columns =[COL])
    ##################################################################################################################
    
    if mantem_OBJECTS == False:
        print('ELIMINA COLUNAS QUE SÃO OBJECT')
        for i in dados.columns:
            if dados[i].dtype == 'object':
                print (i, ' exluidas por serem objects')
                dados=dados.drop(columns =[i])
                
    ####################################################################################################################
    print('elimina as colunas que tem todas as variáveis sem valores')
    dados=dados.dropna(how='all', axis=1)
    
    print('elimina as linha que tem todas as variáveis sem valores')
    dados=dados.dropna(how='all', axis=0)
    
    CR=dados.columns.size
    #elimina as linhas que tem todas as variáveis sem valores
    print('TOTAL DE COLUNAS ELIMINADAS = ',colunas_originais-CR)

    ###############################################################################################################################    
    #IMPUTA OS VALORES FALTANTES
    if IMPUTA_NAN ==True:
        df_numerical = dados.select_dtypes(exclude='object')
        #print(df_numerical.info())
        df_object = dados.select_dtypes(include='object')
        #print(df_object.info())
        
        if INTERPOLATE==True:
            print('INTERPOLA OS VALORES')
            df_numerical = df_numerical.interpolate(method='linear', axis=0).ffill().bfill()
        else:
            df_numerical=df_numerical.fillna(df_numerical.median())
        ################### ELIMINA OUTLIER E INTERPOLA OS RESULTADOS
        df_numerical=df_numerical.dropna(how='any', axis=0) #limpa as linhas que ainda tem um NaN
        dados=pd.concat([df_object,df_numerical], axis=1) #junta as colunas numericas e object

        
    return dados
