from itertools import groupby


def anova_tukey(df,features,export,significancia,show=False):
    import streamlit as st
    import pandas as pd
    import pingouin as pg
    import numpy as np
    import matplotlib
    #import matplotlib.cbook as cbook
    import matplotlib.pyplot as plt
    import matplotlib.text as text
    import matplotlib.dates as mdates

    import seaborn as sns
    from scipy import stats
    import warnings
    warnings.simplefilter(action='ignore')
    import pingouin as pg
    from io import BytesIO

    import shap
    import base64
    import io
    anova = pd.DataFrame()
    #anova_significantes = pd.DataFrame()
    

    significantes = []
    with pd.ExcelWriter(export+'/ANOVA.xlsx') as writer:  
        print ('FAZENDO ANOVA')
        for e,i in enumerate(features):
            aov = pg.rm_anova(dv=i, within=['tratamento', 'ciclo'], subject='repetição', data=df, detailed=True)
            #aov['Parâmetro']=i
            #colunas_reordenadas=['Parâmetro']+aov.columns[:-1].tolist()
            #aov=aov[colunas_reordenadas]
            if aov['p-unc'].min() < significancia:
                pasta = str(e)+' significante'
                pasta = i[:11] +' significante'
                significantes.append(i)



            else:
                pasta = str(e)

            aov.index.name = i    
            aov.to_excel(writer, sheet_name=pasta)

            if show == True:
                st.write ('ANOVA',i)
                st.dataframe (aov)
     
     
     
            #aov.columns = aov.columns.str.replace('Source', i)
            #anova = pd.concat([anova,aov], axis = 0)

    st.write('Variáveis com diferenças significativas na ANOVA')
    st.write(significantes)
    

    #Pairwise Tukey-HSD post-hoc test.
    #References
    #1 Tukey, John W. “Comparing individual means in the analysis of variance.” Biometrics (1949): 99-114.
    #2 Gleason, John R. “An accurate, non-iterative approximation for studentized range quantiles.” Computational statistics & data analysis 31.2 (1999): 147-158.
    
    ################################################################
    # COMPARAÇÃO DOS TRATAMENTOS
    #################################################################

    #SALVA A TABELA DO TESTE DE TUKEY
    with pd.ExcelWriter(export+'/TUKEY_TRATAMENTO.xlsx') as writer:  
        print ('FAZENDO TUKEY TRATAMENTO')
        for i in significantes:
            print ('FAZENDO TUKEY TRATAMENTO PARA:', i)

            #st.write(i)

            import pingouin as pg
            T = df.pairwise_tukey(dv=i, between='tratamento').round(4)
            T['Ho']=np.nan
    
            for j in T.index:
                if T['p-tukey'].loc[j]<significancia:
                    
                    T['Ho'].loc[j]='Rejeita Ho'  

            T.index.name = i
            #st.write ('Tabela de Tukey')    
            #st.dataframe(T) 
            pasta = i[:11]
            T.to_excel(writer, sheet_name=pasta)


            # FAZ A TABELA COM AS MÉDIAS E AS LETRAS
            df_tratamento = df.groupby(by='tratamento').mean().round(4)
            df_tratamento_sorted = df_tratamento.sort_values(by=i, ascending=False)

            #st.dataframe(df_tratamento_sorted)
            pp = pd.DataFrame(index = df_tratamento_sorted.index, columns= df_tratamento_sorted.index)


            for j in T.index:
                
                pp[T['A'].loc[j]].loc[T['B'].loc[j]] = T['p-tukey'].loc[j]
                pp[T['B'].loc[j]].loc[T['A'].loc[j]] = T['p-tukey'].loc[j]
                pp.fillna(1, inplace=True)
                #pp[T['A'].loc[j]].loc[T['A'].loc[j]] = 1# df_tratamento_sorted['valor'].loc[T['A'].loc[j]]

            
            
            pp.index.name = i

            pp_np = pp.to_numpy()
            letra = tukeyLetters(pp_np, means=None, alpha=significancia)

            pp['Médias']= df_tratamento_sorted[i]
            pp['letra'] = letra

            pasta2 = i[:11]+" matrix Tukey"
            pp.to_excel(writer, sheet_name=pasta2)            

            if show == True:
                st.write ('Matriz teste de Tukey',i)
                st.dataframe (pp)

            #df_tratamento_sorted['letra']= letra
            #st.dataframe(df_tratamento_sorted[[i,'letra']])
    st.write('Teste de Tukey dos tratamentos finalizado e salvo na planilha em Excel')



    ################################################################
    # COMPARAÇÃO DOS ciclos dentro de cada tratamento
    #################################################################

    #SALVA A TABELA DO TESTE DE TUKEY
    with pd.ExcelWriter(export+'/TUKEY_CICLOS.xlsx') as writer:  
        for i in significantes:
            print ('FAZENDO TUKEY CICLOS PARA:', i)
            pp_parametro = pd.DataFrame()

            T_parametro = pd.DataFrame()

            for trat in df.tratamento.unique():
                df_trat = df[['ciclo', i]][df['tratamento']==trat]
                #st.write(trat)
                #st.dataframe(df_trat)

                import pingouin as pg
                T = df_trat.pairwise_tukey(dv=i, between='ciclo').round(4)
                T['Ho']=np.nan
        
                for j in T.index:
                    if T['p-tukey'].loc[j]<significancia:
                        
                        T['Ho'].loc[j]='Rejeita Ho'  

                

                linha_separacao_de_tratamento = pd.DataFrame([['','','','','','','','','','' ]], columns=T.columns,index=[trat])
                #linha_separacao_de_tratamento = pd.DataFrame([T.columns], columns=T.columns,index=[trat])
                T_parametro = pd.concat([T_parametro,linha_separacao_de_tratamento,T], axis=0)

                #st.write ('Tabela de Tukey')    
                #st.dataframe(T) 
                


                # FAZ A TABELA COM AS MÉDIAS E AS LETRAS
                df_ciclo = df_trat.groupby(by='ciclo').mean().round(4)
                df_ciclo_sorted = df_ciclo.sort_values(by=i, ascending=False)

                #st.dataframe(df_ciclo_sorted)
                pp = pd.DataFrame(index = df_ciclo_sorted.index, columns= df_ciclo_sorted.index)


                for j in T.index:
                    
                    pp[T['A'].loc[j]].loc[T['B'].loc[j]] = T['p-tukey'].loc[j]
                    pp[T['B'].loc[j]].loc[T['A'].loc[j]] = T['p-tukey'].loc[j]
                    pp.fillna(1, inplace=True)
                    #pp[T['A'].loc[j]].loc[T['A'].loc[j]] = 1# df_tratamento_sorted['valor'].loc[T['A'].loc[j]]

                
                
                pp.index.name = i

                pp_np = pp.to_numpy()
                letra = tukeyLetters(pp_np, means=None, alpha=significancia)

                pp['Médias']= df_ciclo_sorted[i]
                pp['letra'] = letra

                colunas_pp = list('-')*pp.columns.size
                linha_separacao_de_tratamento2 = pd.DataFrame([colunas_pp], columns=pp.columns,index=[trat])
                #linha_separacao_de_tratamento2 = pd.DataFrame([pp.columns], columns=pp.columns,index=[trat])
                pp_parametro = pd.concat([pp_parametro,linha_separacao_de_tratamento2,pp], axis=0)
                          

                if show == True:
                    st.write ('Matriz teste de Tukey',i)
                    st.dataframe (pp)

                #df_tratamento_sorted['letra']= letra
                #st.dataframe(df_tratamento_sorted[[i,'letra']])

            pasta = i[:11] 
            T_parametro.index.name = i
            T_parametro.to_excel(writer, sheet_name=pasta)

            pasta2 = i[:11]+ "Matriz Tukey"
            pp_parametro.index.name = i
            pp_parametro.to_excel(writer, sheet_name=pasta2)  


    st.write('Teste de Tukey dos ciclos dentro de cada tratamento finalizado e salvo na planilha em Excel')










    return 


def tukeyLetters(pp, means=None, alpha=0.05):

    import numpy as np

    '''TUKEYLETTERS - Produce list of group labels for TukeyHSD
    letters = TUKEYLETTERS(pp), where PP is a symmetric matrix of 
    probabilities from a Tukey test, returns alphabetic labels
    for each group to indicate clustering. PP may also be a vector
    from PAIRWISE_TUKEYHSD.
    Optional argument MEANS specifies group means, which is used for
    ordering the letters. ("a" gets assigned to the group with lowest
    mean.) Without this argument, ordering is arbitrary.
    Optional argument ALPHA specifies cutoff for treating groups as
    part of the same cluster.'''

    if len(pp.shape)==1:
        # vector
        G = int(3 + np.sqrt(9 - 4*(2-len(pp))))//2
        ppp = .5*np.eye(G)
        ppp[np.triu_indices(G,1)] = pp    
        pp = ppp + ppp.T
    conn = pp>alpha
    G = len(conn)
    if np.all(conn):
        return ['a' for g in range(G)]
    conns = []
    for g1 in range(G):
        for g2 in range(g1+1,G):
            if conn[g1,g2]:
                conns.append((g1,g2))

    letters = [ [] for g in range(G) ]
    nextletter = 0
    for g in range(G):
        if np.sum(conn[g,:])==1:
            letters[g].append(nextletter)
            nextletter += 1
    while len(conns):
        grp = set(conns.pop(0))
        for g in range(G):
            if all(conn[g, np.sort(list(grp))]):
                grp.add(g)
        for g in grp:
            letters[g].append(nextletter)
        for g in grp:
            for h in grp:
                if (g,h) in conns:
                    conns.remove((g,h))
        nextletter += 1

    if means is None:
        means = np.arange(G)
    means = np.array(means)
    groupmeans = []
    for k in range(nextletter):
        ingroup = [g for g in range(G) if k in letters[g]]
        groupmeans.append(means[np.array(ingroup)].mean())
    ordr = np.empty(nextletter, int)
    ordr[np.argsort(groupmeans)] = np.arange(nextletter)
    result = []
    for ltr in letters:
        lst = [chr(97 + ordr[x]) for x in ltr]
        lst.sort()
        result.append(''.join(lst))
    return result