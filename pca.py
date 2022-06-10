def fatorial_pca(df,features,export, PC, Ylabel, mostra_labels,r2_limite=0.6,significancia = 0.05):
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
    import os
    import numpy as np

    os.chdir(export)
    style="whitegrid"

    
    print ('ANÁLISE DE COMPOENTES PRINCIPAIS')
    with st.expander ('ANÁLISE DE COMPOENTES PRINCIPAIS'):

        import warnings
        warnings.filterwarnings('ignore')
        ###########################################################################  FUNCAO PCA #######################################################
        target = df[Ylabel]
        
        numerico = df[features].select_dtypes(include='number')
        if PC == 0:
            PC = min (numerico.shape)

        rho = df.corr()

        import klib
        print('______________________________________________________________________')
        print ("MATRIZ DE CORRELAÇÃO")
        klib.corr_plot(numerico)
        #plt.show()
        mc_plot = 'matriz_correlacao.png'
        plt.savefig(mc_plot, format = 'png',dpi=100, facecolor='w', edgecolor='b',
                    orientation='portrait', papertype=None, #format=None,
                    transparent=False, bbox_inches='tight', pad_inches=0.2,
                    frameon=None, metadata=None)
        
        st.image(mc_plot, caption= 'Matrix de correlação', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")




        import pingouin as pg
        

        print('______________________________________________________________________')
        st.write('CORRELAÇÕES COM MENOS DE',significancia*100,'% DE SIGNIFICANCIA E R2 >',r2_limite)
        regress = pg.pairwise_corr(df, columns=df.columns,method='pearson')
        regress['r2'] = regress['r']**2
        st.dataframe(regress[(regress['r2']>r2_limite) & (regress['p-unc']<significancia)]) 
        print('______________________________________________________________________')


        st.write ("TESTE DE ESFERICIDADE DE BARBETT")
        chi, pv = teste_de_esfericidade_barbett(numerico)
        
        if pv < 0.05:
            print()
            print()
            print ('________________________________________________________________FAZ O PCA________________________________________________________________________')
            ########################################################
            #### FAZ O PCA

            

            df_std=padroniza(numerico)

            from sklearn.decomposition import PCA
            pca=PCA(n_components=PC, whiten=True)
            Z=pca.fit_transform(df_std)
            scores=pd.DataFrame(data=Z, index=df_std.index)
            scores.columns = scores.columns +1
            ###############################################
            ###     CALCULA OS EIGINVALUES E VARIÂNCIA COMPARTILHADA
            print ('EIGINVALUES E VARIÂNCIA COMPARTILHADA')
            Eigenvalues  = pd.DataFrame([pca.explained_variance_],index=['Eigenvalues'])
            Eigenvalues.columns = Eigenvalues.columns +1
            #display (Eigenvalues)

            PC_validos = (pca.explained_variance_ > 1).sum()

            variacia_compartilhada = pd.DataFrame([pca.explained_variance_ratio_], index=['variancia_compartilhada','variancia_compartilhada_cumulativa'])
            variacia_compartilhada.columns=variacia_compartilhada.columns+1
            for i in variacia_compartilhada.columns:
                if i > 1:
                    variacia_compartilhada[i].at['variancia_compartilhada_cumulativa']= variacia_compartilhada[i-1].at['variancia_compartilhada_cumulativa']+ variacia_compartilhada[i].at['variancia_compartilhada_cumulativa']

            variacia_compartilhada
            pca_informacao = pd.concat([Eigenvalues,variacia_compartilhada], axis=0)
            st.write (pca_informacao)

            #########################################################################################################################
            print('______________________________________________________________________')
            st.write ('__________________________ SCREE PLOT ________________________________')

            from matplotlib.pylab import rcParams
            rcParams['figure.figsize'] = 10,7
            fig, ax = plt.subplots()
            ax = sns.barplot (x=variacia_compartilhada.columns, y=variacia_compartilhada.loc['variancia_compartilhada'])
            ax = sns.lineplot(x=variacia_compartilhada.columns-1, y=variacia_compartilhada.loc['variancia_compartilhada_cumulativa'], marker="o")
            ax.annotate(round(variacia_compartilhada[1].at['variancia_compartilhada_cumulativa'],2), 
            (0, variacia_compartilhada[1].at['variancia_compartilhada_cumulativa']+0.1))
            
            for ii in range(variacia_compartilhada.columns.size+1):
                i=ii-1
                if ii > 1:
                    ax.annotate(round(variacia_compartilhada[ii].at['variancia_compartilhada_cumulativa'],2), 
                    (i, variacia_compartilhada[ii].at['variancia_compartilhada_cumulativa']-.1))
                

            #plt.show()
            scree_plot = 'scree_plot.png'
            #fig = ax.get_figure()
            fig.savefig(scree_plot)

            """plt.savefig(scree_plot, format = 'png',dpi=100, facecolor='w', edgecolor='b',
                        orientation='portrait', papertype=None, #format=None,
                        transparent=False, bbox_inches='tight', pad_inches=0.2,
                        frameon=None, metadata=None)"""
            
            st.image(scree_plot, caption= 'SCREE PLOT', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")




            ################################################

            print('______________________________________________________________________')
            COEFICIENT_Loadings = pd.DataFrame(pca.components_.T ,index = df_std.columns)
            COEFICIENT_Loadings.columns = COEFICIENT_Loadings.columns+1
            #st.dataframe(COEFICIENT_Loadings)   #eigenvector

            ############################################################################################
            ##### CALCULA A CARGA FATORIAL
            ##### QUE NO PAST CHAMA DE CORRELATION LOADINS
            ##### O QUE geralmente chama-se DE LOADING É O COEFICIENT LOADING
            """carga_fatorial=pd.DataFrame()
            for i in scores.columns:
                RR=[]
                for j in numerico.columns:
                    RR.append(scores[i].corr(numerico[j]))
                Rdict={i:RR}
                R = pd.DataFrame(Rdict, index=numerico.columns, columns=[i])
                carga_fatorial=pd.concat([carga_fatorial,R], axis=1)
            st.write('Cargas Fatoriais')
            st.dataframe(carga_fatorial)"""

            ############ forma mais simples de calculo da carga_fatorial
            Eigenvalues_std =(pca.explained_variance_)**0.5
            cargas_fatoriais=COEFICIENT_Loadings*Eigenvalues_std

            st.write('Cargas Fatoriais')
            st.dataframe(cargas_fatoriais)



            print('______________________________________________________________________')
            st.write ('__________________________ LOADING PLOT ______________________________')
            from matplotlib.pylab import rcParams
            rcParams['figure.figsize'] = 10,7
            fig, ax = plt.subplots()
            ax = sns.scatterplot(x=cargas_fatoriais[1], y=cargas_fatoriais[2], marker="o")
            for i in cargas_fatoriais.index:
                ax.annotate(i,(cargas_fatoriais[1].loc[i],cargas_fatoriais[2].loc[i]))
            plt.show()
            loading_plot = 'loading_plot.png'
            fig.savefig(loading_plot)
            
            st.image(loading_plot, caption= 'LOADING PLOT', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")



            #############################################################################################
            #####################  FAZ O GRAFICO DE SCORES #############################################
            print('______________________________________________________________________')
            st.write ('________________________  SCORES PLOT _______________________________')
            X_pca=pd.DataFrame(index=df.index)
            if Ylabel != "":
                X_pca['Grupo']=df[Ylabel]
                X_pca['ciclo']=df['ciclo']
            else:
                X_pca['Grupo']=0

            X_pca['PC1']=scores[1]
            X_pca['PC2']=scores[2]

            from matplotlib.pylab import rcParams
            rcParams['figure.figsize'] = 10,7
            fig, ax = plt.subplots()

            ax = sns.scatterplot(x='PC1', y='PC2', hue='Grupo', data=X_pca)

            if mostra_labels == True:
                #display (X_pca)
                for i in df.index:
                    #print (i, X_pca['PC1'].loc[i], X_pca['PC2'].loc[i])
                    ax.annotate(X_pca['ciclo'].loc[i], (X_pca['PC1'].loc[i], X_pca['PC2'].loc[i]))

            plt.show()
            scores_plot = 'scores_plot.png'
            fig.savefig(scores_plot)
            st.image(scores_plot, caption= 'SCORES PLOT', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")



            #####################################################################################
            ########################## FAZ O BIPLOT ############################################
            print('______________________________________________________________________')
            st.write ('________________________  BI-PLOT _______________________________')
            from yellowbrick.features import PCA
            rcParams['figure.figsize'] = 10,7
            fig, ax = plt.subplots()
            # Specify the features of interest and the target
            #visualizer = PCA(scale=True, classes=X_pca['Grupo'].unique(),proj_features=True)
            visualizer = PCA(scale=True, proj_features=True)
            #visualizer.fit_transform(df_std, X_pca['Grupo'])
            visualizer.fit_transform(df_std)
            visualizer.show(outpath="bi_plot.png")
            st.image("bi_plot.png", caption= 'BI-PLOT', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

            #####################################################################################
            ########################## FAZ O dendograma ############################################
            df_std[Ylabel]=df[Ylabel]
            df_std['ciclo']=df['ciclo']
            df_std = df_std.groupby(by=['tratamento','ciclo']).mean()
            #df_std = df_std.groupby(by=['tratamento']).mean()
            #df_std.set_index(['tratamento','ciclo'], inplace=True)
            #for metodo in ['single','average','weighted','centroid','median','ward']:
            Z = HIERARQUICO(df_std,metodo='ward')

    return #scores, cargas_fatoriais #LOADINGS    



def HIERARQUICO(df, metodo):
    import streamlit as st

    if metodo == 'single':
        print ('[single]','average','weighted','centroid','median','ward')
    elif metodo == 'average':
        print ('single','[average]','weighted','centroid','median','ward')
    elif metodo == 'weighted':
        print ('single','average','[weighted]','centroid','median','ward')
    elif metodo == 'centroid':
        print ('single','average','weighted','[centroid]','median','ward')
    elif metodo == 'median':
        print ('single','average','weighted','centroid','[median]','ward')
    else:
        print ('single','average','weighted','centroid','median','[ward]')



    import matplotlib.pyplot as plt
    
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    
    ###############################################
    ############# FAZ O DENDROGRAMA
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 15, 20
    fig, ax = plt.subplots()
    Z = linkage(df, method = metodo)
    dendrograma = dendrogram(Z, labels=list(df.index),orientation='right')
    plt.title('Dendrograma - método: '+ metodo)
    plt.ylabel('tratamento')
    plt.xlabel('Distância Euclidiana')
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=12)
    plt.show()

    cluste_h = 'agrupamento.png'
    plt.savefig(cluste_h, format = 'png',dpi=200, facecolor='w', edgecolor='b',
                orientation='portrait', papertype=None, #format=None,
                transparent=False, bbox_inches='tight', pad_inches=0.2,
                frameon=None, metadata=None)
    
    st.image(cluste_h, caption= 'Agrupamento Hierarquico', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    return Z

##################################################################################################################
#################### FIM DO CLUSTER



# FAZ O TESTE DE ESFERICIDADE DE BARBETT
def teste_de_esfericidade_barbett(df):
    import streamlit as st
    from scipy.stats.distributions import chi2
    import numpy as np
    rho = df.corr()
    n=df.index.size
    k=rho.index.size
    gl=(k*(k-1))/2
    A = -((n-1)-((2*k+5)/6))
    B = np.log(np.linalg.det(rho ))
    chi=A*B
    pv=chi2.sf(chi,gl)
    st.write('k =',k," gl=",gl," n=",n)
    
    st.write('Teste da Esfericidade de Bartlett: chi² =', chi,  ', p_value =' , pv)
    if pv < 0.05:
        st.write ('Nossos dados são adequados para fazer uma PCA')
    else:
        st.write ('Nossos dados NÃO são adequados para fazer uma PCA')

    return chi, pv

def padroniza(df):
    import streamlit as st
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd
    #################################################
   
    #### pré-tratamento dos dados

    #scaler = StandardScaler()
    #base = scaler.fit_transform(df) 
    #df_std=pd.DataFrame(base, index=df.index, columns = df.columns)
    
    df_std = (df - df.mean())/df.std()

    return df_std
    ################################################
    ########################################################

#calculo do ranking
def ranking(scores,cargas_fatoriais,PC,sinais):
    import streamlit as st
    import numpy as np
    import pandas as pd
    """ esta rotina serne para rankear as obserbações baseados nos scores. 
    Para isso tem que entrar com os scores, as carcas_fatoriais, o número de componentes que tem EigenValues maiores que 1 
    e os fatores para inverter os sinais dos scores, com base na coerencia. Fatores que agrupam Variáveis que tem caracter positivo devem ter Cargas fatoriais positivas. E vice-versa
    Aparentemente, tudo leva a creer os fatores 2,3 e 4 são invertidos
    Assim sinais devem ser = [1,-1,-1,-1] """
    pcs=np.arange(1,PC+1)
    variancia_compartilhada = ((cargas_fatoriais**2).sum(axis=0))/(( cargas_fatoriais**2).sum(axis=0)).sum()
    Ranking = (((scores[pcs]*sinais)*variancia_compartilhada[:PC]).sum(axis=1)).sort_values(ascending=False)
    return Ranking

################################################
##FUNÇÃO PARA ROTACIONAR OS EIXOS DAS CARGAS FATORIAIS

def varimax(CARGAS_FATORIAIS):
    import streamlit as st

    import pandas as pd
    import numpy as np

    colunas =[CARGAS_FATORIAIS.columns[0],CARGAS_FATORIAIS.columns[1]]
    CARGAS_FATORIAIS_np = np.array(CARGAS_FATORIAIS[colunas])
    #print(CARGAS_FATORIAIS_np)
    ### ROTACIONA OS EIXOS #####################
    from statsmodels.multivariate.factor_rotation import rotate_factors
    L, T = rotate_factors(CARGAS_FATORIAIS_np,'varimax')
    np.round(L,3), T


    

    ############################################################################################
    print('______________________________________________________________________')
    print('__________________________ LOADING PLOT ______________________________')
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 10, 10
    #ax = sns.scatterplot(x=CARGAS_FATORIAIS[CARGAS_FATORIAIS.columns[0]], y=CARGAS_FATORIAIS[CARGAS_FATORIAIS.columns[1]], marker="o")
    for i in CARGAS_FATORIAIS.index:
        pass#ax.annotate(i,(CARGAS_FATORIAIS[1].loc[i],CARGAS_FATORIAIS[2].loc[i]))
    #plt.show()
    #############################################################################################


    print('______________________________________________________________________')
    Loadings_rotacionados = pd.DataFrame(L ,index = CARGAS_FATORIAIS.index)
    Loadings_rotacionados.columns = Loadings_rotacionados.columns+1
    Loadings_rotacionados   #eigenvector

    ############################################################################################
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 10, 10
    ax = sns.scatterplot(x=Loadings_rotacionados[1], y=Loadings_rotacionados[2], marker="o")
    for i in Loadings_rotacionados.index:
        ax.annotate(i,(Loadings_rotacionados[1].loc[i],Loadings_rotacionados[2].loc[i]))
    plt.show()
    return Loadings_rotacionados



def removerAcentosECaracteresEspeciais(palavra):
    import streamlit as st
    import unicodedata
    import re

    """
    A remoção de acentos foi baseada em uma resposta no Stack Overflow.
    http://stackoverflow.com/a/517974/3464573
    """
    # Unicode normalize transforma um caracter em seu equivalente em latin.
    nfkd = unicodedata.normalize('NFKD', palavra)
    palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])
    
    # Usa expressão regular para retornar a palavra apenas com números, letras e espaço
    return re.sub('[^a-zA-Z0-9 \\\]', '_', palavraSemAcento)