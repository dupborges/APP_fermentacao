def graficos(df,features,export,col_wrap):
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

    os.chdir(export)
    style="whitegrid"
    style ="darkgrid"
    sns.set_context("talk")
    



    #FAZ O box-plot comparando os tratamentos
    print ('FAZENDO BOXPLOT')
    with st.expander ('BOX-PLOT COMPARANDO OS TRATAMENTOS'):
        for i in features:
            palavra = removerAcentosECaracteresEspeciais(i)
            #boxplot = export+'\\boxplot_'+palavra+'.png'
            boxplot = 'boxplot_'+palavra+'.png'
            import seaborn as sns
            sns.set_theme(style=style)
            f, ax = plt.subplots(figsize=(15, 6))
            ax = sns.boxplot(y=i, x='tratamento', data=df)
            plt.show()
            fig = f.get_figure()
            fig.savefig(boxplot)

            st.image(boxplot, caption= 'Comparação entre os tratamentos: '+i, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
  
    #FAZ O GRÁFICO DE LINHA COMPARANDO OS TRATAMENTOS
    print ('FAZENDO LINEPLOT')
    with st.expander ('GRAFICO LINHA COMPARANDO OS CICLOS E TRATAMENTOS'):
        
        
        for i in features:
            palavra = removerAcentosECaracteresEspeciais(i)
            #boxplot = export+'\\lineplot_'+palavra+'.png'
            linhaplot = 'lineplot_'+palavra+'.png'
            import seaborn as sns
            sns.set_theme(style=style)
            sns.set(font_scale=1)
            g = sns.FacetGrid(df, col="tratamento",  height=3.5, aspect=.65, col_wrap=col_wrap)
            g.refline(y=df[i].median())
            g.map(sns.pointplot, "ciclo", i)
            g.fig.subplots_adjust(top=0.8)
            g.fig.suptitle(i)
            g.add_legend()
            plt.show()
            g.savefig(linhaplot)

            st.image(linhaplot, caption= 'Comparação entre os ciclos e tratamentos: '+i, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


    """ with st.expander ('PAIR-PLOT'):

        pairplot = 'pairplot.png'
        import seaborn as sns
        sns.set_theme(style=style)
        f, ax = plt.subplots(figsize=(15, 6))
        ax = sns.pairplot(data=df)
        plt.show()
        fig = f.get_figure()
        fig.savefig(pairplot)

        st.image(pairplot, caption= 'Pairplot', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")"""


    #FAZ GRÁFICO FEIOSO DE BARRAS
    print ('FAZENDO BARPLOT - FEIOSO')
    with st.expander ('GRAFICO DE BARRAS COMPARANDO OS CICLOS E TRATAMENTOS'):
                
        for i in features:
            palavra = removerAcentosECaracteresEspeciais(i)
            
            barraplot = 'barplot_'+palavra+'.png'
            import seaborn as sns
            sns.set_theme(style=style)
            sns.set(font_scale=1)
            f, ax = plt.subplots(figsize=(15, 6))
            ax = sns.barplot(y=i, x='ciclo', hue='tratamento', data=df)
            #place legend outside top right corner of plot
            plt.legend(bbox_to_anchor=(1.005, 1), loc='upper left', borderaxespad=0)
            plt.show()
            fig = f.get_figure()
            fig.savefig(barraplot)

            st.image(barraplot, caption= 'Comparação entre os tratamentos: '+i, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    print ('acabamos de fazer os gráficos')
    return


def removerAcentosECaracteresEspeciais(palavra):
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