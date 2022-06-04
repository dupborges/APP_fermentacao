call conda activate env_fermentacao
pip freeze > requirement.txt
conda env export > environment.yml --from-history

call streamlit run APP_fermentacao_1.0_.py
call conda deactivate