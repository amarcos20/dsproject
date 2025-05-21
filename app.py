import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt # Removido, usando Plotly para interactividade
import seaborn as sns # Mantido para paletas de cores se necessário, mas Plotly é o foco
# Removido: Não precisamos de importar os modelos e split/metrics do sklearn diretamente para TREINAR na carga
# from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay # Para plotar CM no streamlit, se quiser substituir Plotly
# Importar os tipos de modelos necessários para a secção Análise de Matriz (treino temporário)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier # Adicionado KNN, estava no seu notebook
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier # Adicionado Decision Tree, estava no seu notebook
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier # Adicionados Ensembles

import time
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# Novo: Importar joblib e os para carregar artefactos e gerir caminhos
import joblib
import os

# --- Configuração da Página ---
st.set_page_config(
    page_title="Sistema de Intervenção Estudantil", # Ajustado o título
    page_icon="📊",
    layout="wide", # Use wide layout for better use of space
    initial_sidebar_state="expanded"
)

# --- Estilo CSS Personalizado ---
# Mantido e ligeiramente ajustado para consistência
st.markdown("""
<style>
    /* Headers */
    .main-header {
        font-size: 2.8rem; /* Increased size */
        color: #1A237E; /* Darker Blue */
        text-align: center;
        margin-bottom: 1.5rem; /* Increased margin */
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 2rem; /* Increased size */
        color: #283593; /* Slightly lighter */
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
        border-bottom: 2px solid #C5CAE9; /* Light underline */
        padding-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
        margin-bottom: 1rem;
        line-height: 1.6; /* Improved readability */
    }
    /* Cards */
    .metric-card {
        background-color: #E8EAF6; /* Very light blue */
        border-left: 6px solid #3F51B5; /* Indigo border */
        border-radius: 10px; /* More rounded corners */
        padding: 1.5rem;
        margin-bottom: 1.5rem; /* Increased margin */
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); /* Stronger shadow */
    }
     /* Style for st.metric value - Streamlit's built-in metric uses different classes */
    div[data-testid="stMetric"] label div { /* Targeting the label div in st.metric */
        font-size: 1rem !important; /* Adjust label size */
        color: #555 !important;
    }
     div[data-testid="stMetric"] div[data-testid="stMetricDelta"] div { /* Targeting the value div */
         font-size: 1.8rem !important; /* Larger metric value */
         font-weight: bold !important;
         color: #1A237E !important; /* Darker blue */
     }
    /* Button */
    .stButton > button {
        background-color: #3F51B5; /* Indigo */
        color: white;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
        font-size: 1.1rem; /* Larger button text */
    }
    .stButton > button:hover {
        background-color: #303F9F; /* Darker Indigo */
    }
     /* Adjust sidebar width */
    section[data-testid="stSidebar"] {
        width: 300px !important;
        background-color: #f1f3f4; /* Sidebar background */
    }
    /* Style for tabs */
    .stTabs [data-baseweb="tab-list"] {
		gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
		height: 50px;
		white-space: pre-wrap;
		background-color: #E8EAF6; /* Light background */
		border-radius: 4px 4px 0 0;
		gap: 10px;
		padding: 10px 20px; /* Adjust padding */
        font-size: 1rem;
        font-weight: bold;
    }

    .stTabs [data-baseweb="tab"] svg {
		color: #3F51B5; /* Icon color */
    }

    .stTabs [data-baseweb="tab"]:hover {
		background-color: #C5CAE9; /* Hover background */
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
		background-color: #3F51B5; /* Selected tab background */
		color: white; /* Selected text color */
		border-bottom: 3px solid #FFC107; /* Accent color underline */
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] svg {
		color: white; /* Selected icon color */
    }
     /* Style for st.info, st.warning, st.error */
    div[data-testid="stAlert"] {
        font-size: 1rem;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Background Function (Optional) ---
# def add_bg_from_base64(base64_string): ...

# Função para exibir animação de carregamento
def loading_animation(text="Processando..."):
    progress_text = text
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(0.5)
    my_bar.empty()

# Função para gerar matriz de confusão interativa (mantida do código original)
def plot_confusion_matrix_interactive(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        showscale=True,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        hoverinfo="x+y+z",
    ))

    fig.update_layout(
        title='Matriz de Confusão',
        xaxis_title='Valores Previstos',
        yaxis_title='Valores Reais',
        xaxis=dict(side='top'),
        yaxis=dict(autorange="reversed"),
        margin=dict(t=50, b=50, l=50, r=50),
    )

    return fig, cm

# Função para plotar matriz quadrada com mapa de calor (mais genérica, mantida)
def plot_square_matrix_heatmap(matrix, title="Matriz Quadrada", x_labels=None, y_labels=None):
    matrix_list = [[None if pd.isna(val) else float(val) for val in row] for row in matrix]
    text_matrix = [[None if pd.isna(val) else f"{val:.2f}" for val in row] for row in matrix]

    fig = go.Figure(data=go.Heatmap(
        z=matrix_list,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',
        showscale=True,
        text=text_matrix,
        texttemplate="%{text}",
        hoverinfo="x+y+z",
    ))

    fig.update_layout(
        title=title,
        margin=dict(t=50, b=50, l=50, r=50),
    )

    return fig

# Função para visualizar matriz de correlação (usando plotly express, mantida)
def plot_correlation_matrix_px(df):
    df_numeric = df.select_dtypes(include=np.number)

    if df_numeric.empty:
         return None, None

    corr = df_numeric.corr()

    fig = px.imshow(
        corr,
        labels=dict(color="Correlação"),
        x=corr.columns,
        y=corr.columns,
        color_continuous_scale='RdBu_r',
        range_color=[-1, 1],
        aspect="auto",
        text_auto=".2f",
    )

    fig.update_layout(
        title="Matriz de Correlação",
        margin=dict(t=50, b=50, l=50, r=50),
    )

    return fig, corr

# Função para analisar propriedades de uma matriz quadrada (mantida)
def analyze_square_matrix(matrix, title="Análise de Matriz"):
    st.markdown(f'<h3 class="sub-header">{title}</h3>', unsafe_allow_html=True)

    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        st.error("Input inválido: A matriz deve ser um array NumPy quadrado e 2D.")
        return

    size = matrix.shape[0]
    st.write(f"**Dimensão da matriz:** {size}x{size}")

    trace = np.trace(matrix)
    st.write(f"**Traço da matriz:** {trace:.4f}")
    st.info("O traço é a soma dos elementos na diagonal principal da matriz.")

    try:
        det = np.linalg.det(matrix)
        st.write(f"**Determinante:** {det:.4e}")
        if abs(det) < 1e-9:
            st.warning("⚠️ O determinante é próximo de zero...")
        else:
             st.success("✅ O determinante sugere que a matriz não é singular.")
        st.info("O determinante indica se a matriz é invertível...")
    except np.linalg.LinAlgError:
        st.error("❌ Não foi possível calcular o determinante...")
        det = None

    st.write("**Valores próprios (Eigenvalues):**")
    try:
        eigenvalues = np.linalg.eigvals(matrix)
        sorted_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

        for i, val in enumerate(sorted_eigenvalues):
            original_val = eigenvalues[np.where(np.isclose(np.abs(eigenvalues), val))[0][0]]
            st.write(f"λ{i+1} (Magnitude) = {val:.4f} (Original: {original_val:.4f})")

        if any(val < 1e-9 for val in sorted_eigenvalues):
             st.warning("⚠️ Alguns valores próprios são próximos de zero...")
        else:
             st.success("✅ Os valores próprios indicam que a matriz não tem direções nulas...")
        st.info("Valores próprios representam os fatores de escala...")

    except np.linalg.LinAlgError:
        st.error("❌ Não foi possível calcular os valores próprios.")

    try:
        condition_number = np.linalg.cond(matrix)
        st.write(f"**Número de Condição:** {condition_number:.4e}")
        if condition_number > 1000:
            st.warning("⚠️ Alto número de condição. A matriz é mal condicionada...")
        else:
            st.success("✅ Número de condição razoável. A matriz está bem condicionada.")
        st.info("O número de condição mede a sensibilidade...")
    except np.linalg.LinAlgError:
         st.error("❌ Não foi possível calcular o número de condição.")
    except Exception as e:
         st.error(f"❌ Erro ao calcular número de condição: {e}")


# --- Carregar Artefactos Treinados (Refatorado para retornar status e resultado) ---
@st.cache_resource
def load_pipeline_artefacts_safe():
    # Caminho corrigido para a pasta artifacts (agora está no mesmo nível do script)
    artefacts_path = 'artefacts/'
    preprocessor_path = os.path.join(artefacts_path, 'preprocessor.joblib')
    # Use o nome exato do ficheiro do seu modelo treinado final
    model_path = os.path.join(artefacts_path, 'best_model.joblib') # Ajuste este nome se necessário
    # Caminhos para os ficheiros de nomes de colunas
    original_cols_path = os.path.join(artefacts_path, 'original_input_columns.joblib')
    # Ajuste o nome do ficheiro de features processadas se for diferente
    # Use o nome que aparece na sua imagem: processed_feature_names.joblib (sem after_onehot)
    processed_cols_path = os.path.join(artefacts_path, 'processed_feature_names.joblib') # Nome CORRIGIDO

    try:
        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
        original_cols = joblib.load(original_cols_path)
        processed_cols = joblib.load(processed_cols_path) # Carregar com o nome corrigido

        st.success("✅ Artefactos do pipeline (pré-processador, modelo e nomes de colunas) carregados com sucesso!")
        # Em caso de sucesso, retorna True e os 4 objetos numa tupla
        return True, (preprocessor, model, original_cols, processed_cols)

    except FileNotFoundError as e:
        error_msg = f"❌ Erro ao carregar artefactos essenciais: {e}. Certifique-se de que todos os ficheiros .joblib estão na pasta '{artefacts_path}' e têm os nomes corretos."
        # Em caso de FileNotFoundError, retorna False e a mensagem de erro
        return False, error_msg
    except Exception as e:
        error_msg = f"❌ Ocorreu um erro inesperado ao carregar artefactos: {e}"
        # Em caso de qualquer outro erro, retorna False e a mensagem de erro
        return False, error_msg

# --- Chamar a função de carregamento e verificar o resultado ---
success_artefacts, loaded_artefacts_result = load_pipeline_artefacts_safe()

# Se não foi sucesso, exibir o erro e parar a aplicação
if not success_artefacts:
    st.error(loaded_artefacts_result) # loaded_artefacts_result contém a mensagem de erro
    st.stop() # Parar a execução da app se os artefactos essenciais não carregarem
else:
    # Se foi sucesso, desempacotar os 4 objetos do resultado da tupla
    preprocessor, model, original_cols, processed_cols = loaded_artefacts_result


# --- Carregar o seu Dataset Original para EDA ---
# Use st.cache_data para carregar os dados apenas uma vez
@st.cache_data
def load_student_data():
    data_path = 'student-data.csv' # Assumindo que está no mesmo nível do script e da pasta data/artifacts
    try:
        df = pd.read_csv(data_path)
        st.success(f"✅ Dataset '{data_path}' carregado com sucesso ({df.shape[0]} linhas, {df.shape[1]} colunas).")
        return df
    except FileNotFoundError:
        st.error(f"❌ Erro: O ficheiro '{data_path}' não foi encontrado. Certifique-se de que o dataset está no local correto.")
        st.stop() # Parar a execução se o dataset não for encontrado
    except Exception as e:
        st.error(f"❌ Ocorreu um erro ao carregar o dataset: {e}")
        st.stop() # Parar a execução em caso de outro erro

# Carregar o dataset original
student_df_original = load_student_data()

# Identificar a coluna alvo original
TARGET_ORIGINAL_NAME = 'passed' # Nome da coluna alvo no dataset original
if TARGET_ORIGINAL_NAME not in student_df_original.columns:
    st.error(f"❌ Coluna alvo original '{TARGET_ORIGINAL_NAME}' não encontrada no dataset. A aplicação pode não funcionar corretamente.")
    # Opcional: st.stop() # parar se a coluna alvo não for encontrada no dataset original


# Definir os nomes das classes para a saída da previsão e avaliação
# No seu notebook, 0 foi mapeado para 'no' e 1 para 'yes'.
CLASS_NAMES = ['no', 'yes'] # Correspondem aos valores 0 e 1

# Definir o nome da coluna alvo APÓS o mapeamento (usado no teste processado)
TARGET_PROCESSED_NAME = 'passed_mapped'


# --- Função para carregar os conjuntos de dados processados (treino e teste) ---
@st.cache_data
def load_processed_data(target_col_name):
    # Caminhos para os ficheiros processados (data/processed deve estar no mesmo nível do script)
    processed_train_path = 'data/processed/train_processed.csv'
    processed_test_path = 'data/processed/test_processed.csv'

    train_df_processed = None
    test_df_processed = None
    errors = []

    try:
        train_df_processed = pd.read_csv(processed_train_path)
        if target_col_name not in train_df_processed.columns:
             errors.append(f"❌ Erro: A coluna alvo processada '{target_col_name}' não foi encontrada no ficheiro '{processed_train_path}'.")
             train_df_processed = None # Invalidar o dataframe de treino se a coluna alvo estiver faltando
        else:
             st.success(f"✅ Conjunto de treino processado carregado ({train_df_processed.shape[0]} linhas).")
    except FileNotFoundError:
        errors.append(f"⚠️ Ficheiro de treino processado '{processed_train_path}' não encontrado. Algumas funcionalidades podem estar limitadas.")
    except Exception as e:
        errors.append(f"❌ Ocorreu um erro ao carregar o conjunto de treino processado: {e}")
        train_df_processed = None


    try:
        test_df_processed = pd.read_csv(processed_test_path)
        if target_col_name not in test_df_processed.columns:
             errors.append(f"❌ Erro: A coluna alvo processada '{target_col_name}' não foi encontrada no ficheiro '{processed_test_path}'.")
             test_df_processed = None # Invalidar o dataframe de teste se a coluna alvo estiver faltando
        else:
             st.success(f"✅ Conjunto de teste processado carregado ({test_df_processed.shape[0]} linhas).")
    except FileNotFoundError:
        errors.append(f"⚠️ Ficheiro de teste processado '{processed_test_path}' não encontrado. Algumas funcionalidades podem estar limitadas.")
    except Exception as e:
        errors.append(f"❌ Ocorreu um erro ao carregar o conjunto de teste processado: {e}")
        test_df_processed = None

    # Exibir todos os erros ou avisos acumulados
    for err in errors:
        st.markdown(err) # Usar markdown para permitir ícones

    return train_df_processed, test_df_processed

# Carregar os conjuntos de treino e teste processados
train_df_processed_global, test_df_processed_global = load_processed_data(TARGET_PROCESSED_NAME)


# --- Lista de modelos disponíveis para a secção "Análise de Matriz" ---
# Estes são tipos de modelos que podem ser instanciados e treinados na hora
AVAILABLE_MODELS_FOR_ANALYSIS = {
    "Regressão Logística": LogisticRegression(random_state=42, max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM (Kernel RBF)": SVC(probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}


# --- Sidebar para navegação ---
with st.sidebar:
    # st.image("https://cdn-icons-png.flaticon.com/512/2103/2103658.png", width=100) # Substituir por uma imagem tua
    st.markdown('<h1 class="sub-header" style="text-align: center;">Sistema de Intervenção Estudantil</h1>', unsafe_allow_html=True) # Ajustado Título

    menu = option_menu(
        menu_title=None, # hide menu title
        options=["Início", "Exploração de Dados", "Previsão Individual", "Análise do Modelo Treinado", "Análise de Matriz", "Documentação"], # Ajustado opções
        icons=["house-door", "bar-chart-line", "clipboard-data", "robot", "grid-3x3", "book"], # Ícones correspondentes
        menu_icon="cast", # Ícone geral do menu
        default_index=0, # Página inicial por defeito
    )

    st.markdown("---")
    st.markdown("### Sobre a Aplicação")
    st.info("""
    Ferramenta interativa para explorar o dataset estudantil, fazer previsões
    individuais e analisar o modelo de Machine Learning treinado e suas propriedades.
    """) # Ajustado descrição

    st.markdown("### Desenvolvido com ❤️")
    st.write("Framework: Streamlit")
    st.write("Linguagem: Python")
    st.write("Bibliotecas: scikit-learn, pandas, numpy, plotly, joblib")


# --- Conteúdo Principal ---

if menu == "Início":
    st.markdown('<h1 class="main-header">Bem-vindo ao Sistema de Intervenção Estudantil 🚀</h1>', unsafe_allow_html=True)

    st.markdown('<p class="info-text">Este aplicativo é uma ferramenta interativa baseada no seu modelo de Machine Learning para prever o desempenho estudantil, usando o dataset "UCI Student Performance".</p>', unsafe_allow_html=True) # Ajustado texto

    # Ajustar métricas de resumo na página inicial
    col1, col2, col3 = st.columns(3)

    # Número de amostras no seu dataset
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Amostras no Dataset", f"{student_df_original.shape[0]}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Número de características originais (usando a lista carregada)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'original_cols' in locals() and original_cols is not None:
             st.metric("Características Originais", f"{len(original_cols)}")
        else:
             st.metric("Características Originais", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    # Status do carregamento do modelo/preprocessor
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        # A verificação de sucesso já foi feita no início, se chegamos aqui, carregou.
        st.metric("Status do Pipeline", "Carregado ✅")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<h2 class="sub-header">Funcionalidades:</h2>', unsafe_allow_html=True)

    st.markdown("""
    *   **Exploração de Dados:** Visualize resumos, distribuições e correlações do dataset original.
    *   **Previsão Individual:** Insira dados de um aluno e obtenha uma previsão do seu desempenho final usando o modelo treinado.
    *   **Análise do Modelo Treinado:** Veja as métricas de avaliação e a matriz de confusão do modelo carregado no conjunto de teste.
    *   **Análise de Matriz:** Explore visualmente e analiticamente propriedades de matrizes relevantes (Confusão de *qualquer* modelo, Correlação/Covariância dos seus dados, Matriz Personalizada). # Ajustado texto
    *   **Documentação:** Encontre mais informações sobre a aplicação e o projeto.
    """)


# --- Exploração de Dados (Adaptado para o seu dataset) ---
elif menu == "Exploração de Dados":
    st.markdown('<h1 class="main-header">Exploração do Dataset Estudantil</h1>', unsafe_allow_html=True)

    df = student_df_original.copy()

    st.markdown('<p class="info-text">Analise a estrutura, distribuição e relações entre as características do seu dataset de dados estudantis (`student-data.csv`).</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Resumo Geral", "📈 Distribuições", "🔍 Relações"])

    with tab1:
        st.markdown('<h2 class="sub-header">Resumo Geral do Dataset</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Dimensões do Dataset:**", df.shape)
            if 'original_cols' in locals() and original_cols is not None:
                 st.write(f"**Características (Features):** {len(original_cols)}")
            else:
                 st.warning("Nomes das características originais não carregados.")
                 st.write(f"**Características (Features):** {df.shape[1] - (1 if TARGET_ORIGINAL_NAME in df.columns else 0)}")

            st.write(f"**Amostras:** {df.shape[0]}")

            if TARGET_ORIGINAL_NAME in df.columns:
                 st.write(f"**Variável Alvo:** '{TARGET_ORIGINAL_NAME}'")
                 unique_target_values = df[TARGET_ORIGINAL_NAME].unique().tolist()
                 st.write(f"**Classes:** {', '.join(map(str, unique_target_values))}")

            st.markdown('---')
            st.write("**Primeiras 5 Linhas:**")
            st.dataframe(df.head(), use_container_width=True)

        with col2:
             if TARGET_ORIGINAL_NAME in df.columns:
                 st.write(f"**Distribuição da Coluna '{TARGET_ORIGINAL_NAME}':**")
                 class_counts = df[TARGET_ORIGINAL_NAME].value_counts()
                 fig_pie = px.pie(
                     values=class_counts.values,
                     names=class_counts.index.tolist(),
                     title=f"Distribuição de '{TARGET_ORIGINAL_NAME}'",
                     hole=0.3
                 )
                 fig_pie.update_layout(legend_title_text=TARGET_ORIGINAL_NAME.replace('_', ' ').title())
                 st.plotly_chart(fig_pie, use_container_width=True)
             else:
                  st.info(f"Não é possível mostrar a distribuição da coluna alvo '{TARGET_ORIGINAL_NAME}'.")

        st.markdown('<h2 class="sub-header">Estatísticas Descritivas</h2>', unsafe_allow_html=True)
        st.dataframe(df.describe(include='all'), use_container_width=True)

    with tab2:
        st.markdown('<h2 class="sub-header">Distribuição das Características</h2>', unsafe_allow_html=True)
        st.markdown('<p class="info-text">Visualize a distribuição de cada característica do seu dataset.</p>', unsafe_allow_html=True)

        feature_options_dist = original_cols if 'original_cols' in locals() and original_cols is not None else df.columns.tolist()
        if TARGET_ORIGINAL_NAME in feature_options_dist:
             feature_options_dist.remove(TARGET_ORIGINAL_NAME)

        selected_feature_dist = st.selectbox(
            "Selecione uma característica para visualizar a distribuição:",
            options=feature_options_dist
        )

        if selected_feature_dist:
             dtype = df[selected_feature_dist].dtype
             if dtype in [np.number, 'int64', 'float64']:
                  fig_hist = px.histogram(
                      df,
                      x=selected_feature_dist,
                      marginal="box",
                      title=f'Distribuição de "{selected_feature_dist}"'
                  )
                  st.plotly_chart(fig_hist, use_container_width=True)
             elif dtype == 'object' or pd.api.types.is_categorical_dtype(df[selected_feature_dist]):
                  counts_df = df[selected_feature_dist].value_counts().reset_index()
                  counts_df.columns = [selected_feature_dist, 'Count']
                  fig_bar = px.bar(
                      counts_df,
                      x=selected_feature_dist,
                      y='Count',
                      title=f'Distribuição de "{selected_feature_dist}"'
                  )
                  st.plotly_chart(fig_bar, use_container_width=True)
             else:
                 st.info(f"A característica '{selected_feature_dist}' tem um tipo de dado ({dtype}) que não é suportado para visualização de distribuição neste momento.")

    with tab3:
        st.markdown('<h2 class="sub-header">Relações entre Características</h2>', unsafe_allow_html=True)
        st.markdown('<p class="info-text">Analise a relação entre pares de características no seu dataset, coloridas pela classe alvo.</p>', unsafe_allow_html=True)

        st.markdown('### Matriz de Correlação', unsafe_allow_html=True)
        st.markdown('<p class="info-text">Veja a correlação linear entre as características numéricas.</p>', unsafe_allow_html=True)

        df_features_only = df[original_cols] if 'original_cols' in locals() and original_cols is not None else df.drop(columns=[TARGET_ORIGINAL_NAME] if TARGET_ORIGINAL_NAME in df.columns else [])
        df_numeric_for_corr = df_features_only.select_dtypes(include=np.number)

        if df_numeric_for_corr.empty:
             st.warning("Não há colunas numéricas entre as características usadas para calcular a matriz de correlação no seu dataset.")
        else:
            fig_corr, corr_matrix = plot_correlation_matrix_px(df_numeric_for_corr)
            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown('### Scatter Plot por Situação Final', unsafe_allow_html=True)
        st.markdown(f'<p class="info-text">Selecione duas características numéricas para visualizar sua relação e como a "{TARGET_ORIGINAL_NAME}" se distribui.</p>', unsafe_allow_html=True)

        numeric_cols_for_scatter_options = df_features_only.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols_for_scatter_options) < 2:
             st.info("São necessárias pelo menos duas características numéricas entre as usadas no seu dataset para o scatter plot.")
        elif TARGET_ORIGINAL_NAME not in df.columns:
            st.warning(f"Coluna alvo '{TARGET_ORIGINAL_NAME}' não encontrada no seu dataset para colorir o scatter plot.")
            col_x, col_y = st.columns(2)
            with col_x:
                 feature_x = st.selectbox("Selecione a característica X", numeric_cols_for_scatter_options, index=0, key="scatter_x_no_color")
            with col_y:
                 default_y_index = 1 if len(numeric_cols_for_scatter_options) > 1 and numeric_cols_for_scatter_options[0] == feature_x else 0
                 feature_y = st.selectbox("Selecione a característica Y", [col for col in numeric_cols_for_scatter_options if col != feature_x], index=default_y_index, key="scatter_y_no_color")

            if feature_x and feature_y:
                fig_scatter = px.scatter(
                    df,
                    x=feature_x,
                    y=feature_y,
                    title=f"Dispersão: {feature_x} vs {feature_y} (Sem Cor por Classe)",
                    opacity=0.7,
                    hover_data=[feature_x, feature_y]
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        else:
            col_x, col_y = st.columns(2)
            with col_x:
                feature_x = st.selectbox("Selecione a característica X", numeric_cols_for_scatter_options, index=0, key="scatter_x_color")
            with col_y:
                default_y_index = 1 if len(numeric_cols_for_scatter_options) > 1 and numeric_cols_for_scatter_options[0] == feature_x else 0
                feature_y = st.selectbox("Selecione a característica Y", [col for col in numeric_cols_for_scatter_options if col != feature_x], index=default_y_index, key="scatter_y_color")

            if feature_x and feature_y:
                 fig_scatter = px.scatter(
                     df,
                     x=feature_x,
                     y=feature_y,
                     color=TARGET_ORIGINAL_NAME,
                     labels={"color": TARGET_ORIGINAL_NAME.replace('_', ' ').title()},
                     title=f"Dispersão: {feature_x} vs {feature_y} por {TARGET_ORIGINAL_NAME.replace('_', ' ').title()}",
                     opacity=0.7,
                     hover_data={TARGET_ORIGINAL_NAME:False, feature_x:True, feature_y:True}
                 )
                 fig_scatter.update_layout(legend_title_text=TARGET_ORIGINAL_NAME.replace('_', ' ').title())
                 st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                 st.warning("Selecione características válidas para o scatter plot.")


# --- Nova Secção: Previsão Individual ---
elif menu == "Previsão Individual":
    st.markdown('<h1 class="main-header">Sistema de Previsão de Desempenho Estudantil</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Insira os dados de um aluno para obter a previsão se ele passará no exame final, usando o modelo treinado.</p>', unsafe_allow_html=True)

    st.info("Certifique-se de inserir os dados com precisão para obter uma previsão mais fiável.")

    st.markdown('<h2 class="sub-header">Dados do Aluno</h2>', unsafe_allow_html=True)

    if 'original_cols' not in locals() or original_cols is None:
        st.error("Não foi possível carregar os nomes das características originais. A secção de Previsão Individual não está disponível.")
    else:
        input_data = {}

        original_dtypes = student_df_original[original_cols].dtypes

        numeric_features = [col for col in original_cols if original_dtypes[col] in [np.number, 'int64', 'float64']]
        categorical_features = [col for col in original_cols if original_dtypes[col] == 'object']

        st.markdown("### Características Numéricas")
        cols_num = st.columns(4)
        col_idx = 0
        for feature in numeric_features:
            min_val = student_df_original[feature].min()
            max_val = student_df_original[feature].max()
            mean_val = student_df_original[feature].mean()

            with cols_num[col_idx % 4]:
                input_data[feature] = st.number_input(
                    f"{feature.replace('_', ' ').title()}",
                    min_value=float(min_val) if pd.notna(min_val) else 0.0,
                    max_value=float(max_val) if pd.notna(max_val) else None,
                    value=float(mean_val) if pd.notna(mean_val) else 0.0,
                    step=1.0 if original_dtypes[feature] == 'int64' else 0.1,
                    format="%f" if original_dtypes[feature] == 'float64' else "%d",
                    key=f"input_{feature}"
                )
            col_idx += 1

        st.markdown("### Características Categóricas/Binárias")
        cols_cat = st.columns(4)
        col_idx = 0
        for feature in categorical_features:
            options = student_df_original[feature].dropna().unique().tolist()

            with cols_cat[col_idx % 4]:
                 input_data[feature] = st.selectbox(
                     f"{feature.replace('_', ' ').title()}",
                     options=options,
                     index=0,
                     key=f"input_{feature}"
                 )
            col_idx += 1

        st.markdown("---")
        if st.button("🚀 Prever Resultado do Aluno"):
            input_df = pd.DataFrame([input_data], columns=original_cols)
            st.write("Dados de entrada para previsão:")
            st.dataframe(input_df, use_container_width=True)

            loading_animation("Aplicando pré-processamento...")
            try:
                input_processed = preprocessor.transform(input_df)
                st.success("✅ Pré-processamento aplicado.")

                loading_animation("Fazendo previsão...")
                prediction = model.predict(input_processed)

                y_proba_input = None
                if hasattr(model, 'predict_proba'):
                     y_proba_input = model.predict_proba(input_processed)

                predicted_class_index = prediction[0]
                predicted_class_label = CLASS_NAMES[predicted_class_index]

                st.markdown('<h2 class="sub-header">Resultado da Previsão:</h2>', unsafe_allow_html=True)

                if predicted_class_label == 'yes':
                     st.balloons()
                     st.success(f"🎉 Previsão: O aluno **PROVAVELMENTE PASSARÁ** no exame final!")
                else:
                     st.warning(f"😟 Previsão: O aluno **PROVAVELMENTE NÃO PASSARÁ** no exame final.")

                st.markdown("---")
                st.markdown("#### Detalhes da Previsão")
                st.write(f"- Classe Prevista: **{predicted_class_label}**")

                if y_proba_input is not None:
                     probability_of_yes = y_proba_input[0][CLASS_NAMES.index('yes')]
                     probability_of_no = y_proba_input[0][CLASS_NAMES.index('no')]
                     st.write(f"- Probabilidade de Passar ('yes'): **{probability_of_yes:.2f}**")
                     st.write(f"- Probabilidade de Não Passar ('no'): **{probability_of_no:.2f}**")
                else:
                     st.info("Probabilidades não disponíveis para este modelo.")

                st.info("Nota: Esta é uma previsão baseada no modelo treinado...")

            except Exception as e:
                 st.error(f"❌ Ocorreu um erro ao fazer a previsão: {e}")
                 st.warning("Verifique se todos os dados de entrada estão corretos...")


# --- Análise do Modelo Treinado (Avaliação do modelo CARREGADO) ---
elif menu == "Análise do Modelo Treinado":
    st.markdown('<h1 class="main-header">Análise do Modelo Treinado para Intervenção Estudantil</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Aqui pode ver as métricas de avaliação e a matriz de confusão do modelo (`best_model.joblib`) que foi treinado no seu dataset e guardado como artefacto.</p>', unsafe_allow_html=True)

    st.warning("⚠️ Esta secção mostra a performance do modelo PRÉ-TREINADO nos dados de teste processados, não treina um novo modelo.")

    # Verificar se os dados de teste processados foram carregados e se o modelo carregado existe
    if test_df_processed_global is None:
        st.warning("Conjunto de teste processado não foi carregado. Esta secção não está disponível. Verifique o caminho do ficheiro 'data/processed/test_processed.csv'.")
    elif model is None:
         st.error("Modelo treinado ('best_model.joblib') não foi carregado. Esta secção não está disponível.")
    elif 'processed_cols' not in locals() or processed_cols is None:
         st.error("Não foi possível carregar os nomes das características processadas. A secção de Análise do Modelo Treinado não está disponível.")
    else: # Se chegamos aqui, todos os artefactos e dados de teste processados foram carregados
        if TARGET_PROCESSED_NAME in test_df_processed_global.columns:
            X_test_processed = test_df_processed_global.drop(columns=[TARGET_PROCESSED_NAME])
            y_test_processed = test_df_processed_global[TARGET_PROCESSED_NAME]

            if st.button("Avaliar o Modelo Treinado no Conjunto de Teste"):
                loading_animation("Avaliando o modelo treinado...")
                try:
                    y_pred_loaded_model = model.predict(X_test_processed)

                    y_proba_loaded_model = None
                    if hasattr(model, 'predict_proba'):
                        y_proba_loaded_model = model.predict_proba(X_test_processed)


                    st.markdown('<h2 class="sub-header">Métricas de Avaliação no Conjunto de Teste</h2>', unsafe_allow_html=True)

                    accuracy = accuracy_score(y_test_processed, y_pred_loaded_model)
                    report_dict = classification_report(y_test_processed, y_pred_loaded_model,
                                                        target_names=CLASS_NAMES,
                                                        output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report_dict).transpose()

                    roc_auc = None
                    if y_proba_loaded_model is not None:
                         try:
                              roc_auc = roc_auc_score(y_test_processed, y_proba_loaded_model[:, 1])
                         except Exception as auc_e:
                              st.warning(f"Não foi possível calcular AUC ROC: {auc_e}. O modelo pode não ter predict_proba válido ou apenas uma classe está presente nos dados de teste.")


                    col_metrics1, col_metrics2 = st.columns(2)

                    with col_metrics1:
                        st.markdown("#### Relatório de Classificação")
                        st.dataframe(report_df.round(2), use_container_width=True)

                        st.markdown("#### Métricas Resumo")
                        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                        with col_met1: st.metric("Acurácia", f"{accuracy:.2f}")
                        with col_met2:
                             if 'weighted avg' in report_df.index:
                                 st.metric("Precisão (Avg)", f"{report_df.loc['weighted avg', 'precision']:.2f}")
                             else: st.info("N/A")
                        with col_met3:
                            if 'weighted avg' in report_df.index:
                                st.metric("Recall (Avg)", f"{report_df.loc['weighted avg', 'recall']:.2f}")
                            else: st.info("N/A")
                        with col_met4:
                            if 'weighted avg' in report_df.index:
                                st.metric("F1-Score (Avg)", f"{report_df.loc['weighted avg', 'f1-score']:.2f}")
                            else: st.info("N/A")
                        if roc_auc is not None:
                             st.metric("AUC ROC", f"{roc_auc:.2f}")
                        else:
                             st.info("AUC ROC: N/A")


                    with col_metrics2:
                         fig_cm_loaded_model, cm_matrix_loaded_model = plot_confusion_matrix_interactive(
                             y_test_processed, y_pred_loaded_model, class_names=CLASS_NAMES
                         )
                         st.plotly_chart(fig_cm_loaded_model, use_container_width=True)

                    st.markdown("---")
                    st.markdown('<h3 class="sub-header">Análise da Matriz (Matriz de Confusão)</h3>', unsafe_allow_html=True)
                    analyze_square_matrix(cm_matrix_loaded_model, title="Propriedades Matemáticas da CM")

                    if cm_matrix_loaded_model.shape == (2, 2):
                         tn, fp, fn, tp = cm_matrix_loaded_model.ravel()
                         st.write(f"**Verdadeiros Positivos (TP):** {tp}")
                         st.write(f"**Verdadeiros Negativos (TN):** {tn}")
                         st.write(f"**Falsos Positivos (FP):** {fp}")
                         st.write(f"**Falsos Negativos (FN):** {fn}")
                         st.info("""
                         *   **TP:** Previsto Passou ('yes'), Real Passou ('yes')
                         *   **TN:** Previsto Não Passou ('no'), Real Não Passou ('no')
                         *   **FP:** Previsto Passou ('yes'), Real Não Passou ('no') - Intervenção perdida...
                         *   **FN:** Previsto Não Passou ('no'), Real Passou ('yes') - Intervenção desnecessária...
                         """)
                         st.warning("💡 No contexto de intervenção estudantil, Falsos Negativos (FN) são geralmente mais críticos...")


                    st.markdown('<h3 class="sub-header">Importância das Características (Modelo Treinado)</h3>', unsafe_allow_html=True)
                    st.markdown('<p class="info-text">Quais características foram mais relevantes para a decisão do seu modelo treinado, em relação aos dados PÓS pré-processamento.</p>', unsafe_allow_html=True)

                    processed_feature_names_for_plot = processed_cols

                    if hasattr(model, 'feature_importances_'):
                        feature_importance_df = pd.DataFrame({
                            'Característica Processada': processed_feature_names_for_plot,
                            'Importância': model.feature_importances_
                        }).sort_values('Importância', ascending=False)

                        fig_importance = px.bar(
                            feature_importance_df.head(min(20, len(feature_importance_df))),
                            x='Importância',
                            y='Característica Processada',
                            orientation='h',
                            title=f"Importância das Características (Processadas) para o Modelo Treinado"
                        )
                        fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_importance, use_container_width=True)
                        st.info("A importância mostrada é para as características APÓS o pré-processamento...")

                    elif hasattr(model, 'coef_'):
                         if model.coef_.ndim == 1:
                             feature_coef_df = pd.DataFrame({
                                 'Característica Processada': processed_feature_names_for_plot,
                                 'Coeficiente': model.coef_[0]
                             }).sort_values('Coeficiente', ascending=False)

                             coef_min = feature_coef_df['Coeficiente'].min()
                             coef_max = feature_coef_df['Coeficiente'].max()
                             abs_max = max(abs(coef_min), abs(coef_max))

                             fig_coef = px.bar(
                                 feature_coef_df.head(min(20, len(feature_coef_df))),
                                 x='Coeficiente',
                                 y='Característica Processada',
                                 orientation='h',
                                 color='Coeficiente',
                                 color_continuous_scale='RdBu',
                                 range_color=[-abs_max, abs_max],
                                 title=f"Coeficientes das Características (Processadas) para o Modelo Treinado"
                             )
                             fig_coef.update_layout(yaxis={'categoryorder':'total ascending'})
                             st.plotly_chart(fig_coef, use_container_width=True)
                             st.info("Coeficientes para características APÓS pré-processamento...")

                         else:
                              st.info("O modelo treinado tem coeficientes, mas a visualização direta da importância/impacto é complexa para este caso.")

                    else:
                        st.info("O modelo treinado não fornece importância ou coeficientes de característica de forma padrão...")

                except Exception as e:
                     st.error(f"❌ Ocorreu um erro ao avaliar o modelo treinado: {e}")
                     st.warning("Verifique se o conjunto de teste processado corresponde ao formato esperado pelo modelo carregado.")

        elif test_df_processed_global is None:
             st.warning("Conjunto de teste processado não foi carregado...")


# --- Análise de Matriz (Adaptado para incluir seleção de modelo para CM) ---
elif menu == "Análise de Matriz":
    st.markdown('<h1 class="main-header">Análise de Matriz</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Explore visualmente e analiticamente diferentes tipos de matrizes importantes em Machine Learning, usando os seus dados ou escolhendo um modelo para a Matriz de Confusão.</p>', unsafe_allow_html=True)

    matrix_type = st.selectbox(
        "Selecione o tipo de matriz para análise",
        ["Matriz de Confusão (Escolher Modelo)", "Matriz de Correlação (Seu Dataset)", "Matriz de Covariância (Seu Dataset)", "Matriz Personalizada"] # Ajustado nome para CM
    )

    # Secção Matriz de Confusão (Escolher Modelo)
    if matrix_type == "Matriz de Confusão (Escolher Modelo)":
        st.markdown('<h2 class="sub-header">Matriz de Confusão de Modelo Selecionado</h2>', unsafe_allow_html=True)
        st.markdown('<p class="info-text">Selecione um tipo de modelo para treinar *temporariamente* nos seus dados processados e ver a Matriz de Confusão no conjunto de teste.</p>', unsafe_allow_html=True)

        # Verificar se os dados de treino e teste processados foram carregados
        if train_df_processed_global is None or test_df_processed_global is None:
            st.warning("Os conjuntos de treino ou teste processados não foram carregados. Não é possível gerar a Matriz de Confusão. Verifique os ficheiros em 'data/processed/'.")
        elif 'processed_cols' not in locals() or processed_cols is None:
            st.error("Não foi possível carregar os nomes das características processadas. A geração da Matriz de Confusão não está disponível.")
        else: # Se os dados processados e processed_cols foram carregados
            # Separar X e y dos dataframes processados globais
            if TARGET_PROCESSED_NAME in train_df_processed_global.columns and TARGET_PROCESSED_NAME in test_df_processed_global.columns:
                 X_train_processed = train_df_processed_global.drop(columns=[TARGET_PROCESSED_NAME])
                 y_train_processed = train_df_processed_global[TARGET_PROCESSED_NAME]
                 X_test_processed = test_df_processed_global.drop(columns=[TARGET_PROCESSED_NAME])
                 y_test_processed = test_df_processed_global[TARGET_PROCESSED_NAME]

                 # Verificar compatibilidade de colunas processadas
                 if list(X_train_processed.columns) != list(X_test_processed.columns) or list(X_train_processed.columns) != processed_cols:
                      st.error("❌ Erro de compatibilidade: As colunas dos dados de treino/teste processados não correspondem aos nomes das features processadas carregadas.")
                      st.warning("Verifique se os ficheiros em 'data/processed/' foram gerados consistentemente com os artefactos.")
                 else: # Se os dados são compatíveis
                    # Seletor de modelo
                    selected_model_name = st.selectbox(
                        "Escolha o tipo de modelo para gerar a Matriz de Confusão:",
                        list(AVAILABLE_MODELS_FOR_ANALYSIS.keys())
                    )

                    if st.button(f"Gerar Matriz de Confusão para {selected_model_name}"):
                        loading_animation(f"Treinando {selected_model_name} e gerando Matriz de Confusão...")
                        try:
                            # Instanciar o modelo selecionado (com parâmetros padrão definidos em AVAILABLE_MODELS_FOR_ANALYSIS)
                            model_instance = AVAILABLE_MODELS_FOR_ANALYSIS[selected_model_name]

                            # Treinar o modelo temporariamente nos dados de treino processados
                            model_instance.fit(X_train_processed, y_train_processed)

                            # Fazer previsões no conjunto de teste processado
                            y_pred = model_instance.predict(X_test_processed)

                            st.markdown('<h3 class="sub-header">Resultados para o Modelo Selecionado</h3>', unsafe_allow_html=True)

                            # Plotar a Matriz de Confusão
                            fig_cm, cm_matrix = plot_confusion_matrix_interactive(y_test_processed, y_pred, class_names=CLASS_NAMES)

                            col_cm_viz, col_cm_analysis = st.columns(2)

                            with col_cm_viz:
                                st.plotly_chart(fig_cm, use_container_width=True)

                            with col_cm_analysis:
                                st.markdown('<h3 class="sub-header">Análise da Matriz de Confusão</h3>', unsafe_allow_html=True)
                                st.write(f"Resultados para o modelo **{selected_model_name}** no conjunto de teste processado:")

                                # Análise VP/VN/FP/FN para binário
                                if cm_matrix.shape == (2, 2):
                                    tn, fp, fn, tp = cm_matrix.ravel()
                                    st.write(f"**Verdadeiros Positivos (TP):** {tp}")
                                    st.write(f"**Verdadeiros Negativos (TN):** {tn}")
                                    st.write(f"**Falsos Positivos (FP):** {fp}")
                                    st.write(f"**Falsos Negativos (FN):** {fn}")
                                    st.info("TP: Previsto Passou, Real Passou | TN: Previsto Não Passou, Real Não Passou | FP: Previsto Passou, Real Não Passou | FN: Previsto Não Passou, Real Passou")
                                    st.warning("💡 No contexto de intervenção estudantil, Falsos Negativos (FN) são geralmente mais críticos...")


                                st.markdown('---')
                                st.markdown('#### Propriedades da Matriz (CM como matriz genérica)')
                                analyze_square_matrix(cm_matrix, title="Propriedades Matemáticas da CM")


                        except Exception as e:
                             st.error(f"❌ Ocorreu um erro ao treinar ou avaliar o modelo: {e}")
                             st.warning("Verifique a compatibilidade entre o modelo e os dados processados.")

                 # else: # Mensagens de erro de compatibilidade já exibidas

            else: # Mensagem para quando os dados não estão prontos ou compatíveis
                pass # As mensagens de erro ou aviso já foram exibidas acima

            # else: # Mensagens de erro caso as colunas alvo não estejam nos dataframes processados
            #      if TARGET_PROCESSED_NAME not in train_df_processed_global.columns:
            #           st.error(f"A coluna alvo '{TARGET_PROCESSED_NAME}' não foi encontrada no dataframe de treino processado.")
            #      if TARGET_PROCESSED_NAME not in test_df_processed_global.columns:
            #           st.error(f"A coluna alvo '{TARGET_PROCESSED_NAME}' não foi encontrada no dataframe de teste processado.")


    # Secção Matriz de Correlação - usar o dataset original
    elif matrix_type == "Matriz de Correlação (Seu Dataset)":
        st.markdown('<h2 class="sub-header">Análise de Matriz de Correlação</h2>', unsafe_allow_html=True)
        st.markdown('<p class="info-text">Analise a correlação linear entre as características numéricas do seu dataset original (`student-data.csv`).</p>', unsafe_allow_html=True)

        df_features_only = student_df_original[original_cols] if 'original_cols' in locals() and original_cols is not None else student_df_original.drop(columns=[TARGET_ORIGINAL_NAME] if TARGET_ORIGINAL_NAME in student_df_original.columns else [])
        df_numeric_for_corr = df_features_only.select_dtypes(include=np.number)

        if df_numeric_for_corr.empty:
             st.warning("Não há colunas numéricas entre as características usadas para calcular a matriz de correlação no seu dataset.")
        else:
            fig_corr, corr_matrix = plot_correlation_matrix_px(df_numeric_for_corr)
            if fig_corr is not None and corr_matrix is not None:
                col_corr_viz, col_corr_analysis = st.columns(2)
                with col_corr_viz:
                    st.plotly_chart(fig_corr, use_container_width=True)
                with col_corr_analysis:
                    st.markdown('<h3 class="sub-header">Análise da Matriz de Correlação</h3>', unsafe_allow_html=True)
                    analyze_square_matrix(corr_matrix.values, title="Propriedades Matemáticas da Matriz de Correlação")
                    st.markdown('#### Pares com Alta Correlação', unsafe_allow_html=True)
                    st.markdown('<p class="info-text">Identifica pares de características com forte correlação linear (|r| > 0.7).</p>', unsafe_allow_html=True)
                    corr_unstacked = corr_matrix.stack().reset_index()
                    corr_unstacked.columns = ['Feature1', 'Feature2', 'Correlation']
                    high_corr_pairs = corr_unstacked[
                        (abs(corr_unstacked['Correlation']) > 0.7) &
                        (corr_unstacked['Feature1'] != corr_unstacked['Feature2'])
                    ]
                    high_corr_pairs = high_corr_pairs.loc[
                        high_corr_pairs[['Feature1', 'Feature2']].apply(lambda x: tuple(sorted(x)), axis=1).drop_duplicates().index
                    ]
                    if not high_corr_pairs.empty:
                        st.dataframe(high_corr_pairs.round(4))
                        st.warning("⚠️ Alta correlação entre características...")
                    else:
                        st.info("Não foram encontrados pares de características com correlação linear forte...")
            else:
                 st.info("Não há dados numéricos suficientes entre as características originais no seu dataset para calcular a matriz de correlação.")

    elif matrix_type == "Matriz de Covariância (Seu Dataset)":
        st.markdown('<h2 class="sub-header">Análise de Matriz de Covariância</h2>', unsafe_allow_html=True)
        st.markdown('<p class="info-text">Analise a covariância entre as características numéricas do seu dataset original (`student-data.csv`).</p>', unsafe_allow_html=True)

        df_for_cov = student_df_original[original_cols].select_dtypes(include=np.number)

        if df_for_cov.empty or df_for_cov.shape[1] < 2:
             st.info("Não há dados numéricos suficientes entre as características originais para calcular a matriz de covariância no seu dataset.")
        else:
             cov_matrix = df_for_cov.cov()
             col_cov_viz, col_cov_analysis = st.columns(2)
             with col_cov_viz:
                 fig_cov = plot_square_matrix_heatmap(
                     cov_matrix.values,
                     title="Matriz de Covariância",
                     x_labels=cov_matrix.columns,
                     y_labels=cov_matrix.columns
                 )
                 st.plotly_chart(fig_cov, use_container_width=True)
             with col_cov_analysis:
                 st.markdown('<h3 class="sub-header">Análise da Matriz de Covariância</h3>', unsafe_allow_html=True)
                 analyze_square_matrix(cov_matrix.values, title="Propriedades Matemáticas da Matriz de Covariância")
                 st.markdown('#### Interpretação da Covariância', unsafe_allow_html=True)
                 st.info("""
                 *   **Covariância Positiva:** As duas características tendem a aumentar ou diminuir juntas.
                 *   **Covariância Negativa:** Uma característica tende a aumentar enquanto a outra diminui.
                 *   **Covariância Próxima de Zero:** Pouca ou nenhuma relação linear.
                 A covariância é dependente da escala dos dados. Para uma medida sem escala, veja a Correlação.
                 Os valores na diagonal são as variâncias de cada característica.
                 """)

    elif matrix_type == "Matriz Personalizada":
        st.markdown('<h2 class="sub-header">Análise de Matriz Personalizada</h2>', unsafe_allow_html=True)
        st.markdown('<p class="info-text">Insira uma matriz quadrada manualmente ou gere uma aleatoriamente para analisar suas propriedades matemáticas.</p>', unsafe_allow_html=True)

        matrix_option = st.radio("Escolha uma opção:", ["Gerar matriz aleatória", "Inserir matriz manualmente"])
        custom_matrix = None

        if matrix_option == "Gerar matriz aleatória":
            st.markdown('### Gerar Matriz Aleatória', unsafe_allow_html=True)
            size_rand = st.slider("Dimensão da matriz", 2, 8, 3)
            random_type = st.selectbox("Tipo de matriz", ["Aleatória Geral", "Simétrica", "Diagonal", "Triangular Superior"])
            if st.button("Gerar Matriz Aleatória"):
                loading_animation("Gerando matriz aleatória...")
                if random_type == "Aleatória Geral":
                    custom_matrix = np.random.rand(size_rand, size_rand) * 10 - 5
                elif random_type == "Simétrica":
                    temp = np.random.rand(size_rand, size_rand) * 10 - 5
                    custom_matrix = (temp + temp.T) / 2
                elif random_type == "Diagonal":
                    custom_matrix = np.diag(np.random.rand(size_rand) * 10)
                elif random_type == "Triangular Superior":
                    custom_matrix = np.triu(np.random.rand(size_rand, size_rand) * 10 - 5)

        else:
            st.markdown('### Inserir Matriz Manualmente', unsafe_allow_html=True)
            st.warning("Insira os valores da matriz...")
            size_manual = st.number_input("Dimensão da matriz", 2, 6, 3, 1)
            matrix_inputs_str = []
            st.write(f"Insira {size_manual} linhas, cada uma com {size_manual} números:")
            for i in range(size_manual):
                matrix_inputs_str.append(st.text_input(f"Linha {i+1} (valores separados por vírgula ou espaço)", key=f"manual_matrix_row_{i}"))

            if st.button("Analisar Matriz Manual"):
                loading_animation("Processando matriz manual...")
                try:
                    parsed_rows = []
                    for i, row_str in enumerate(matrix_inputs_str):
                        values_str_list = [x.strip() for x in row_str.replace(',', ' ').split() if x.strip()]
                        values = [float(x) for x in values_str_list]
                        if len(values) != size_manual:
                            st.error(f"❌ Erro na Linha {i+1}: Esperava {size_manual} números...")
                            custom_matrix = None
                            break
                        parsed_rows.append(values)
                    if custom_matrix is None and len(parsed_rows) == size_manual:
                        custom_matrix = np.array(parsed_rows)
                        st.success("✅ Matriz inserida e processada com sucesso!")
                except ValueError:
                    st.error("❌ Erro ao converter valores para números...")
                    custom_matrix = None
                except Exception as e:
                    st.error(f"❌ Ocorreu um erro inesperado ao processar a matriz: {e}")
                    custom_matrix = None

        if custom_matrix is not None:
            st.markdown('---')
            col_cust_viz, col_cust_analysis = st.columns(2)
            with col_cust_viz:
                fig_cust = plot_square_matrix_heatmap(custom_matrix, title="Matriz Personalizada")
                st.plotly_chart(fig_cust, use_container_width=True)
            with col_cust_analysis:
                analyze_square_matrix(custom_matrix, title="Análise da Matriz Personalizada")


# --- Documentação (Ajustado texto) ---
elif menu == "Documentação":
    st.markdown('<h1 class="main-header">Documentação e Exemplos</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Bem-vindo à secção de documentação...</p>', unsafe_allow_html=True)

    st.markdown('<h2 class="sub-header">Sobre o Dataset</h2>', unsafe_allow_html=True)
    st.markdown(f"""
    A aplicação utiliza o seu dataset original: **`student-data.csv`**. Este dataset contém informações sobre alunos...
    """)

    st.markdown('<h2 class="sub-header">Sobre o Modelo de Previsão Confiança</h2>', unsafe_allow_html=True) # Ajustado para clarificar
    st.markdown("""
    Um modelo de classificação binária foi treinado no dataset `student-data.csv` para prever se um aluno passará ou não...
    *   O **Pré-processador** (`preprocessor.joblib`)...
    *   O **Modelo Treinado Principal** (`best_model.joblib`) é o resultado do processo de treino e otimização realizado no seu notebook e é usado para a Previsão Individual e secção de Análise.
    Pode obter previsões individuais na secção "Previsão Individual" e ver a avaliação detalhada deste modelo principal no conjunto de teste na secção "Análise do Modelo Treinado".
    """) # Adicionado detalhes sobre o pré-processador e modelo

    st.markdown('<h2 class="sub-header">Sobre a Análise de Matriz</h2>', unsafe_allow_html=True)
    st.markdown("""
    A secção "Análise de Matriz" permite visualizar e analisar propriedades matemáticas...
    *   **Matriz de Confusão (Escolher Modelo):** Permite selecionar diferentes tipos de modelos para visualizar o seu desempenho *temporário* no conjunto de teste processado. Útil para comparar o desempenho de diferentes algoritmos.
    *   **Matriz de Correlação (Seu Dataset):** Mostra a correlação linear entre pares de variáveis numéricas no seu dataset original.
    *   **Matriz de Covariância (Seu Dataset):** Semelhante à correlação, mas dependente da escala...
    *   **Matriz Personalizada:** Permite introduzir qualquer matriz quadrada...
    """) # Ajustado descrições

    st.markdown('<h2 class="sub-header">Próximos Passos e Melhorias</h2>', unsafe_allow_html=True)
    st.markdown("""
    Pode considerar as seguintes melhorias...
    """)


# --- Footer ---
st.markdown("---")
st.markdown("© 2025 Sistema de Intervenção Estudantil. Desenvolvido com Streamlit.")
