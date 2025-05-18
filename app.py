import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io # Importar io para capturar output de data.info()
import os
# Importar métricas e utilitários adicionais de sklearn necessários para as secções Modelos e Previsão
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score # Importar para referência, não usados diretamente na app
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder # Importar para referência no preprocessor
from sklearn.compose import ColumnTransformer # Importar para referência no preprocessor
# Importar modelos usados, se necessário para a interface de seleção ou referência
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# Pode precisar importar XGBoost se o usaste e guardaste
# import xgboost as xgb

# --- Configuração da Página ---
st.set_page_config(
    page_title="Student Intervention System",
    page_icon="📚",
    layout="wide", # Use wide layout for better use of space
    initial_sidebar_state="expanded"
)

# --- Estilo CSS Personalizado ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem; /* Increased size */
        color: #3366FF;
        text-align: center;
        margin-bottom: 1.5rem; /* Increased margin */
        font-weight: bold;
    }
    .sub-header {
        font-size: 2rem; /* Increased size */
        color: #4682B4; /* SteelBlue */
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
        border-bottom: 2px solid #4682B4; /* Add a line */
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #eef2f7; /* Light blue background */
        border-left: 5px solid #3366FF; /* Blue border */
        border-radius: 5px;
        padding: 1.5rem; /* Increased padding */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Softer shadow */
        margin-bottom: 1rem;
    }
     /* Style for st.metric value */
    .stMetric > div > div > div > div:first-child {
        font-size: 1.2rem;
        color: #3366FF;
    }
     /* Style for st.metric label */
    .stMetric > div > div > div > div:last-child {
         font-size: 0.9rem;
         color: #555;
    }
    .prediction-card {
        padding: 2rem; /* Increased padding */
        border-radius: 10px;
        margin-top: 2rem; /* Increased margin */
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .prediction-pass {
        background-color: #e8f5e9; /* Light green */
        border: 2px solid #4CAF50; /* Green */
        color: #2E7D32; /* Dark green text */
    }
    .prediction-fail {
        background-color: #ffebee; /* Light red */
        border: 2px solid #F44336; /* Red */
        color: #D32F2F; /* Dark red text */
    }
    .feature-importance-bar {
        height: 20px;
        background-color: #4682B4; /* SteelBlue */
        margin-bottom: 5px;
        border-radius: 3px;
    }
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #388E3C; /* Darker green */
    }
     /* Adjust sidebar width */
    section[data-testid="stSidebar"] {
        width: 300px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Carregamento de Dados e Modelos (Cache) ---
# Usamos st.cache_resource para carregar o modelo e preprocessor UMA VEZ
# Usamos st.cache_data para carregar os dados UMA VEZ

@st.cache_resource
def load_artefacts(artefacts_dir='artefacts'):
    """Carrega o preprocessor e o modelo treinado."""
    preprocessor_path = os.path.join(artefacts_dir, 'preprocessor.joblib')
    model_path = os.path.join(artefacts_dir, 'best_model.joblib')
    feature_names_path = os.path.join(artefacts_dir, 'processed_feature_names.joblib')

    preprocessor, model, processed_feature_names = None, None, None

    try:
        preprocessor = joblib.load(preprocessor_path)
        st.success("✅ Pré-processador carregado com sucesso!")
    except FileNotFoundError:
        st.error(f"Erro: Ficheiro do pré-processador '{preprocessor_path}' não encontrado.")
    except Exception as e:
        st.error(f"Erro ao carregar o pré-processador: {e}")

    try:
        model = joblib.load(model_path)
        st.success("✅ Modelo treinado carregado com sucesso!")
    except FileNotFoundError:
        st.error(f"Erro: Ficheiro do modelo '{model_path}' não encontrado.")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")

    try:
        processed_feature_names = joblib.load(feature_names_path)
        st.success("✅ Nomes das features processadas carregados com sucesso!")
    except FileNotFoundError:
        st.error(f"Erro: Ficheiro com nomes das features processadas '{feature_names_path}' não encontrado.")
    except Exception as e:
        st.error(f"Erro ao carregar nomes das features processadas: {e}")


    if preprocessor is None or model is None or processed_feature_names is None:
         st.warning("Certifica-te de que executaste a script de treino/notebook para gerar e guardar todos os artefactos necessários ('artefacts' folder).")
         # Retorna None para que as secções que dependem deles saibam que não podem funcionar
         return None, None, None

    return preprocessor, model, processed_feature_names

@st.cache_data
def load_data(file_path='student-data.csv'):
    """Carrega os dados originais."""
    try:
        data = pd.read_csv(file_path)
        st.success(f"✅ Dados originais '{file_path}' carregados com sucesso!")
        # Adicionar mapeamento da coluna alvo se o 'passed' estiver no CSV original
        if 'passed' in data.columns:
            data['passed_mapped'] = data['passed'].map({'yes': 1, 'no': 0})
        return data
    except FileNotFoundError:
        st.error(f"Erro: Ficheiro de dados '{file_path}' não encontrado.")
        st.warning("Gerando dados fictícios para demonstração. A previsão não será baseada no modelo treinado com os teus dados.")
        return generate_mock_data()
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        st.warning("Gerando dados fictícios para demonstração.")
        return generate_mock_data()


# Função para gerar dados fictícios (usada se o CSV original não for encontrado)
# Adicionei mais colunas para que se assemelhe mais ao teu dataset real
def generate_mock_data():
    """Gera dados fictícios para demonstração quando o arquivo original não é encontrado."""
    st.info("Gerando dados fictícios...")
    np.random.seed(42)
    n_samples = 50 # Reduzi o número para mock data

    # Nomes de colunas baseados no teu notebook
    cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
            'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
            'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
            'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
            'passed', 'G1', 'G2', 'G3'] # Incluir G1, G2, G3 e passed

    data = pd.DataFrame(index=range(n_samples), columns=cols)

    # Preencher com dados aleatórios (tentando simular tipos originais)
    for col in cols:
        if col in ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                   'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']:
            # Colunas categóricas (Object) - usar valores comuns do teu EDA
            if col == 'school': options = ['GP', 'MS']
            elif col == 'sex': options = ['F', 'M']
            elif col == 'address': options = ['U', 'R']
            elif col == 'famsize': options = ['LE3', 'GT3']
            elif col == 'Pstatus': options = ['T', 'A']
            elif col == 'Mjob' or col == 'Fjob': options = ['at_home', 'health', 'other', 'services', 'teacher']
            elif col == 'reason': options = ['course', 'home', 'other', 'reputation']
            elif col == 'guardian': options = ['father', 'mother', 'other']
            elif col in ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']: options = ['yes', 'no']
            else: options = ['Cat_A', 'Cat_B'] # Fallback
            data[col] = np.random.choice(options, n_samples)

        elif col in ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout',
                     'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']:
             # Colunas numéricas (Int64) - usar ranges aproximados do teu EDA
            if col == 'age': data[col] = np.random.randint(15, 23, n_samples)
            elif col in ['Medu', 'Fedu']: data[col] = np.random.randint(0, 5, n_samples)
            elif col in ['traveltime', 'studytime', 'failures']: data[col] = np.random.randint(0, 4, n_samples)
            elif col in ['famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']: data[col] = np.random.randint(1, 6, n_samples)
            elif col == 'absences': data[col] = np.random.randint(0, 15, n_samples) # Range menor para mock
            elif col in ['G1', 'G2', 'G3']: data[col] = np.random.randint(0, 21, n_samples)
            else: data[col] = np.random.randint(0, 10, n_samples) # Fallback


    # Gerar coluna 'passed' baseada em G3 (nota final) para mock data
    # Define um threshold aleatório para simular aprovação
    mock_threshold = np.random.randint(9, 11) # threshold de 9 ou 10 para mock
    data['passed'] = np.where(data['G3'] >= mock_threshold, 'yes', 'no')
    data['passed_mapped'] = np.where(data['G3'] >= mock_threshold, 1, 0)

    return data


# --- Execução inicial (carregamento de artefactos e dados) ---
# Define TEST_SIZE para usar na descrição da secção de pré-processamento
TEST_SIZE = 0.2 # Ajusta este valor se usaste um diferente no teu notebook

preprocessor, model, processed_feature_names = load_artefacts()
data = load_data()

# Certificar que as colunas de input originais são identificadas (para a secção Previsão)
# Isto é feito após o carregamento dos dados originais
original_input_columns = []
if data is not None:
     # Excluir colunas alvo e quaisquer outras que não sejam features de input originais
     # Assumimos que 'passed' e 'passed_mapped' (se existir) são as colunas alvo a excluir
     cols_to_exclude = ['passed', 'passed_mapped']
     original_input_columns = [col for col in data.columns.tolist() if col not in cols_to_exclude]
     # Verifica a lista 'original_input_columns' para ter a certeza que contém as 30 colunas de input esperadas


# --- Sidebar ---
with st.sidebar:
    # st.image("https://via.placeholder.com/150x150.png?text=SIS", width=150) # Substituir por uma imagem tua se quiseres
    st.title("Student Intervention System")

    # Menu de navegação com ícones
    page = st.radio(
        "Menu",
        ["🏠 Início", "📊 EDA", "🔍 Pré-processamento", "🧠 Modelos e Avaliação", "🔮 Previsão"],
        #label_visibility="collapsed" # Uncomment if you want to hide the label "Menu"
    )

    st.markdown("---")
    st.markdown("### Sobre o Projeto")
    st.info("""
    Sistema de Intervenção Estudantil para prever o sucesso académico
    (passar/reprovar) e identificar alunos em risco.
    """)

    st.markdown("### Autor")
    st.markdown("Afonso Miguel Vieira Marcos - 202404088")
    st.markdown("---")
    st.write("Framework: Streamlit")
    st.write("Versão Streamlit:", st.__version__)
    # Check if joblib is loaded successfully before printing version
    try:
         # Tentar carregar um modelo dummy apenas para obter a versão do sklearn associada ao joblib
         # Isto pode falhar se joblib não estiver instalado ou se houver problemas
         sklearn_version = joblib.__version__
         st.write("Versão scikit-learn:", sklearn_version)
    except Exception as e:
         st.write("Versão scikit-learn: N/A")
         # st.warning(f"Could not get scikit-learn version from joblib: {e}") # Debugging info


# --- Conteúdo Principal ---

if page == "🏠 Início":
    st.markdown('<h1 class="main-header">Student Intervention System</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Bem-vindo à Aplicação Streamlit
        
        Esta aplicação demonstra a **análise, processamento e modelagem de dados** de estudantes para prever 
        se um estudante irá passar ou reprovar, com base em características académicas e sociais.
        
        ### Objetivo do Projeto
        O objetivo principal é criar um sistema preditivo que possa identificar precocemente alunos em risco
        de reprovação, permitindo intervenções direcionadas para melhorar seu desempenho.
        
        ### Características do Dataset
        O conjunto de dados original contém diversas variáveis sobre os estudantes. Pode explorar
        esses dados na secção **EDA**.
        """)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if data is not None and 'passed_mapped' in data.columns:
            total_students = len(data)
            pass_rate = (data['passed_mapped'].sum() / total_students) * 100
            num_features = len(original_input_columns)

            st.metric("Total de Alunos", f"{total_students}")
            st.metric("Taxa de Aprovação", f"{pass_rate:.1f}%")
            st.metric("Variáveis de Input", f"{num_features}")
        else:
             st.warning("Dados não carregados, métricas indisponíveis.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Como usar a aplicação
    st.markdown('<h2 class="sub-header">Como Usar Esta Aplicação</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 1. Explorar Dados")
        st.markdown("Visualize estatísticas e gráficos para entender os dados.")

    with col2:
        st.markdown("### 2. Pré-processamento")
        st.markdown("Veja como os dados são preparados para os algoritmos de ML.")

    with col3:
        st.markdown("### 3. Modelos")
        st.markdown("Compare diferentes algoritmos de classificação.")

    with col4:
        st.markdown("### 4. Previsão")
        st.markdown("Faça previsões para novos alunos em tempo real.")

elif page == "📊 EDA":
    st.markdown('<h1 class="main-header">Análise Exploratória de Dados (EDA)</h1>', unsafe_allow_html=True)

    st.write("Esta secção apresenta uma visão geral e visualizações importantes do dataset original.")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Visão Geral", "📈 Distribuições", "🔍 Correlações", "📊 Visualizações Chave"])

    if data is None:
         st.warning("Não foi possível carregar os dados para EDA. A secção está limitada ou a mostrar dados fictícios.")
         # Se os dados fictícios foram carregados, continuar com eles
         if not data.empty:
              st.dataframe(data.head()) # Mostrar pelo menos que há dados

    else: # Se os dados foram carregados (originais ou fictícios)
        with tab1:
            st.markdown('<h2 class="sub-header">Informação Geral do Dataset</h2>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Dimensões do Dataset:**", data.shape)

                # Mostrar info() de forma amigável
                st.write("**Informação do DataFrame:**")
                buffer = io.StringIO()
                data.info(buf=buffer)
                st.text(buffer.getvalue())

                # Valores ausentes
                missing_values = data.isnull().sum()
                if missing_values.sum() > 0:
                    st.write("**Valores Ausentes por Coluna:**")
                    st.dataframe(missing_values[missing_values > 0].rename("Valores Ausentes"))
                    st.warning(f"Total de valores ausentes no dataset: {missing_values.sum()}")
                else:
                    st.success("✅ Não existem valores ausentes no dataset.")

            with col2:
                st.write("**Primeiras 5 Linhas:**")
                st.dataframe(data.head(5), use_container_width=True)

                st.write("**Estatísticas Descritivas (Numéricas):**")
                st.dataframe(data.describe().round(2), use_container_width=True)

            st.markdown('<h3 class="sub-header" style="font-size: 1.5rem; border-bottom: none;">Valores Únicos por Coluna</h3>', unsafe_allow_html=True)
            unique_values_info = {}
            for col in data.columns:
                 num_unique = data[col].nunique()
                 unique_values = data[col].unique()
                 try:
                     # Tenta ordenar apenas se forem tipos compatíveis
                     sorted_unique_values = np.sort(unique_values)
                     values_str = ', '.join(map(str, sorted_unique_values))
                 except TypeError: # Se não der para ordenar (tipos mistos ou object)
                     values_str = ', '.join(map(str, unique_values))
                 # Limitar string para não ficar muito longa
                 if len(values_str) > 150:
                     values_str = values_str[:150] + '...'
                 unique_values_info[col] = {"Num Únicos": num_unique, "Valores Exemplos": values_str}
            st.dataframe(pd.DataFrame.from_dict(unique_values_info, orient='index'))


        with tab2:
            st.markdown('<h2 class="sub-header">Distribuição das Variáveis</h2>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                # Distribuição da variável alvo
                st.write("### Distribuição da Variável Alvo (passed):")

                # Calcular contagens
                if 'passed' in data.columns:
                    passed_counts = data['passed'].value_counts()
                    passed_pct = data['passed'].value_counts(normalize=True) * 100

                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.bar(['Aprovado', 'Reprovado'],
                           [passed_counts.get('yes', 0), passed_counts.get('no', 0)], # Use .get para evitar KeyError se uma classe não existir
                           color=['#4CAF50', '#F44336'])

                    # Adicionar percentagens
                    for i, p in enumerate([passed_pct.get('yes', 0), passed_pct.get('no', 0)]):
                        ax.text(i, passed_counts.iloc[i]/2, f'{p:.1f}%',
                                ha='center', va='center', color='white', fontweight='bold')

                    ax.set_ylabel('Número de Estudantes')
                    ax.set_title('Distribuição de Aprovados e Reprovados')
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("Coluna 'passed' não encontrada para mostrar a distribuição do alvo.")


            with col2:
                # Seletor de variável numérica para histograma
                num_cols = data.select_dtypes(include=np.number).columns.tolist()
                # Excluir passed_mapped se existir
                num_cols = [col for col in num_cols if col != 'passed_mapped']

                if num_cols:
                     selected_num = st.selectbox(
                         "Selecione uma variável numérica para visualizar a distribuição:",
                         options=num_cols
                     )

                     # Criar histograma com KDE para a variável selecionada
                     fig, ax = plt.subplots(figsize=(8, 5))
                     sns.histplot(data=data, x=selected_num, kde=True, ax=ax)

                     # Adicionar média e mediana
                     mean_val = data[selected_num].mean()
                     median_val = data[selected_num].median()

                     ax.axvline(mean_val, color='red', linestyle='--', label=f'Média: {mean_val:.2f}')
                     ax.axvline(median_val, color='green', linestyle=':', label=f'Mediana: {median_val:.2f}')
                     ax.legend()

                     ax.set_title(f'Distribuição de {selected_num}')
                     ax.set_xlabel(selected_num)
                     st.pyplot(fig)
                     plt.close(fig)
                else:
                     st.info("Não há colunas numéricas para mostrar distribuições.")


            # Variáveis categóricas
            st.markdown("### Distribuição de Variáveis Categóricas")

            cat_cols = data.select_dtypes(include='object').columns.tolist()
            # Excluir passed se existir
            cat_cols = [col for col in cat_cols if col != 'passed']

            if cat_cols:
                selected_cat = st.selectbox(
                    "Selecione uma variável categórica para visualizar a distribuição:",
                    options=cat_cols
                )

                fig, ax = plt.subplots(figsize=(10, 6))

                # Contar valores e plotar
                cat_counts = data[selected_cat].value_counts().sort_values(ascending=False)
                cat_pct = data[selected_cat].value_counts(normalize=True).sort_values(ascending=False) * 100

                # Usar cores do Seaborn para barras
                palette = sns.color_palette("viridis", len(cat_counts))
                bars = ax.bar(cat_counts.index, cat_counts.values, color=palette)

                # Adicionar percentagens
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    # Position text above the bar, adjust y position slightly
                    ax.text(bar.get_x() + bar.get_width()/2, height + (ax.get_ylim()[1]*0.01),
                            f'{cat_pct.iloc[i]:.1f}%',
                            ha='center', va='bottom', fontsize=9)

                ax.set_title(f'Distribuição de {selected_cat}')
                ax.set_ylabel('Contagem')
                ax.set_xlabel(selected_cat)

                # Rotacionar labels se houver muitas categorias
                if len(cat_counts) > 5:
                    plt.xticks(rotation=45, ha='right')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                 st.info("Não há colunas categóricas para mostrar distribuições.")


        with tab3:
            st.markdown('<h2 class="sub-header">Análise de Correlações</h2>', unsafe_allow_html=True)

            # Selecionar apenas variáveis numéricas para a matriz de correlação
            num_data = data.select_dtypes(include=np.number)

            if not num_data.empty and num_data.shape[1] > 1:
                 # Calcular correlação
                 corr_matrix = num_data.corr()

                 # Criar mapa de calor com Seaborn
                 fig, ax = plt.subplots(figsize=(12, 10))
                 heatmap = sns.heatmap(
                     corr_matrix,
                     annot=True,
                     cmap='coolwarm', # Use a diverging colormap
                     fmt=".2f",
                     linewidths=0.5,
                     ax=ax,
                     annot_kws={"size": 8} # Adjust font size for annotations
                 )

                 plt.title('Matriz de Correlação entre Variáveis Numéricas', fontsize=15)
                 plt.xticks(rotation=45, ha='right', fontsize=8)
                 plt.yticks(fontsize=8)

                 # Ajustar layout para evitar corte das labels
                 plt.tight_layout()

                 st.pyplot(fig)
                 plt.close(fig)

                 # Top correlações com a variável alvo
                 if 'passed_mapped' in num_data.columns:
                     st.write("### Top 10 Correlações com 'passed_mapped'")

                     # Calcular correlações com o alvo, remover o próprio alvo e ordenar
                     passed_corr = corr_matrix['passed_mapped'].drop('passed_mapped', errors='ignore').sort_values(ascending=False)

                     if not passed_corr.empty:
                         fig, ax = plt.subplots(figsize=(10, 6))
                         # Garantir que só pegamos até 10 se existirem
                         top_corr = passed_corr.head(10)
                         bars = ax.barh(
                             top_corr.index,
                             top_corr.values,
                             color=plt.cm.viridis(np.linspace(0, 1, len(top_corr))) # Dynamic color based on number of bars
                         )

                         # Add values on the bars
                         for i, bar in enumerate(bars):
                              width = bar.get_width()
                              # Determine horizontal alignment based on bar width
                              ha = 'left' if width > 0 else 'right'
                              # Position text slightly outside the bar
                              x_pos = width + (ax.get_xlim()[1] * 0.01) if width > 0 else width - (ax.get_xlim()[1] * 0.01)
                              ax.text(
                                  x_pos,
                                  bar.get_y() + bar.get_height()/2,
                                  f'{top_corr.values[i]:.2f}',
                                  ha=ha,
                                  va='center',
                                  fontsize=9
                              )


                         ax.set_xlabel('Correlação com passed_mapped')
                         ax.set_title('Top Variáveis Correlacionadas com Aprovação')
                         ax.grid(axis='x', linestyle='--', alpha=0.7)
                         ax.set_axisbelow(True)

                         plt.tight_layout()
                         st.pyplot(fig)
                         plt.close(fig)
                     else:
                         st.info("Nenhuma correlação encontrada (apenas uma coluna numérica além do alvo?).")
                 else:
                      st.warning("Coluna 'passed_mapped' não encontrada nas colunas numéricas para calcular correlações com o alvo.")
            else:
                 st.info("Não há colunas numéricas suficientes para calcular e mostrar a matriz de correlação.")


        with tab4:
            st.markdown('<h2 class="sub-header">Visualizações Chave</h2>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                # Box plot para 'absences' por situação final
                st.write("### Faltas por Situação Final")

                if 'passed' in data.columns and 'absences' in data.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    sns.boxplot(
                        x='passed',
                        y='absences',
                        data=data,
                        palette={'yes': '#4CAF50', 'no': '#F44336'},
                        ax=ax
                    )

                    # Adicionar estatísticas por grupo (média e mediana)
                    for i, status in enumerate(['yes', 'no']):
                        subset = data[data['passed'] == status]
                        if not subset.empty:
                            mean_val = subset['absences'].mean()
                            median_val = subset['absences'].median()

                            # Adicionar linha para média e mediana
                            x_pos = 0 if status == 'yes' else 1
                            ax.hlines(mean_val, x_pos-0.3, x_pos+0.3, colors='blue', linestyles='--',
                                    label='Média' if i == 0 else "") # Label only for the first line for legend
                            ax.hlines(median_val, x_pos-0.3, x_pos+0.3, colors='orange', linestyles=':',
                                    label='Mediana' if i == 0 else "") # Label only for the first line

                            # Add text labels slightly above/below lines
                            ax.text(x_pos, mean_val + ax.get_ylim()[1]*0.01, f'Média: {mean_val:.1f}',
                                    ha='center', fontsize=9, color='blue')
                            ax.text(x_pos, median_val - ax.get_ylim()[1]*0.02, f'Mediana: {median_val:.1f}',
                                    ha='center', fontsize=9, color='orange')


                    ax.set_title('Distribuição de Faltas por Situação Final')
                    ax.set_xlabel('Situação Final')
                    ax.set_ylabel('Número de Faltas')
                    ax.set_xticklabels(['Passou', 'Não Passou'])

                    # Add legend if lines were added
                    if 'Média' in ax.get_legend_handles_labels()[1]:
                         ax.legend(loc='upper right')


                    # Adicionar grid e melhorar o visual
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    ax.set_axisbelow(True) # Ensure grid is behind data

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                     st.info("Colunas 'passed' ou 'absences' não encontradas para esta visualização.")

            with col2:
                 # Scatter plot entre duas variáveis numéricas com hue por situação final
                st.write("### Relação entre Variáveis Numéricas")

                num_cols = data.select_dtypes(include=np.number).columns.tolist()
                # Remover a coluna alvo
                num_cols = [col for col in num_cols if col != 'passed_mapped']

                if num_cols and 'passed' in data.columns:
                     # Definir colunas padrão para mostrar a relação entre notas (se existirem)
                     default_x = num_cols.index('G1') if 'G1' in num_cols else (0 if num_cols else None)
                     default_y = num_cols.index('G3') if 'G3' in num_cols else (min(1, len(num_cols)-1) if len(num_cols)>1 else None)

                     if default_x is not None and default_y is not None:
                         x_col = st.selectbox("Variável X:", num_cols, index=default_x)
                         y_col = st.selectbox("Variável Y:", num_cols, index=default_y)

                         fig, ax = plt.subplots(figsize=(10, 6))

                         # Scatter plot com transparência para ver densidade de pontos
                         sns.scatterplot(
                             x=x_col,
                             y=y_col,
                             hue='passed',
                             palette={'yes': '#4CAF50', 'no': '#F44336'},
                             s=80,
                             alpha=0.7,
                             data=data,
                             ax=ax
                         )

                         # Adicionar linha de regressão para cada grupo
                         for status, color in zip(['yes', 'no'], ['#4CAF50', '#F44336']):
                             subset = data[data['passed'] == status]
                             # Verificar se há dados suficientes (pelo menos 2 pontos)
                             if len(subset) > 1 and x_col in subset.columns and y_col in subset.columns:
                                 # Need to handle potential all-NaN values in columns before regplot
                                 if not subset[[x_col, y_col]].isnull().all().any():
                                     sns.regplot(
                                         x=x_col,
                                         y=y_col,
                                         data=subset,
                                         scatter=False, # Don't plot points again
                                         ax=ax,
                                         line_kws={'color': color, 'linestyle': '--'},
                                         ci=None # No confidence interval for cleaner look
                                     )

                         # Adicionar rótulos
                         ax.set_title(f'Scatter Plot: {x_col} vs {y_col} por Situação Final')
                         ax.set_xlabel(x_col)
                         ax.set_ylabel(y_col)

                         # Alterar legenda
                         handles, labels = ax.get_legend_handles_labels()
                         # Ensure labels are correct for 'yes' and 'no'
                         legend_labels = ['Passou', 'Não Passou'] if 'yes' in labels and 'no' in labels else labels
                         ax.legend(handles, legend_labels, title='Situação Final')


                         # Adicionar grid
                         ax.grid(linestyle='--', alpha=0.5)
                         ax.set_axisbelow(True)

                         plt.tight_layout()
                         st.pyplot(fig)
                         plt.close(fig)
                     else:
                         st.info("Não há colunas numéricas suficientes para gerar um scatter plot.")
                else:
                     st.info("Não há colunas numéricas ou a coluna 'passed' não foi encontrada para esta visualização.")


            # Análise de tempo livre vs faltas
            st.write("### Relação entre Nível de Tempo Livre e Média de Faltas")

            if 'freetime' in data.columns and 'absences' in data.columns:
                fig, ax = plt.subplots(figsize=(10, 6))

                # Agrupar dados por tempo livre e calcular média de faltas
                # Use observed=True to avoid warnings with categorical data in recent pandas
                freetime_groups = data.groupby('freetime', observed=True)['absences'].agg(['mean', 'median', 'count']).reset_index()

                if not freetime_groups.empty:
                    # Plotar linha para média e barras para contagem
                    ax1 = ax
                    line = ax1.plot(
                        freetime_groups['freetime'],
                        freetime_groups['mean'],
                        'o-',
                        color='#3366FF',
                        linewidth=3,
                        markersize=10,
                        label='Média de Faltas'
                    )

                    ax1.set_xlabel('Nível de Tempo Livre (1: muito baixo, 5: muito alto)')
                    ax1.set_ylabel('Média de Faltas', color='#3366FF')
                    ax1.tick_params(axis='y', labelcolor='#3366FF')
                    ax1.set_xticks(freetime_groups['freetime']) # Ensure all unique freetime values are ticks

                    # Adicionar valores nas linhas
                    for x, y in zip(freetime_groups['freetime'], freetime_groups['mean']):
                        # Add a small vertical offset to the text
                        ax1.annotate(f'{y:.1f}', (x, y), xytext=(0, 10),
                                   textcoords='offset points', ha='center', fontsize=9)

                    # Criar segundo eixo para contagem de estudantes
                    ax2 = ax1.twinx()
                    bars = ax2.bar(
                        freetime_groups['freetime'],
                        freetime_groups['count'],
                        alpha=0.5, # More transparency
                        color='#32CD32',
                        width=0.6, # Wider bars
                        label='Número de Estudantes'
                    )

                    # Adicionar contagem nas barras
                    for bar in bars:
                        height = bar.get_height()
                        # Add a small vertical offset to the text
                        ax2.text(
                            bar.get_x() + bar.get_width()/2,
                            height,
                            f'{int(height)}',
                            ha='center',
                            va='bottom',
                            fontsize=9
                        )

                    ax2.set_ylabel('Número de Estudantes', color='#32CD32')
                    ax2.tick_params(axis='y', labelcolor='#32CD32')

                    # Combinar legendas dos dois eixos
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

                    ax1.set_title('Relação entre Nível de Tempo Livre e Faltas')
                    ax1.grid(axis='y', linestyle='--', alpha=0.7) # Grid only on y-axis

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.markdown("""
                    **Conclusões sobre 'freetime' vs 'absences' (com base na tua análise):**
                    1.  **Menos tempo livre associado a mais faltas:** Alunos com nível 1 de tempo livre têm, em média, mais faltas.
                    2.  **Relação não linear:** A partir do nível 3, o número de faltas estabiliza, com leve aumento nos níveis 4 e 5.
                    3.  **Distribuição de alunos:** A maioria dos alunos reporta níveis médios de tempo livre (3-4).
                    """)
                else:
                    st.info("Dados insuficientes nos grupos de 'freetime' para gerar este gráfico.")
            else:
                 st.info("Colunas 'freetime' ou 'absences' não encontradas para esta visualização.")


elif page == "🔍 Pré-processamento":
    st.markdown('<h1 class="main-header">Processamento de Dados</h1>', unsafe_allow_html=True)
    st.write("As etapas de pré-processamento foram aplicadas para preparar os dados para os modelos de Machine Learning.")

    st.markdown('<h2 class="sub-header">Pipeline de Pré-processamento</h2>', unsafe_allow_html=True)

    # Use HTML and CSS for a simple flow diagram
    st.markdown("""
    <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">

        <div style="text-align:center; padding: 1rem; background-color: #e0f7fa; border-radius: 5px; height: 150px; display: flex; flex-direction: column; justify-content: center; width: 200px;">
            <h4>1. Dados Brutos</h4>
            <p style="font-size: 0.9rem;">• Dataset Original</p>
            <p style="font-size: 0.9rem;">• Valores Categóricos/Numéricos</p>
            <p style="font-size: 0.9rem;">• Variável Target ('passed')</p>
        </div>

        <div style="font-size: 2rem; margin: 0 10px;">&rarr;</div>

        <div style="text-align:center; padding: 1rem; background-color: #fff3e0; border-radius: 5px; height: 150px; display: flex; flex-direction: column; justify-content: center; width: 200px;">
            <h4>2. Target Mapping</h4>
            <p style="font-size: 0.9rem;">• 'yes' &rarr; 1</p>
            <p style="font-size: 0.9rem;">• 'no' &rarr; 0</p>
            <p style="font-size: 0.9rem;">• Nova coluna 'passed_mapped'</p>
        </div>

         <div style="font-size: 2rem; margin: 0 10px;">&rarr;</div>

        <div style="text-align:center; padding: 1rem; background-color: #e8f5e9; border-radius: 5px; height: 150px; display: flex; flex-direction: column; justify-content: center; width: 200px;">
            <h4>3. Split Treino/Teste</h4>
            <p style="font-size: 0.9rem;">• Dados divididos ({:.0f}% Teste)</p>
            <p style="font-size: 0.9rem;">• Separação X (features) e y (target)</p>
            <p style="font-size: 0.9rem;">• Estratificado (mantém proporção do target)</p>
        </div>

        <div style="font-size: 2rem; margin: 10px 0;">&darr;</div> <!-- Vertical arrow below the last box -->

    </div>

     <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; margin-top: 0px;"> /* New row, centered */

        <div style="font-size: 2rem; margin: 0 10px;">&rarr;</div>

        <div style="text-align:center; padding: 1rem; background-color: #e3f2fd; border-radius: 5px; height: 150px; display: flex; flex-direction: column; justify-content: center; width: 200px;">
            <h4>5. Dados Processados</h4>
            <p style="font-size: 0.9rem;">• Numéricos Escalados</p>
            <p style="font-size: 0.9rem;">• Categóricos One-Hot Encoded</p>
            <p style="font-size: 0.9rem;">• Prontos para Modelagem</p>
        </div>

         <div style="font-size: 2rem; margin: 0 10px;">&larr;</div> /* Arrow back */


        <div style="text-align:center; padding: 1rem; background-color: #fce4ec; border-radius: 5px; height: 150px; display: flex; flex-direction: column; justify-content: center; width: 200px;">
            <h4>4. ColumnTransformer</h4>
            <p style="font-size: 0.9rem;">• **Numéricos:** MinMaxScaler</p>
            <p style="font-size: 0.9rem;">• **Categóricos:** OneHotEncoder</p>
            <p style="font-size: 0.9rem;">• Fit no Treino, Transform no Teste/Novos Dados</p>
        </div>


    </div>

    """.format(TEST_SIZE*100), unsafe_allow_html=True)


    st.subheader("Etapas Detalhadas")
    st.markdown("""
    1.  **Carregamento e Target Mapping:** O ficheiro CSV é carregado. A coluna 'passed' (string: 'yes'/'no') é convertida para uma coluna numérica ('passed_mapped': 1/0) que os modelos podem utilizar.
    2.  **Separação de Features e Target:** As colunas de input (features, X) são separadas da coluna alvo (target, y). As colunas 'passed' e 'passed_mapped' são removidas das features.
    3.  **Divisão Treino/Teste:** O dataset é dividido aleatoriamente em conjuntos de treino e teste, com {:.0f}% dos dados reservados para teste (valor definido no código). A divisão é estratificada para garantir que a distribuição da variável alvo fosse semelhante em ambos os conjuntos.
    4.  **ColumnTransformer:** É configurado um `ColumnTransformer` para aplicar transformações específicas a colunas de diferentes tipos:
        *   Colunas numéricas (e.g., idade, faltas) são escalonadas usando `MinMaxScaler`. Isto coloca os valores numa escala entre 0 e 1, o que é importante para modelos sensíveis à magnitude dos features (como SVM e KNN).
        *   Colunas categóricas (e.g., escola, sexo, emprego dos pais) são convertidas em representações numéricas binárias usando `OneHotEncoder`. Para colunas com apenas duas categorias (binárias), `drop='if_binary'` remove uma das colunas resultantes para evitar multicolinearidade.
    5.  **Aplicação das Transformações:** O `ColumnTransformer` é primeiro ajustado (fit) apenas nos dados de treino para aprender os parâmetros de escalonamento (min/max) e as categorias únicas para o one-hot encoding. Depois, é usado para transformar (transform) tanto os dados de treino quanto os dados de teste. **É crucial usar apenas `transform` nos dados de teste para evitar data leakage.**
    """.format(TEST_SIZE*100))

    st.subheader("Exemplo de Dados Processados")
    st.write("Após o pré-processamento, os dados de input são representados por um array numérico de alta dimensionalidade.")

    # Tentar carregar um snippet dos dados processados se existirem os ficheiros
    try:
        # Assumindo que guardaste os processed dataframes em CSV
        # ATENÇÃO: Verifica o nome correto do ficheiro que guardaste no teu notebook
        processed_train_path = 'data/processed/train_processed.csv' # OU 'data/processed/train_student_data_processed_final_v2.csv' se usaste esse nome
        if os.path.exists(processed_train_path):
             train_df_processed_example = pd.read_csv(processed_train_path)
             st.dataframe(train_df_processed_example.head(), use_container_width=True)
             st.write(f"Shape dos dados de treino processados: {train_df_processed_example.shape}")
             st.write(f"O número de colunas aumentou de {len(original_input_columns)} para {train_df_processed_example.shape[1]-1} devido ao One-Hot Encoding e exclusão do target.") # -1 para excluir a coluna alvo
        else:
             st.warning(f"Ficheiro CSV com dados de treino processados não encontrado ('{processed_train_path}'). Não é possível mostrar um exemplo.")

    except Exception as e:
        st.error(f"Erro ao carregar dados de treino processados para exemplo: {e}")


elif page == "🧠 Modelos e Avaliação":
    st.markdown('<h1 class="main-header">Modelagem e Avaliação</h1>', unsafe_allow_html=True)
    st.write("Nesta secção, explorámos diferentes algoritmos de classificação para prever o sucesso dos estudantes.")

    st.subheader("Modelos Experimentados")
    st.markdown("""
    Foram avaliados vários modelos comuns para tarefas de classificação:
    *   Regressão Logística
    *   K-Nearest Neighbors (KNN)
    *   Support Vector Machine (SVM)
    *   Árvore de Decisão
    *   Random Forest
    *   Gradient Boosting
    *   AdaBoost
    *   *(Menciona XGBoost ou outros se usaste)*
    """)

    st.subheader("Estratégia de Avaliação")
    st.markdown("""
    1.  **Baseline:** Calcular a performance de um modelo simples que prevê sempre a classe maioritária para ter uma referência.
    2.  **Validação Cruzada (CV):** Avaliar cada modelo com parâmetros padrão usando `StratifiedKFold` no conjunto de treino. Isto fornece uma estimativa mais robusta da performance geral do modelo, reduzindo a dependência de uma única divisão treino/teste. A estratificação é importante devido ao desequilíbrio de classes.
    3.  **Otimização de Hiperparâmetros:** Para os modelos mais promissores identificados na CV, foi utilizada a técnica `GridSearchCV` para encontrar a melhor combinação de hiperparâmetros que maximiza uma métrica de avaliação relevante (e.g., F1-score, AUC ROC), novamente usando CV no conjunto de treino.
    4.  **Avaliação Final no Conjunto de Teste:** O melhor modelo (ou modelos) otimizado(s) foi(ram) avaliado(s) no conjunto de teste (dados nunca vistos durante o treino ou otimização) para uma estimativa final e imparcial da performance.
    """)

    st.subheader("Resultados de Avaliação")

    # --- Nota Importante ---
    st.warning("""
    Para mostrar os resultados concretos aqui (métricas, matriz de confusão, curvas ROC),
    precisas de ter guardado estes resultados (ou os modelos treinados e os dados de teste)
    quando executaste o teu notebook de modelagem.

    A secção abaixo mostra como **apresentar** estes resultados assumindo que os tens disponíveis.
    Por favor, adapta o código para carregar ou recalcular as métricas/gráficos com base nos teus artefactos salvos.
    """)
    # --- Fim Nota Importante ---

    # Exemplo de como mostrar métricas (precisa de carregar os resultados reais)
    st.markdown("### Métricas Chave no Conjunto de Teste")
    # Cria um dataframe dummy para demonstração se não houver artefactos
    if model is not None:
        # Idealmente, carregarias um dataframe com as métricas finais de um ficheiro
        # Ex: final_metrics_df = pd.read_csv('artefacts/final_metrics.csv')
        # st.dataframe(final_metrics_df, use_container_width=True)
        st.info("Adapta esta secção para carregar e mostrar o dataframe com as métricas de avaliação final dos modelos.")

        # Exemplo de como mostrar métricas apenas do modelo carregado
        st.markdown(f"#### Métricas para o modelo carregado ({model.__class__.__name__}) no conjunto de teste:")
        try:
             # Recalcular métricas para o modelo carregado nos dados de teste processados
             test_df_processed_for_metrics = pd.read_csv('data/processed/test_processed.csv') # Ajusta o nome
             y_test_app_metrics = test_df_processed_for_metrics['passed_mapped']
             X_test_app_metrics_processed = test_df_processed_for_metrics.drop(columns=['passed_mapped'])
             X_test_app_metrics_processed = X_test_app_metrics_processed[processed_feature_names] # Re-order

             y_pred_test_app_metrics = model.predict(X_test_app_metrics_processed)

             metrics = {
                'Acurácia': accuracy_score(y_test_app_metrics, y_pred_test_app_metrics),
                'Precisão': precision_score(y_test_app_metrics, y_pred_test_app_metrics, zero_division=0),
                'Recall': recall_score(y_test_app_metrics, y_pred_test_app_metrics, zero_division=0),
                'F1-Score': f1_score(y_test_app_metrics, y_pred_test_app_metrics, zero_division=0)
             }
             if hasattr(model, 'predict_proba'):
                  y_proba_test_app_metrics = model.predict_proba(X_test_app_metrics_processed)[:, 1]
                  metrics['AUC ROC'] = roc_auc_score(y_test_app_metrics, y_proba_test_app_metrics)

             st.dataframe(pd.DataFrame([metrics]).round(4).T, use_container_width=True)

        except FileNotFoundError:
             st.warning("Ficheiro CSV com dados de teste processados não encontrado ('data/processed/test_processed.csv'). Não é possível calcular métricas.")
        except Exception as e:
             st.error(f"Erro ao calcular métricas para o modelo carregado: {e}")


    st.markdown("### Matriz de Confusão do Melhor Modelo")
    st.write(f"A matriz de confusão para o modelo **{model.__class__.__name__ if model else 'N/A'}** no conjunto de teste:")

    if model is not None and data is not None and preprocessor is not None and processed_feature_names is not None:
         try:
              # Recalcular a matriz de confusão para o modelo carregado nos dados de teste processados
              test_df_processed_for_cm = pd.read_csv('data/processed/test_processed.csv') # Ajusta o nome do teu ficheiro
              y_test_app_cm = test_df_processed_for_cm['passed_mapped']
              X_test_app_cm_processed = test_df_processed_for_cm.drop(columns=['passed_mapped'])
              X_test_app_cm_processed = X_test_app_cm_processed[processed_feature_names] # Re-order columns

              y_pred_test_app_cm = model.predict(X_test_app_cm_processed)

              cm_app_cm = confusion_matrix(y_test_app_cm, y_pred_test_app_cm)
              disp_app_cm = ConfusionMatrixDisplay(confusion_matrix=cm_app_cm, display_labels=['Não Passou (0)', 'Passou (1)'])
              fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
              disp_app_cm.plot(ax=ax_cm, cmap='Blues', values_format='d')
              ax_cm.set_title('Matriz de Confusão (Teste)')
              st.pyplot(fig_cm)
              plt.close(fig_cm)

         except FileNotFoundError:
              st.warning("Ficheiro CSV com dados de teste processados não encontrado ('data/processed/test_processed.csv'). Não é possível mostrar a matriz de confusão.")
         except KeyError:
              st.warning("Coluna 'passed_mapped' não encontrada no ficheiro de teste processado.")
         except Exception as e:
              st.error(f"Erro ao calcular/mostrar matriz de confusão: {e}")
    else:
         st.warning("Modelo, pré-processador ou dados de teste processados não foram carregados corretamente. Matriz de confusão indisponível.")


    st.markdown("### Curva ROC (Receiver Operating Characteristic)")
    st.write("A curva ROC mostra o tradeoff entre a taxa de verdadeiros positivos (TPR) e a taxa de falsos positivos (FPR) para diferentes thresholds de classificação.")

    if model is not None and data is not None and preprocessor is not None and processed_feature_names is not None:
         try:
              # Recalcular ou carregar dados de teste processados e y_test_num para plotar a curva ROC
              test_df_processed_for_roc = pd.read_csv('data/processed/test_processed.csv') # Ajusta o nome do teu ficheiro
              y_test_app_roc = test_df_processed_for_roc['passed_mapped']
              X_test_app_roc_processed = test_df_processed_for_roc.drop(columns=['passed_mapped'])
              X_test_app_roc_processed = X_test_app_roc_processed[processed_feature_names] # Re-order columns

              # Precisamos das probabilidades para a curva ROC
              if hasattr(model, "predict_proba"):
                   y_proba_test_app_roc = model.predict_proba(X_test_app_roc_processed)[:, 1] # Probabilidade da classe positiva (1)

                   fpr, tpr, _ = roc_curve(y_test_app_roc, y_proba_test_app_roc)
                   roc_auc = auc(fpr, tpr)

                   fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
                   ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model.__class__.__name__} (AUC = {roc_auc:.2f})')
                   ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatório (AUC = 0.50)')
                   ax_roc.set_xlabel('Taxa de Falsos Positivos (FPR)')
                   ax_roc.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
                   ax_roc.set_title('Curva ROC (Teste)')
                   ax_roc.legend(loc="lower right")
                   ax_roc.grid(True)
                   st.pyplot(fig_roc)
                   plt.close(fig_roc)

              else:
                   st.warning(f"O modelo {model.__class__.__name__} não suporta `predict_proba` (necessário para a curva ROC).")

         except FileNotFoundError:
              st.warning("Ficheiro CSV com dados de teste processados não encontrado ('data/processed/test_processed.csv'). Não é possível mostrar a curva ROC.")
         except KeyError:
              st.warning("Coluna 'passed_mapped' não encontrada no ficheiro de teste processado.")
         except Exception as e:
              st.error(f"Erro ao calcular/mostrar curva ROC: {e}")
    else:
         st.warning("Modelo, pré-processador ou dados de teste processados não foram carregados corretamente. Curva ROC indisponível.")


    st.markdown("### Importância das Features")
    st.write("Análise de quais características foram mais relevantes para a decisão do modelo (se aplicável ao modelo carregado).")

    if model is not None and processed_feature_names is not None:
         try:
              # Verificar se o modelo tem feature_importances_ (para modelos baseados em árvore)
              if hasattr(model, 'feature_importances_'):
                   importances = model.feature_importances_
                   feature_imp_df = pd.DataFrame({'Feature': processed_feature_names, 'Importance': importances})
                   feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False).head(20) # Mostrar top 20

                   st.write("Top 20 Features Mais Importantes:")
                   fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
                   sns.barplot(x='Importance', y='Feature', data=feature_imp_df, ax=ax_fi, palette='viridis')
                   ax_fi.set_title(f'Importância das Features ({model.__class__.__name__})')
                   ax_fi.set_xlabel('Importância')
                   ax_fi.set_ylabel('Feature Processada') # Label shows processed names
                   plt.tight_layout()
                   st.pyplot(fig_fi)
                   plt.close(fig_fi)

              # Verificar se o modelo tem coef_ (para modelos lineares)
              elif hasattr(model, 'coef_'):
                   # Need to handle multi-class coef_ if applicable, but yours is binary
                   coefs = model.coef_[0] # Assume classification binária, pega os coeficientes para a classe 1
                   feature_coef_df = pd.DataFrame({'Feature': processed_feature_names, 'Coefficient': coefs})
                   # Sort by absolute value to see features with strong impact in either direction
                   feature_coef_df = feature_coef_df.reindex(feature_coef_df['Coefficient'].abs().sort_values(ascending=False).index).head(20) # Sort by absolute value, show top 20

                   st.write("Top 20 Features Mais Relevantes (Coeficientes Absolutos):")
                   fig_fc, ax_fc = plt.subplots(figsize=(10, 8))
                   sns.barplot(x='Coefficient', y='Feature', data=feature_coef_df, ax=ax_fc, palette='coolwarm') # Use diverging palette
                   ax_fc.set_title(f'Coeficientes das Features ({model.__class__.__name__})')
                   ax_fc.set_xlabel('Coeficiente')
                   ax_fc.set_ylabel('Feature Processada') # Label shows processed names
                   plt.tight_layout()
                   st.pyplot(fig_fc)
                   plt.close(fig_fc)

              else:
                   st.info(f"O modelo {model.__class__.__name__} não fornece importância ou coeficientes de feature de forma padrão.")

         except Exception as e:
              st.error(f"Erro ao mostrar importância das features: {e}")
    else:
         st.warning("Modelo ou nomes das features processadas não carregados corretamente. Importância das Features indisponível.")


    st.markdown("### Interpretação e Conclusões")
    st.markdown("""
    Com base nas métricas e visualizações de avaliação:
    *   **Melhor Modelo:** *(Adapta esta parte para nomear o modelo que teve melhor performance no teu teste, por exemplo: "O modelo Random Forest Otimizado obteve o melhor F1-score e AUC ROC...")*
    *   **Performance Geral:** *(Comenta se a performance é boa o suficiente para o problema. Compara com a baseline. Ex: "O modelo final demonstra uma melhoria significativa sobre a baseline, indicando que as features e o algoritmo capturam padrões relevantes.")*
    *   **Análise da Matriz de Confusão:** *(Discute os erros. Olhando para a Matriz de Confusão acima, identifica quantos VP, VN, FP, FN o modelo teve no teste. Ex: "Observando a matriz de confusão, o modelo identificou corretamente X alunos que Passaram (Verdadeiros Positivos) e Y alunos que Não Passaram (Verdadeiros Negativos). No entanto, classificou incorretamente Z alunos que Não Passaram como Passaram (Falsos Positivos) e W alunos que Passaram como Não Passaram (Falsos Negativos).")*
        *   **Falsos Positivos (Não Passou classificado como Passou):** Este é um erro crítico, pois leva a não intervir num aluno que precisaria. *Comenta a taxa de FPs.*
        *   **Falsos Negativos (Passou classificado como Não Passou):** Este erro leva a uma intervenção desnecessária, o que é menos crítico, mas ineficiente. *Comenta a taxa de FNs.*
        *   **Trade-off:** Dependendo do custo de cada tipo de erro, pode ser necessário ajustar o threshold de previsão para minimizar o erro mais custoso.
    *   **Importância das Features:** *(Se mostraste as features mais importantes, discute-as. Ex: "As features mais importantes para o modelo incluem [Lista features]. Isto alinha-se com a intuição de que [explica porquê faz sentido].")*
    *   **Limitações e Próximos Passos:** *(Menciona desafios e o que faria a seguir. Ex: "O desequilíbrio de classes pode ter afetado alguns modelos. Técnicas de reamostragem como SMOTE poderiam ser exploradas. Coletar mais dados, especialmente de alunos em risco, seria benéfico. Explorar a explicabilidade do modelo com ferramentas como SHAP ou LIME seria importante para entender melhor as decisões e ganhar confiança.")*
    """)


elif page == "🔮 Previsão":
    st.markdown('<h1 class="main-header">Previsão para um Novo Estudante</h1>', unsafe_allow_html=True)
    st.write("Insere as características de um estudante para obter uma previsão sobre o seu sucesso académico (Passar/Não Passar).")

    if model is None or preprocessor is None or data is None or not original_input_columns:
         st.warning("Não foi possível carregar o modelo, pré-processador ou dados originais. A secção de previsão está indisponível.")
         if data is not None and not original_input_columns:
              st.warning("A lista de colunas de input originais está vazia. Verifica o carregamento dos dados.")
         # No need to st.stop() here, just show the warning and the section below won't execute the prediction part
    else: # Only proceed if all necessary components are loaded
        st.subheader("Introduzir Dados do Estudante")
        st.write("Por favor, preenche os campos abaixo com os dados do estudante:")

        # Obter as estatísticas descritivas para definir ranges/valores padrão nos inputs numéricos
        # Ensure num_stats is calculated only if data is available and has numerical columns
        num_cols_for_stats = data[original_input_columns].select_dtypes(include=np.number).columns.tolist()
        num_stats = data[num_cols_for_stats].describe().T if num_cols_for_stats else pd.DataFrame() # Empty dataframe if no numeric cols

        # Dicionário para armazenar os inputs do utilizador
        user_inputs = {}

        # Exibir inputs em colunas para um layout mais compacto
        num_cols_per_row = 3
        cols = st.columns(num_cols_per_row)
        col_idx = 0

        # Organizar inputs por tipo (numérico vs categórico) para melhor apresentação
        numeric_input_cols = [col for col in original_input_columns if data[col].dtype in [np.number, 'int64', 'float64']] # Use numpy dtypes check
        nominal_input_cols = [col for col in original_input_columns if data[col].dtype == 'object']

        # Criar inputs para colunas nominais (Selectbox com opções únicas)
        st.markdown("#### Características Categóricas")
        cols = st.columns(num_cols_per_row)
        col_idx = 0
        for col in nominal_input_cols:
            options = data[col].unique().tolist()
            # Handle potential NaN in options if they exist in original data
            options = [opt for opt in options if pd.notna(opt)]
            with cols[col_idx]:
                 user_inputs[col] = st.selectbox(f"{col}", options)
            col_idx = (col_idx + 1) % num_cols_per_row

        # Quebra de linha para nova secção visualmente (ensure column layout is respected)
        if col_idx != 0:
             # Fill the remaining columns with empty space if the last row is not full
             for i in range(num_cols_per_row - col_idx):
                  with cols[col_idx + i]:
                       st.write("") # Add empty space to fill the row
        st.markdown("---") # Horizontal rule

        # Criar inputs para colunas numéricas (Number Input ou Slider com min/max do dataset)
        st.markdown("#### Características Numéricas")
        cols = st.columns(num_cols_per_row)
        col_idx = 0
        for col in numeric_input_cols:
             if col in num_stats.index: # Check if stats are available for this column
                 min_val = float(num_stats.loc[col, 'min'])
                 max_val = float(num_stats.loc[col, 'max'])
                 mean_val = float(num_stats.loc[col, 'mean']) # Usar média como valor default ou um valor razoável

                 with cols[col_idx]:
                     # Use number_input or slider depending on the scale/type (int/float)
                     # Adjust step based on data type
                     step_val = 1 if data[col].dtype == np.int64 else 0.01 # Assuming floats exist
                     user_inputs[col] = st.number_input(f"{col}",
                                                       min_value=min_val,
                                                       max_value=max_val,
                                                       value=mean_val, # Default to mean
                                                       step=step_val,
                                                       format="%f" if step_val != 1 else "%d" # Format based on step
                                                       )
                 col_idx = (col_idx + 1) % num_cols_per_row
             else:
                 # Handle case where column is in original_input_columns but not in num_stats (shouldn't happen if logic is correct)
                 st.warning(f"Estatísticas não encontradas para a coluna numérica '{col}'. Ignorando input para esta feature.")


        # Ensure all inputs are collected even if layout uses columns
        # This dictionary user_inputs should now contain all values


        st.markdown("---") # Horizontal rule before the button

        # Botão para prever (centralizado)
        predict_button_col = st.columns(3)[1] # Create 3 columns and use the middle one
        with predict_button_col:
            if st.button("✨ Fazer Previsão ✨", use_container_width=True):
                # Preparar os dados de input para o modelo
                # Criar um DataFrame com uma única linha a partir dos inputs do utilizador
                input_df = pd.DataFrame([user_inputs])

                # IMPORTANTE: Assegurar que a ordem das colunas no input_df é a mesma
                # que a ordem esperada pelo preprocessor treinado.
                # A lista `processed_feature_names` guarda a ordem APÓS o pre-processamento,
                # mas o `preprocessor.transform` espera a ordem das colunas ORIGINAIS de input (X).
                # A lista `original_input_columns` guarda os nomes das colunas X originais.
                # Usar `original_input_columns` para reordenar o input_df
                try:
                     input_df = input_df[original_input_columns] # Re-order columns to match original X
                     # Ensure dtypes match original data if possible, though preprocessor should handle
                     # For categorical, objects are fine. For numeric, ensure they are numeric.
                     for col in input_df.columns:
                          if col in data.columns:
                               input_df[col] = input_df[col].astype(data[col].dtype)

                except KeyError as e:
                     st.error(f"Erro: A coluna de input '{e}' não foi encontrada no DataFrame criado. Verifica se os nomes das colunas nos inputs correspondem aos dados originais.")
                     st.stop()


                # Aplicar o preprocessor treinado
                try:
                    input_processed = preprocessor.transform(input_df)
                    # input_processed é agora um numpy array.
                    # Se o teu modelo foi treinado com um DataFrame (por exemplo, se usaste pipelines com pandas-friendly transformers),
                    # talvez precises de converter input_processed de volta para DataFrame AQUI,
                    # usando `processed_feature_names` para os nomes das colunas.
                    # Ex: input_processed_df = pd.DataFrame(input_processed, columns=processed_feature_names)
                    # Mas para modelos sklearn padrão (RF, LR, SVM), um array numpy geralmente funciona.
                    # Vamos assumir que o modelo treinado (`model`) aceita o output numpy do `preprocessor.transform`.

                except Exception as e:
                     st.error(f"Erro ao pré-processar o input: {e}")
                     st.warning("Verifica se todos os inputs foram preenchidos corretamente e se correspondem aos tipos esperados.")
                     st.stop()


                # Fazer a previsão com o modelo treinado
                try:
                     prediction = model.predict(input_processed)
                     prediction_proba = model.predict_proba(input_processed)

                except Exception as e:
                     st.error(f"Erro ao fazer a previsão com o modelo: {e}")
                     st.warning("Verifica se o modelo foi carregado corretamente e se o input processado tem o formato esperado.")
                     st.stop()


                # Interpretar e mostrar o resultado
                st.subheader("Resultado da Previsão")

                predicted_class = prediction[0] # 0 ou 1
                # prediction_proba é um array [[prob_classe_0, prob_classe_1]]
                probability = prediction_proba[0][predicted_class] # Probabilidade da classe prevista

                if predicted_class == 1:
                    st.markdown(f'<div class="prediction-card prediction-pass">Previsão: PASSOU</div>', unsafe_allow_html=True)
                    st.info(f"Probabilidade de passar: **{probability:.2f}**")
                else:
                    st.markdown(f'<div class="prediction-card prediction-fail">Previsão: NÃO PASSOU</div>', unsafe_allow_html=True)
                    st.info(f"Probabilidade de não passar: **{probability:.2f}**")

                st.write("---")
                st.write("Nota: Esta previsão baseia-se no modelo treinado com os dados fornecidos e deve ser interpretada com cuidado.")


# --- Footer (Opcional) ---
st.markdown("---")
st.markdown("Aplicação desenvolvida com Streamlit")

# Adicione plt.close('all') no final para garantir que todas as figuras do matplotlib são fechadas
# Isso é uma boa prática em Streamlit para evitar memory leaks em algumas versões/ambientes
plt.close('all')