import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from sklearn import metrics
from sklearn import ensemble, tree, linear_model, svm
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import dash_table
import joblib

# Importa a nova classe de treinamento de modelo
from model_trainer import LoanModelTrainer

# --- Carregamento de Dados e Treinamento do Modelo ---
# Instancia e executa a classe de treinamento
trainer = LoanModelTrainer(
    "dataset/card.asc", "dataset/account.asc", "dataset/disp.asc", 
    "dataset/client.asc", "dataset/district.asc", "dataset/order.asc", 
    "dataset/loan.asc", "dataset/trans.asc"
)
(trained_models, X_train, X_test, y_train, y_test, sc, X_orig, X_scaled_test, df, df_for_plotting) = trainer.train_and_save_models()


# Prepara as colunas para o dash_table.DataTable
columns_with_types = [{"name": i, "id": i, "type": "numeric" if pd.api.types.is_numeric_dtype(df[i]) else "text"} for i in df.columns]

# --- Configuração do Dashboard ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Previsão de Incumprimento de Empréstimos Bancários"
server = app.server

header = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.Span("💰", className="me-2"),
                    dbc.NavbarBrand("Previsão de Incumprimento de Empréstimos Bancários", class_name="fw-bold text-wrap", style={"color": "black"}),
                ], className="d-flex align-items-center"
            ),
            dbc.Badge("Dashboard", color="primary", className="ms-auto")
        ]
    ),
    color="light",
    class_name="shadow-sm mb-3"
)

# --- 1. Aba ASK (PERGUNTAR) ---
ask_tab = dcc.Markdown(
    """
    ### ❓ **PERGUNTAR** — A Pergunta de Negócio
    Esta seção define o problema central do negócio.

    **Tarefa de Negócio**: Como banco, queremos prever quais solicitantes de empréstimo têm alto risco de **incumprimento** (não conseguir pagar o empréstimo). Ao identificar clientes "bons" versus "ruins", podemos melhorar nosso processo de aprovação de empréstimos, gerenciar o risco de forma mais eficaz e oferecer suporte proativo para evitar inadimplências.

    **Partes Interessadas**: Os principais usuários desta análise são **Gerentes de Banco**, **Analistas de Risco** e equipes de **Atendimento ao Cliente**. Eles precisam de uma maneira clara e acionável de entender quem tem maior probabilidade de incumprimento e por quê.

    **Entregáveis**: O produto final é este painel interativo, que fornece uma visão abrangente de nossa análise, desde a preparação dos dados até o desempenho do modelo e as recomendações finais.
    """, className="p-4"
)

# --- 2. Aba PREPARE (PREPARAR) ---
prepare_tab = html.Div(
    children=[
        html.H4(["📝 ", html.B("PREPARAR"), " — Preparando os Dados"], className="mt-4"),
        html.P("Para construir nosso modelo preditivo, primeiro precisamos limpar e preparar um grande dataset contendo informações sobre nossos clientes, suas contas e empréstimos anteriores."),
        html.H5("Fonte e Preparação dos Dados"),
        html.P(
            ["Estamos trabalhando com um dataset de um banco coletado em 1999, combinando oito tabelas diferentes (cliente, conta, empréstimo, etc.). Mesclamos essas tabelas em um único dataset mestre e criamos novos recursos, como `anos_de_emprestimo` e `saldo_medio_antes_do_emprestimo`. O objetivo desse processo é criar uma única tabela limpa da qual nossos modelos possam aprender."]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Dataset Final Mesclado"),
                            dbc.CardBody(
                                [
                                    html.P(f"Linhas: {df.shape[0]}"),
                                    html.P(f"Recursos: {df.shape[1]}"),
                                ]
                            ),
                        ], className="mb-3"
                    )
                ),
            ]
        ),
        html.H5("Recursos Chave e Seus Significados"),
        dbc.Table.from_dataframe(
            pd.DataFrame({
                "Recurso": ["avg_balance_3M_before_loan", "min_balance_before_loan", "times_balance_below_5K", "age", "years_card_issued", "amount"],
                "Descrição": [
                    "Saldo médio da conta nos 3 meses que antecederam a solicitação do empréstimo.",
                    "O saldo mais baixo registrado na conta antes da aprovação do empréstimo. Um número muito baixo aqui pode ser um sinal de alerta. 🚩",
                    "O número de vezes que o saldo da conta caiu abaixo de 5.000. Um número maior sugere instabilidade financeira.",
                    "A idade do cliente.",
                    "O número de anos em que o cliente teve um cartão de crédito com o banco.",
                    "O valor do empréstimo concedido."
                ]
            }),
            striped=True, bordered=True, hover=True
        ),
        html.H5("Amostra do Dataset (Primeiras 10 Linhas)"),
        dash_table.DataTable(
            id='sample-table',
            columns=columns_with_types,
            data=df.head(10).to_dict('records'),
            sort_action="native",
            filter_action="native",
            page_action="none",
            style_table={'overflowX': 'auto', 'width': '100%'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold',
                'textAlign': 'center',
            },
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'font-size': '12px',
                'minWidth': '80px', 'width': 'auto', 'maxWidth': '150px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
        ),
    ], className="p-4"
)

# --- 3. Aba ANALYZ (ANALISAR) ---
analyze_tab = html.Div(
    children=[
        html.H4(["📈 ", html.B("ANALISAR"), " — Encontrando Padrões e Construindo Modelos"], className="mt-4"),
        html.P(
            ["A aba Analisar é onde transformamos nossos dados preparados em insights acionáveis e avaliamos a eficácia de nossos modelos de aprendizado de máquina. Ela é dividida em duas sub-abas principais: ", html.B("Análise Exploratória de Dados (AED)"), " e ", html.B("Desempenho do Modelo"), "."]
        ),
        dbc.Tabs([
            dbc.Tab(label="Análise Exploratória de Dados", children=[
                html.Div(
                    children=[
                        html.P(
                            ["A seção de AED nos ajuda a entender as principais características de nossos dados antes de começar a modelagem. É como verificar os ingredientes antes de cozinhar."]
                        ),
                        html.H5("Distribuição de Incumprimento", className="mt-4"),
                        html.P(
                            ["O gráfico de pizza abaixo mostra que nossos dados estão ", html.B("desequilibrados"), 
"—uma pequena porcentagem de clientes de fato inadimpliu. Isso é comum em dados bancários e é por isso que uma alta pontuação de precisão por si só pode ser enganosa. Um modelo que prevê que ninguém irá inadimplir ainda seria ~90% preciso, mas seria inútil para identificar clientes em risco. Não estamos apenas olhando para as porcentagens; estamos vendo um problema de negócio crítico: ", html.B("desequilíbrio de classes"), 
". A fatia grande para 'Sem Incumprimento' (status 0) e a fatia minúscula para 'Incumprimento' (status 1) significa que um modelo pode alcançar alta precisão simplesmente prevendo 'Sem Incumprimento' o tempo todo. É por isso que não podemos confiar apenas na precisão e precisamos de métricas mais robustas, que encontraremos na seção 'Desempenho do Modelo'."]
                        ),
                        dcc.Graph(
                            id="status-pie-chart",
                            figure=go.Figure(
                                data=[go.Pie(labels=df_for_plotting["status"].value_counts().keys().tolist(),
                                             values=df_for_plotting["status"].value_counts().values.tolist(),
                                             marker=dict(colors=['#1f77b4', '#ff7f0e'], line=dict(color="white", width=1.3)),
                                             hoverinfo="label+percent", hole=0.5)],
                                layout=go.Layout(title="Distribuição de Incumprimento de Empréstimos (0=Sem Incumprimento, 1=Incumprimento)", height=400, margin=dict(t=50, b=50))
                            )
                        ),
                        html.H5("Taxa de Incumprimento por Faixa Etária", className="mt-4"),
                        html.P(
                            ["Este gráfico de barras empilhadas mostra a porcentagem de inadimplentes e não inadimplentes em diferentes faixas etárias. Ele nos ajuda a ver se certas faixas etárias são mais propensas à incumprimento. A visualização revela que, embora o número total de empréstimos varie por idade, a porcentagem de inadimplências dentro de cada grupo é relativamente semelhante. Ao empilhar as barras para 'Sem Incumprimento' e 'Incumprimento', podemos ver a proporção de cada resultado dentro de cada faixa etária. Estamos procurando por diferenças significativas na taxa de incumprimento entre as faixas etárias. Com base nos dados, a ", html.B("faixa etária de 45-50 anos é a mais propensa à incumprimento"), ", com uma porcentagem ligeiramente maior de inadimplências em comparação com outros grupos de idade."]
                        ),
                        dcc.Graph(
                            id="age-default-plot",
                            figure=go.Figure(
                                data=[go.Bar(
                                    x=df_for_plotting.groupby('age_bin')['status'].value_counts(normalize=True).unstack()[0].index,
                                    y=df_for_plotting.groupby('age_bin')['status'].value_counts(normalize=True).unstack()[0].values,
                                    name='Sem Incumprimento',
                                    marker_color='#1f77b4'
                                ), go.Bar(
                                    x=df_for_plotting.groupby('age_bin')['status'].value_counts(normalize=True).unstack()[1].index,
                                    y=df_for_plotting.groupby('age_bin')['status'].value_counts(normalize=True).unstack()[1].values,
                                    name='Incumprimento',
                                    marker_color='#ff7f0e'
                                )],
                                layout=go.Layout(
                                    barmode='stack',
                                    title="Porcentagem de Incumprimento por Faixa Etária",
                                    yaxis_title="Porcentagem",
                                    xaxis_title="Faixa Etária",
                                    height=450, margin=dict(t=50, b=50)
                                )
                            )
                        ),
                        html.H5("A Importância de Dados Específicos de Transações", className="mt-4"),
                        html.P(
                            ["Nossa análise destaca o valor de focar em ", html.B("dados específicos e granulares"), ". Neste projeto, criamos recursos detalhados a partir de dados brutos de transações, como `avg_balance_before_loan` e `times_balance_below_5K`. Estes são muito mais informativos do que um simples valor total de transação do cliente, pois capturam comportamentos específicos — como saques a descoberto frequentes ou saldos baixos — que são fortes indicadores de estabilidade financeira e da probabilidade de incumprimento. Uma métrica 'total' simples esconderia esses sinais de risco cruciais, dificultando a previsão precisa do risco do cliente."]
                        ),
                    ], className="p-4"
                )
            ]),
            dbc.Tab(label="Desempenho do Modelo", children=[
                html.Div(
                    children=[
                        html.P(
                            ["Esta seção é sobre como avaliar nossos modelos para escolher o melhor para a tarefa. Não estamos apenas procurando uma 'pontuação alta', mas sim um modelo que seja genuinamente bom em detectar clientes de alto risco."]
                        ),
                        html.H5("Métricas de Desempenho do Modelo", className="mt-4"),
                        html.P(
                            ["Para realmente avaliar nossos modelos, nos concentramos em várias métricas-chave além da simples precisão:",
                             html.Ul([
                                 html.Li([html.B("Precisão (Precision):"), " Pense na Precisão como o custo de um alarme falso. Se nosso modelo tem alta precisão, as pessoas que ele sinaliza para acompanhamento são muito provavelmente inadimplentes reais. Dos clientes que previmos que inadimpliriam, quantos realmente o fizeram? Alta precisão é boa para evitar alarmes falsos."]),
                                 html.Li([html.B("Recall:"), " Pense no Recall como o custo de um aviso perdido. Se nosso modelo tem alto recall, ele é muito bom em encontrar a maioria das pessoas que irão inadimplir, para que não percamos um cliente de alto risco. De todos os clientes que inadimpliram, quantos nosso modelo identificou com sucesso? O alto recall é crucial para um banco pegar o máximo de clientes em risco possível."]),
                                 html.Li([html.B("F1-Score:"), " Um equilíbrio entre precisão e recall, fornecendo uma única métrica para comparar modelos. Esta é a média harmônica da precisão e do recall. É um único número que nos ajuda a comparar modelos quando a precisão e o recall são importantes."]),
                                 html.Li([html.B("ROC-AUC:"), " Esta é uma métrica de resumo poderosa. Ela mede a capacidade do modelo de distinguir entre as duas classes (inadimplentes vs. não inadimplentes). Uma pontuação mais próxima de 1.0 é melhor."])
                             ])
                            ]
                        ),
                        html.P([
                            "Os modelos ", html.B("Random Forest"), ", ", html.B("Decision Tree"), ", ",
                            html.B("Gradient Boosting"), " e ", html.B("SVM"), 
                            " demonstram um desempenho perfeito na identificação de inadimplentes. ",
                            "Cada um alcançou ", html.B("100% de Precisão, Recall, F1-Score e Acurácia"),
                            ", juntamente com um ", html.B("AUC de 1.00"), 
                            ". Isso significa que eles classificaram tanto os inadimplentes quanto os não inadimplentes sem um único erro, ",
                            "evitando quaisquer clientes de alto risco perdidos e garantindo resultados de negócio confiáveis. O modelo de ", html.B("Regressão Logística"), " também teve um forte desempenho, com uma ",
                            html.B("Acurácia de 99%"), " e um ", html.B("AUC de 1.00"),
                            ". No entanto, ele perdeu ", html.B("2 inadimplentes reais"), " (", html.B("Recall = 0.91"), 
                            "), o que reduziu seu ", html.B("F1-Score para inadimplentes para 0.95"),
                            ". Isso o torna menos confiável do que os outros modelos que alcançaram uma detecção perfeita."
                        ]),

                        html.H6("Matriz de Confusão", className="mt-4"),
                        html.P(
                            ["A matriz de confusão é uma tabela que divide as previsões do nosso modelo em quatro categorias:", 
                             html.Ul([
                                 html.Li([html.B("Verdadeiros Positivos (VP):"), " Inadimplentes corretamente previstos."]),
                                 html.Li([html.B("Verdadeiros Negativos (VN):"), " Não inadimplentes corretamente previstos."]),
                                 html.Li([html.B("Falsos Positivos (FP):"), " Inadimplentes incorretamente previstos (erro Tipo I). Estes são os 'alarmes falsos'."]),
                                 html.Li([html.B("Falsos Negativos (FN):"), " Não inadimplentes incorretamente previstos (erro Tipo II). Estes são os 'avisos perdidos' que um banco quer evitar a todo custo, pois representam uma perda financeira potencial."])
                             ])
                            ]
                        ),
                        dbc.Row([
                            dbc.Col(
                                html.Div([
                                    html.H6("Selecione um Modelo:"),
                                    dcc.Dropdown(
                                        id="model-dropdown",
                                        options=[{'label': name, 'value': name} for name in trained_models.keys()],
                                        value='Random Forest',
                                        clearable=False,
                                    ),
                                    dcc.Graph(id="confusion-matrix-plot"),
                                ]), md=6
                            ),
                            dbc.Col(
                                html.Div([
                                    html.H6("Relatório de Desempenho do Modelo:"),
                                    html.Pre(id="classification-report-text"),
                                ]), md=6
                            ),
                        ]),
                        html.Hr(),
                        html.H5("Importância dos Recursos", className="mt-4"),
                        html.P(
                            ["Este gráfico de barras nos mostra em quais recursos o modelo selecionado mais se baseou para fazer suas previsões. Estamos vendo o 'processo de pensamento' do modelo. Quanto maior a barra, mais influente foi esse recurso. Neste caso, os dois recursos mais importantes foram ", html.B("`avg_balance_before_loan`"), " e ", html.B("`avg_amount_trans_before_loan`"), ". Este é um insight crítico porque valida o processo de preparação dos dados — nosso trabalho na engenharia de recursos valeu a pena, criando sinais significativos para o modelo."]
                        ),
                        dcc.Graph(id="feature-importance-plot"),
                        html.Hr(),
                        html.H5("Curva Característica de Operação do Receptor (ROC)", className="mt-4"),
                        html.P(id="roc-curve-description"),
                        dcc.Graph(id="roc-curve-plot"),
                    ], className="p-4"
                )
            ])
        ])
    ]
)

# --- 4. Aba ACT (AGIR) ---
act_tab = dcc.Markdown(
    """
    ### 🚀 **AGIR** — Recomendações e Próximos Passos
    Esta é a seção mais importante, pois traduz os insights dos dados em uma estratégia de negócio.

    - **Priorizar com Dados**: Os modelos identificaram indicadores de risco chave, como o `min_balance_before_loan` e o `times_balance_below_5K` do cliente. Estes são preditores poderosos de futura incumprimento. Os gerentes de banco devem usar esses insights para criar regras de avaliação de risco mais robustas. Por exemplo, qualquer solicitante cujo saldo caia abaixo de um certo limite várias vezes pode exigir uma análise mais cuidadosa.

    - **Retenção Proativa**: Em vez de esperar que os clientes inadimplam, o banco pode usar o modelo implementado para obter uma lista diária de contas com alto risco. Um representante de atendimento ao cliente pode então entrar em contato proativamente com esses clientes para oferecer aconselhamento financeiro, um pequeno empréstimo de emergência ou um plano de pagamento flexível, reduzindo assim o risco de uma perda.

    - **Implementar o Melhor Modelo**: O modelo **Gradient Boosting** é nossa principal recomendação para implementação devido ao seu desempenho superior em todas as métricas. Este modelo será o cérebro por trás de nossa nova estratégia proativa de risco de empréstimos, ajudando o banco a tomar decisões mais inteligentes e baseadas em dados.
    """, className="p-4"
)

app.layout = dbc.Container(
    [
        header,
        dbc.Tabs(
            [
                dbc.Tab(ask_tab, label="Perguntar"),
                dbc.Tab(prepare_tab, label="Preparar"),
                dbc.Tab(analyze_tab, label="Analisar"),
                dbc.Tab(act_tab, label="Agir"),
            ]
        ),
    ],
    fluid=True,
)

# --- Callbacks ---
@app.callback(
    Output("confusion-matrix-plot", "figure"),
    Output("classification-report-text", "children"),
    Output("feature-importance-plot", "figure"),
    Output("roc-curve-plot", "figure"),
    Output("roc-curve-description", "children"),
    Input('model-dropdown', 'value')
)
def update_metrics_and_importance(selected_model):
    model = trained_models[selected_model]
    
    # Verifica se o modelo precisa de dados escalados ou não
    if selected_model in ['SVM', 'Logistic Regression']:
        X_test_for_pred = X_scaled_test
    else:
        X_test_for_pred = X_test

    y_pred = model.predict(X_test_for_pred)
    
    # 1. Gráfico da Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)

    # Reordena para o layout VP-FP / FN-VN
    z_data = np.array([
        [cm[1, 1], cm[0, 1]],  # VP, FP
        [cm[1, 0], cm[0, 0]]  # FN, VN
    ])

    cm_text = np.array([
        [f'VP: {cm[1, 1]}', f'FP: {cm[0, 1]}'],
        [f'FN: {cm[1, 0]}', f'VN: {cm[0, 0]}']
    ])

    # Inverte as linhas para a exibição correta de cima para baixo
    z_data = np.flipud(z_data)
    cm_text = np.flipud(cm_text)

    fig_cm = ff.create_annotated_heatmap(
        z=z_data,
        x=["Previsto Incumprimento (1)", "Previsto Sem Incumprimento (0)"],
        y=["Incumprimento Real (1)", "Sem Incumprimento Real (0)"],
        annotation_text=cm_text,
        colorscale='Blues'
    )

    fig_cm.update_layout(title=f"Matriz de Confusão ({selected_model})", height=450, margin=dict(t=50, b=50))

    # 2. Relatório de Classificação
    report = classification_report(y_test, y_pred, output_dict=False, zero_division=0)
    
    # 3. Gráfico de Importância dos Recursos
    fig_fi = go.Figure()
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_orig.columns
        df_importance = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
        fig_fi.add_trace(go.Bar(
            x=df_importance['importance'],
            y=df_importance['feature'],
            orientation='h'
        ))
        fig_fi.update_layout(
            title=f"Importância dos Recursos para {selected_model}",
            xaxis_title="Importância",
            yaxis_title="Recurso",
            height=500,
            margin=dict(l=150, t=50, b=50)
        )
    else:
        fig_fi.update_layout(title=f"Importância dos Recursos não Disponível para {selected_model}")
        
    # 4. Novo Gráfico e Descrição da Curva ROC
    fig_roc = go.Figure()
    roc_description_list = []
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_for_pred)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        roc_auc = metrics.auc(fpr, tpr)
        
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{selected_model} (AUC = {roc_auc:.2f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Palpite Aleatório (AUC = 0.5)', line=dict(dash='dash', color='gray')))
        fig_roc.update_layout(
            title="Curva ROC",
            xaxis_title="Taxa de Falso Positivo",
            yaxis_title="Taxa de Verdadeiro Positivo",
            height=450,
            margin=dict(t=50, b=50),
            legend=dict(x=0.6, y=0.1)
        )
        
        # Mescla o texto antigo e o novo
        roc_description_list.extend([
            html.P(["A curva ROC traça a ", html.B("Taxa de Verdadeiro Positivo"), " contra a ", html.B("Taxa de Falso Positivo"), ". Quanto mais próxima a curva estiver do canto superior esquerdo, melhor o modelo é para distinguir entre as duas classes (inadimplentes e não inadimplentes). A Área Sob a Curva (AUC) fornece uma única métrica para resumir o desempenho do modelo. O gráfico que você está vendo mostra o 'trade-off' para cada modelo entre encontrar inadimplentes reais (Taxa de Verdadeiro Positivo) e sinalizar incorretamente não inadimplentes como de alto risco (Taxa de Falso Positivo). Um bom modelo terá uma curva que se inclina para o canto superior esquerdo, ficando bem acima da linha diagonal de 'palpite aleatório', indicando que é muito melhor do que um lançamento de moeda para separar os dois grupos."]),
            html.P(["Todos os modelos, incluindo a ", html.B("Regressão Logística"), ", alcançaram uma ", html.B("curva ROC perfeita com um AUC de 1.00"), ", confirmando sua capacidade de distinguir entre clientes que irão e não irão inadimplir."]),
            html.P(["O modelo atualmente selecionado, ", html.B(selected_model), ", tem um ", html.B("AUC"), " de ", html.B(f"{roc_auc:.2f}.")])
        ])

    else:
        fig_roc.update_layout(title=f"Curva ROC não Disponível para {selected_model}")
        roc_description_list.append(html.P("A curva ROC não está disponível para este modelo, pois ele não suporta previsões de probabilidade."))

    return fig_cm, report, fig_fi, fig_roc, roc_description_list
    
# Executa o aplicativo
if __name__ == "__main__":
    app.run(debug=True)