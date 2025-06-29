import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from numpy import heaviside
from textwrap import dedent
import plotly.graph_objects as go


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME], requests_pathname_prefix='/dash_sli/' )
server = app.server

cabecalho = html.H1("Modelo SIL",className="bg-primary text-white p-2 mb-4")

descricao = dcc.Markdown(
    '''
    É apresentado o modelo para a dinâmica populacional de raiva em raposas na Europa
    [Anderson et al., Nature, vol. 289, pp. 765-771 (1981)](https://www.nature.com/articles/289765a0). Fez-se a suposição de que
    os indivíduos da população podem ser classificados como suscetíveis ($$S$$), infectados
    que ainda não são infecciosos ($$L$$, referindo-se à latência), e infecciosos ($$I$$).
    ''', mathjax=True
)

parametros = dcc.Markdown(
    '''
    * $$a=1 \\text{ ano}^{-1}$$: taxa de natalidade média
    * $$\\mu=0.5 \\text{ ano}^{-1}$$: taxa de mortalidade natural média
    * $$\\alpha=73 \\text{ ano}^{-1}$$: taxa de mortalidade pela raiva (letalidade)
    * $$\\sigma=13 \\text{ ano}^{-1}$$: inverso do período médio de latência
    * $$K=2 \\frac{\\text{ animais}}{\\text{km}^2}$$: capacidade de suporte do meio
    * $$\\beta=77 \\frac{\\text{km}^2}{\\text{ano}}$$: taxa de contatos potencialmente infeccciosos
    ''', mathjax=True
)
cond_inicial = dcc.Markdown(
    '''
    * S: número de indivíduos suscetíveis
    * L: número de indivíduos latentes
    * I: número de indivíduos infectantes


    ''', mathjax=True
)

perguntas = dcc.Markdown(
    '''
    1. Considere uma situação com os parâmetros acima e com densidades iniciais de animais suscetíveis, latentes e infecciosas,
    respectivamente, de 2.5, 0.1 e 0.1 raposas por $$km^2$$. Qual o tempo necessário para se atingir uma condição de equilíbrio?

    2. Mantendo os parâmetros do item 1, reduza o valor da taxa de letalidade pela doença ($$\\alpha$$) para, por exemplo, $$50.0 \\text{ ano}^{-1}$$.
    Teste também outros valores. A condição de equilíbrio é atingida antes ou depois do tempo observado no item anterior? A densidade de suscetíveis
    no equilíbrio é diferente do item 1 ?

    3. Mantenha os parâmetros iniciais e reduza apenas a taxa $$\\sigma$$, que está relacionada à patogenicidade da agente infeccioso. O valor sugerido é
    $$\\sigma=3 \\text{ ano}^{-1}$$. Observe o gráfico para a proporção de animais positivos para a raiva. A proporção de positivos no equilíbrio é diferente da observada no item 1 ? Teste também outros valores de $$\\sigma$$.

    4. Reduza agora a taxa de contatos potencialmente infectantes ($$\\beta$$), mantendo novamente os outros parâmetros como no item 1. O valor sugerido é $$\\beta=60.0 \\text{ km}^{2}\\text{ano}^{-1}$$. O que ocorre com a densidade de suscetíveis no equilíbrio em relação ao observado no item 1 ?
    ''', mathjax=True
)

textos_descricao = html.Div(
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    descricao, title="Descrição do modelo"
                ),
                dbc.AccordionItem(
                    parametros, title="Parâmetros do modelo"
                ),
                dbc.AccordionItem(
                    cond_inicial, title="Condições iniciais"
                ),
                dbc.AccordionItem(
                    perguntas, title="Perguntas"
                ),
            ],
            start_collapsed=True,
        )
    )

ajuste_condicoes_iniciais = html.Div(
        [
            html.P("Ajuste das condições iniciais", className="card-header border-dark mb-3"),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''$$S$$ total de suscetíveis''', mathjax=True), html_for="s_init"),
                    dcc.Slider(id="s_init", min=0.1, max=5, value=2.5, tooltip={"placement": "bottom", "always_visible": False}),
                ],
                className="m-2",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''$$L$$ total de latentes ''', mathjax=True), html_for="i_init"),
                    dcc.Slider(id="l_init", min=0, max=0.2, value=0.1, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''$$I$$ total de infectados ''', mathjax=True), html_for="r_init"),
                    dcc.Slider(id="i_init", min=0.01, max=0.2, value=0.1, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),

        ],
        className="card border-dark mb-3",
    )

ajuste_parametros = html.Div(
        [
            html.P("Ajuste dos parâmetros", className="card-header border-dark mb-3"),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Taxa de natalidade ($$a$$)''', mathjax=True), html_for="alpha"),
                    dcc.Slider(id="a", min=0.5, max=3.0, value=1.0, tooltip={"placement": "bottom", "always_visible": False}),
                ],
                className="m-2",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Taxa de contatos potencialmente infectantes ($$\\beta$$): ''', mathjax=True), html_for="beta"),
                    dcc.Slider(id="beta", min=1, max=150, value=77, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Taxa de mortalidade natural ($$\\mu$$):''', mathjax=True), html_for="gamma"),
                    dcc.Slider(id="mu", min=0.1, max=0.9, value=0.5, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Taxa de mortalidade pela doença ($$\\alpha$$):''', mathjax=True), html_for="delta"),
                    dcc.Slider(id="alpha", min=1, max=150, value=73, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Taxa relacionada à patogenicidade do agente ($$\\sigma$$):''', mathjax=True), html_for="nu"),
                    dcc.Slider(id="sigma", min=1, max=25, value=13, tooltip={"placement": "bottom", "always_visible": False}, className="card-text"),
                ],
                className="m-1",
            ),
            html.Div(
                [
                    dbc.Label(dcc.Markdown('''Capacidade suporte (K):''', mathjax=True), html_for="vacinacao"),
                    dcc.Slider(id='K', min=0.1, max=10, value=2, tooltip={"placement": "bottom", "always_visible": False}, className="card-text" ),
                ],
                className="m-1",
            ),
        ],
        className="card border-dark mb-3",
    )

def ode_sys(state, t, a, beta, mu, alpha, sigma, K):
    s, l, i=state
    ds_dt=(a-mu)*s-((a-mu)/K)*s*(s+i+l)-beta*s*i
    di_dt=beta*s*i-(sigma+mu+((a-mu)/K)*(s+l+i))*l
    dl_dt=sigma*l-(alpha+mu+((a-mu)/K)*(s+l+i))*i
    return [ds_dt, di_dt, dl_dt]

@app.callback(Output('population_chart', 'figure'),
              [Input('s_init', 'value'),
              Input('l_init', 'value'),
              Input('i_init', 'value'),
              Input('a', 'value'),
              Input('beta', 'value'),
              Input('mu', 'value'),
              Input('alpha', 'value'),
              Input('sigma', 'value'),
              Input('K', 'value')])
def gera_grafico(s_init, i_init, l_init, a, beta, mu, alpha, sigma, K):
    t_begin = 0.
    t_end = 70.
    t_nsamples = 10000
    t_eval = np.linspace(t_begin, t_end, t_nsamples)
    sol = odeint(func=ode_sys,
                    y0=[s_init, i_init, l_init],
                    t=t_eval,
                    args=( a, beta, mu, alpha, sigma, K))
    s,l,i = sol.T
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_eval, y=s, name='Suscetível',
                             line=dict(color='#00b400', width=4)))
    fig.add_trace(go.Scatter(x=t_eval, y=l, name ='Latente',
                             line=dict(color='#ff0000', width=4, dash='dot')))
    fig.add_trace(go.Scatter(x=t_eval, y=i, name='Infectado',
                             line=dict(color='#0000ff', width=4, dash='dashdot')))
    fig.update_layout(title='Dinâmica Modelo SLI',
                       xaxis_title='Tempo (anos)',
                       yaxis_title='Indivíduos')
    return fig

app.layout = dbc.Container([
                cabecalho,
                dbc.Row([
                        dbc.Col(html.Div(ajuste_parametros), width=3),
                        dbc.Col(html.Div([ajuste_condicoes_iniciais,html.Div(textos_descricao)]), width=3),
                        dbc.Col(dcc.Graph(id='population_chart', className="shadow-sm rounded-3 border-primary",
                                style={'height': '500px'}), width=6),
                ]),
              ], fluid=True),


if __name__ == '__main__':
    app.run(debug=False)
