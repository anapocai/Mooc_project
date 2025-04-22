import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

# Chargement des données (exemple adapté)
df = pd.read_csv("employ.csv")

# Nettoyage des données
df['time'] = df['time'].astype(str)
df = df[df['indicator.label'].str.contains("8.5.2")]  # Filtrer sur le taux de chômage

# Initialisation de l'app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Impact Mondial du COVID-19 sur le Chômage (SDG 8.5.2)"),
    
    # Contrôles interactifs
    html.Div([
        dcc.Dropdown(
            id='country-dropdown',
            options=[{'label': c, 'value': c} for c in sorted(df['ref_area.label'].unique())],
            value=['Afghanistan', 'France', 'United States'],  # Exemples par défaut
            multi=True,
            placeholder="Sélectionner des pays"
        ),
        dcc.Dropdown(
            id='age-dropdown',
            options=[{'label': a.split(":")[-1], 'value': a} 
                    for a in df['classif1.label'].unique()],
            value="Age (Youth, adults): 15+",
            placeholder="Groupe d'âge"
        ),
        dcc.RangeSlider(
            id='year-slider',
            min=2019,
            max=2022,
            step=1,
            marks={str(year): str(year) for year in range(2019, 2023)},
            value=[2020, 2021]  # Période COVID par défaut
        )
    ], style={'width': '80%', 'margin': 'auto'}),
    
    # Graphiques
    dcc.Graph(id='global-trend-chart'),
    dcc.Graph(id='gender-gap-chart'),
    
    # Carte choroplèthe (nécessite une colonne 'iso_alpha3' dans les données)
    dcc.Graph(id='world-map'),
    
    # Tableau de données
    dash_table.DataTable(
        id='datatable',
        columns=[{"name": i, "id": i} for i in ['ref_area.label', 'time', 'sex.label', 'classif1.label', 'obs_value']],
        style_table={'overflowX': 'auto'},
        page_size=10
    ),
    
    # Notes
    html.Div([
        html.H4("Notes méthodologiques"),
        html.Ul([
            html.Li("Données SDG 8.5.2 (taux de chômage) - Source OIT"),
            html.Li("Période COVID-19 : 2020-2021"),
            html.Li("Break in series signalé par 🔴 dans les tooltips")
        ])
    ])
])

# Callbacks
@app.callback(
    [Output('global-trend-chart', 'figure'),
     Output('gender-gap-chart', 'figure'),
     Output('world-map', 'figure'),
     Output('datatable', 'data')],
    [Input('country-dropdown', 'value'),
     Input('age-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_dashboard(selected_countries, selected_age, years):
    filtered_df = df[
        (df['ref_area.label'].isin(selected_countries)) & 
        (df['classif1.label'] == selected_age) &
        (df['time'].between(str(years[0]), str(years[1])))
    ]
    
    # 1. Courbe de tendance mondiale
    trend_fig = px.line(
        filtered_df[filtered_df['sex.label'] == 'Total'],
        x='time', y='obs_value', color='ref_area.label',
        title=f"Évolution du chômage ({selected_age.split(':')[-1]})",
        labels={'obs_value': 'Taux de chômage (%)', 'time': 'Année'},
        hover_data=['note_source.label']
    )
    trend_fig.update_traces(line=dict(width=3))
    
    # 2. Écart Hommes-Femmes
    gap_fig = px.bar(
        filtered_df[filtered_df['sex.label'].isin(['Male', 'Female'])],
        x='time', y='obs_value', color='sex.label',
        facet_col='ref_area.label',
        title="Écart de genre par pays",
        barmode='group'
    )
    
    # 3. Carte mondiale (nécessite des codes pays ISO3)
    map_fig = px.choropleth(
        filtered_df[filtered_df['time'] == str(years[1])],
        locations='ref_area.label',  # Remplacer par une colonne ISO3 si disponible
        locationmode='country names',
        color='obs_value',
        hover_name='ref_area.label',
        color_continuous_scale='Reds',
        title=f"Chômage en {years[1]} ({selected_age.split(':')[-1]})"
    )
    
    # 4. Mise à jour du tableau
    table_data = filtered_df.to_dict('records')
    
    return trend_fig, gap_fig, map_fig, table_data

if __name__ == '__main__':
    app.run_server(debug=True)