import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
import numpy as np
from dash.dependencies import Input, Output, State

# ========================================
# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES
# ========================================

# Charger les fichiers
air_transport_df = pd.read_excel('./Air_passengers.xls', sheet_name='Data', skiprows=3)
rail_transport_df = pd.read_excel('./RailTrain_passengers.xls', sheet_name='Data', skiprows=3)
covid_restrictions_df = pd.read_csv('./RestrictionsCovid_country.csv', encoding='cp1252', delimiter=";")
countries_mapping = pd.read_csv('./iso_countries_full.csv', encoding='cp1252', delimiter=";")


# Nettoyer les noms de colonnes
def clean_column_names(df):
    df.columns = df.columns.str.strip()
    if 'Country' in df.columns:
        df = df.rename(columns={'Country': 'Country Name'})
    return df

air_transport_df = clean_column_names(air_transport_df)
rail_transport_df = clean_column_names(rail_transport_df)

# Filtrer les pays avec le mapping
valid_countries = countries_mapping[['Country Name', 'Country Code', 'Continent']].drop_duplicates()

# Fonction pour vérifier les données ferroviaires
def check_missing_rail_data(rail_df):
    rail_long = transform_transport_data(rail_df, 'rail')
    merged = rail_long.merge(
        valid_countries[['Country Code']],
        on='Country Code',
        how='inner'
    )
    return merged


# Fonction pour transformer les données de transport
def transform_transport_data(df, transport_type):
    df = df.drop(columns=["Indicator Name", "Indicator Code"], errors='ignore').copy()
    
    if 'Country Name' not in df.columns:
        if 'Country' in df.columns:
            df = df.rename(columns={'Country': 'Country Name'})
        else:
            raise ValueError("Colonne 'Country Name' introuvable")
    
    year_cols = [col for col in df.columns if col not in ['Country Name', 'Country Code']]
    
    df_long = pd.melt(
        df,
        id_vars=['Country Name', 'Country Code'],
        value_vars=year_cols,
        var_name="Year",
        value_name="Values"
    )
    
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
    df_long = df_long.dropna(subset=['Year', 'Values'])
    df_long['TransportType'] = transport_type
    
    return df_long

# Transformation des données
try:
    air_transport_long = transform_transport_data(air_transport_df, 'air')
    rail_transport_long = transform_transport_data(rail_transport_df, 'rail')
    rail_data_check = check_missing_rail_data(rail_transport_df)
except Exception as e:
    print(f"Erreur lors de la transformation des données: {str(e)}")
    raise

# Données mondiales - Air (utilisation directe de 'World')
world_air = air_transport_long[air_transport_long['Country Name'] == 'World']

# Données mondiales - Rail (somme des pays valides)
world_rail = rail_transport_long.merge(
    valid_countries[['Country Code']],
    on='Country Code',
    how='inner'
).groupby('Year')['Values'].sum().reset_index()
world_rail['TransportType'] = 'rail'
world_rail['Country Name'] = 'World (aggregated)'
world_rail['Country Code'] = 'WLD'

# Fusionner avec les restrictions COVID
covid_restrictions_df = covid_restrictions_df.rename(columns={
    "adm0_name": "Country Name", 
    "iso3": "Country Code"
}).drop_duplicates(subset=['Country Code'])

# Préparer les données pays avec gestion des données manquantes
def prepare_country_data(air_df, rail_df):
    years = [2018, 2019, 2020, 2021]
    
    air_df = air_df.merge(
        valid_countries[['Country Code', 'Country Name', 'Continent']],
        on=['Country Name', 'Country Code'],
        how='inner'
    )
    rail_df = rail_df.merge(
        valid_countries[['Country Code', 'Country Name', 'Continent']],
        on=['Country Name', 'Country Code'],
        how='inner'
    )
    
    # Préparation des données aériennes
    air_pivot = air_df[air_df['Year'].isin(years)].pivot_table(
        index=['Country Name', 'Country Code', 'Continent'],
        columns='Year',
        values='Values',
        aggfunc='sum'
    ).add_prefix('air_').reset_index()
    
    # Préparation des données ferroviaires
    rail_pivot = rail_df[rail_df['Year'].isin([2018, 2019, 2020, 2021])].pivot_table(
        index=['Country Name', 'Country Code', 'Continent'],
        columns='Year',
        values='Values',
        aggfunc='sum'
    ).add_prefix('rail_').reset_index()
    
    merged = air_pivot.merge(
        rail_pivot,
        on=['Country Name', 'Country Code', 'Continent'],
        how='outer'
    ).merge(
        covid_restrictions_df[['Country Code', 'Priority level', 'Restrictions']],
        on='Country Code',
        how='left'
    )
    
    # Initialiser les colonnes decline
    merged['air_decline'] = np.nan
    merged['rail_decline'] = np.nan
    merged['air_comparison'] = ''
    merged['rail_comparison'] = ''
    
    # Calcul des évolutions pour l'aérien
    if 'air_2019' in merged.columns and 'air_2021' in merged.columns:
        merged['air_decline'] = ((merged['air_2021'] - merged['air_2019']) / 
                                merged['air_2019']) * 100
        merged['air_comparison'] = '2019-2021'
    
    # Calcul des évolutions pour le ferroviaire
    if 'rail_2019' in merged.columns:
        # Essayer d'abord avec 2021 si disponible
        if 'rail_2021' in merged.columns:
            has_2021 = ~merged['rail_2021'].isna()
            merged.loc[has_2021, 'rail_decline'] = ((merged.loc[has_2021, 'rail_2021'] - 
                                                   merged.loc[has_2021, 'rail_2019']) / 
                                                  merged.loc[has_2021, 'rail_2019']) * 100
            merged.loc[has_2021, 'rail_comparison'] = '2019-2021'
        
        # Pour les autres, utiliser 2020 si disponible
        if 'rail_2020' in merged.columns:
            has_2020 = ~merged['rail_2020'].isna()
            mask = (has_2020 & (~has_2021 if 'rail_2021' in merged.columns else has_2020))
            merged.loc[mask, 'rail_decline'] = ((merged.loc[mask, 'rail_2020'] - 
                                              merged.loc[mask, 'rail_2019']) / 
                                             merged.loc[mask, 'rail_2019']) * 100
            merged.loc[mask, 'rail_comparison'] = '2019-2020'
    
    # Calcul des impacts absolus (en milliers de passagers)
    for ttype in ['air', 'rail']:
        if f'{ttype}_2019' in merged.columns:
            if f'{ttype}_2021' in merged.columns:
                merged[f'{ttype}_abs_impact'] = (merged[f'{ttype}_2021'] - merged[f'{ttype}_2019']) / 1000
            elif f'{ttype}_2020' in merged.columns:
                merged[f'{ttype}_abs_impact'] = (merged[f'{ttype}_2020'] - merged[f'{ttype}_2019']) / 1000
    
    # Calcul du total (adapté pour prendre en compte la nouvelle logique)
    if 'air_2019' in merged.columns and 'rail_2019' in merged.columns:
        merged['total_2019'] = merged['air_2019'] + merged['rail_2019']
        
        # Pour le total "latest", prendre soit 2021 soit 2020 selon ce qui est disponible
        merged['total_latest'] = merged.get('air_2021', merged.get('air_2020', np.nan)) + \
                               merged.get('rail_2021', merged.get('rail_2020', np.nan))
        
        merged['total_decline'] = ((merged['total_latest'] - merged['total_2019']) / 
                               merged['total_2019']) * 100
    
    merged['log_passengers'] = np.log10(merged['total_2019'].clip(lower=1))
    
    # Ne supprimer que les lignes où les deux déclins sont manquants
    if 'air_decline' in merged.columns and 'rail_decline' in merged.columns:
        return merged.dropna(subset=['air_decline', 'rail_decline'], how='all')
    elif 'air_decline' in merged.columns:
        return merged.dropna(subset=['air_decline'])
    elif 'rail_decline' in merged.columns:
        return merged.dropna(subset=['rail_decline'])
    else:
        return merged




country_data = prepare_country_data(air_transport_long, rail_transport_long)

# Préparer les données continentales
def prepare_continent_data(country_df):
    """Prépare les données agrégées par continent avec calcul des indicateurs"""
    if 'Continent' not in country_df.columns:
        raise ValueError("La colonne 'Continent' est manquante")
    
    # 1. Agrégation des données par continent
    continent_df = country_df.groupby('Continent').agg({
        'air_2018': 'sum',
        'air_2019': 'sum',
        'air_2020': 'sum',
        'air_2021': 'sum',
        'rail_2018': 'sum',
        'rail_2019': 'sum',
        'rail_2020': 'sum',
        'rail_2021': 'sum',
        'Priority level': lambda x: x.mode()[0] if not x.mode().empty else ''
    }).reset_index()

    # 2. Calcul des déclins
    # Pour l'aérien
    if 'air_2019' in continent_df.columns and 'air_2021' in continent_df.columns:
        continent_df['air_decline'] = ((continent_df['air_2021'] - continent_df['air_2019']) / 
                                     continent_df['air_2019']) * 100
    
    # Pour le ferroviaire (avec fallback sur 2020 si 2021 manquant)
    if 'rail_2019' in continent_df.columns:
        if 'rail_2021' in continent_df.columns:
            continent_df['rail_decline'] = ((continent_df['rail_2021'] - continent_df['rail_2019']) / 
                                         continent_df['rail_2019']) * 100
        elif 'rail_2020' in continent_df.columns:
            continent_df['rail_decline'] = ((continent_df['rail_2020'] - continent_df['rail_2019']) / 
                                         continent_df['rail_2019']) * 100

    # 3. Calcul de l'impact global (uniquement si les déclins existent)
    if 'air_decline' in continent_df.columns and 'rail_decline' in continent_df.columns:
        continent_df['global_impact'] = (
            abs(continent_df['air_decline'].fillna(0)) * 0.7 + 
            abs(continent_df['rail_decline'].fillna(0)) * 0.3
        )
    elif 'air_decline' in continent_df.columns:
        continent_df['global_impact'] = abs(continent_df['air_decline'].fillna(0))
    elif 'rail_decline' in continent_df.columns:
        continent_df['global_impact'] = abs(continent_df['rail_decline'].fillna(0))
    else:
        continent_df['global_impact'] = 0  # Valeur par défaut

    # 4. Conversion des unités
    for ttype in ['air', 'rail']:
        for year in [2018, 2019, 2020, 2021]:
            col = f'{ttype}_{year}'
            if col in continent_df.columns:
                # Conversion systématique en millions
                continent_df[col] = continent_df[col] / 1_000_000
                continent_df[f'{ttype}_unit'] = 'M'

    
    return continent_df

continent_data = prepare_continent_data(country_data)


def get_impact_color(impact_score):
    if impact_score > 70: 
        return '#b2182b'
    elif impact_score > 50: 
        return '#e66101'
    elif impact_score > 30: 
        return '#f7b801'
    else: 
        return '#4dac26'

def create_continent_info_cards():
    cards = []
    for idx, row in continent_data.iterrows():
        # Gestion sécurisée de global_impact
        impact_score = row.get('global_impact', 0)
        impact_color = get_impact_color(impact_score)
        
        def format_value(row, ttype):
            value = row.get(f'{ttype}_2021', row.get(f'{ttype}_2020', np.nan))
            unit = row.get(f'{ttype}_unit', 'M')
            return format_passengers(value, unit)
        
        card_content = [
            dbc.CardHeader(
                html.H4(row['Continent'], className="mb-0"),
                style={'backgroundColor': impact_color, 'color': 'white'}
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div("✈️ Trafic aérien", className="kpi-label"),
                        html.Div([
                            html.Span(format_value(row, 'air'), className="kpi-value"),
                            html.Span(f"({row['air_decline']:.1f}%)", 
                                     className="kpi-change",
                                     style={'color': '#dc3545' if row['air_decline'] < 0 else '#28a745'})
                        ], className="kpi-stack")
                    ], md=12)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.Div("🚆 Trafic ferroviaire", className="kpi-label"),
                        html.Div([
                            html.Span(format_value(row, 'rail'), className="kpi-value"),
                            html.Span(f"({row['rail_decline']:.1f}%)", 
                                     className="kpi-change",
                                     style={'color': '#dc3545' if row['rail_decline'] < 0 else '#28a745'})
                        ], className="kpi-stack")
                    ], md=12)
                ], className="mb-3"),
                
                html.Div([
                    html.Small("Impact COVID", className="text-muted d-block mb-1"),
                    dbc.Progress(
                        value=row['global_impact'],
                        max=100,
                        style={'height': '8px', 'marginBottom': '5px'},
                        color="danger" if row['global_impact'] > 50 else "warning"
                    ),
                    html.Small(f"Impact total: {row['global_impact']:.1f}%", 
                             className="text-muted")
                ])
            ])
        ]
        
        card = dbc.Card(
            card_content,
            className="continent-card mb-4",
            style={
                "borderLeft": f"5px solid {impact_color}",
                "height": "100%"
            }
        )
        cards.append(dbc.Col(card, md=6, lg=4))
    
    return dbc.Row(cards, className="mt-4")

# ========================================
# 2. FONCTIONS DE VISUALISATION
# ========================================

def format_passengers(value):
    """Formate les valeurs de passagers de manière cohérente dans tout le dashboard"""
    if pd.isna(value):
        return "N/A"
    
    value = float(value)
    abs_value = abs(value)
    
    # Règles basées sur l'analyse des données brutes
    if abs_value >= 1_000_000_000:  # ≥ 1 milliard
        return f"{value/1_000_000_000:,.1f}B"
    elif abs_value >= 100_000_000:   # ≥ 100 millions
        return f"{value/1_000_000:,.0f}M"
    elif abs_value >= 10_000_000:    # ≥ 10 millions
        return f"{value/1_000_000:,.1f}M"
    elif abs_value >= 1_000_000:     # ≥ 1 million
        return f"{value/1_000_000:,.2f}M"
    elif abs_value >= 100_000:       # ≥ 100 mille
        return f"{value/1_000:,.0f}K"
    elif abs_value >= 10_000:        # ≥ 10 mille
        return f"{value/1_000:,.1f}K"
    elif abs_value >= 1_000:         # ≥ 1 mille
        return f"{value/1_000:,.2f}K"
    else:                            # Valeurs brutes
        return f"{value:,.0f}"

def smart_format(value, transport_type):
    """Formatage intelligent selon le type de transport et la valeur"""
    if pd.isna(value):
        return "N/A"
    
    abs_value = abs(value)
    
    # Règles spécifiques par type de transport
    if transport_type == 'air':
        if abs_value >= 1_000_000:  # ≥ 1 million
            return f"{value/1_000_000:,.1f}M"
        elif abs_value >= 1_000:    # ≥ 1 mille
            return f"{value/1_000:,.0f}K"
        else:
            return f"{value:,.0f}"
    else:  # rail
        if abs_value >= 1_000_000:  # ≥ 1 million
            return f"{value/1_000_000:,.1f}M"
        elif abs_value >= 10_000:   # ≥ 10 mille
            return f"{value/1_000:,.0f}K"
        else:
            return f"{value:,.0f}"




def format_impact(value):
    """Formate spécifiquement les impacts (valeurs négatives)"""
    if pd.isna(value):
        return "N/A"
    
    value = float(value)
    abs_value = abs(value)
    
    if abs_value >= 1_000_000:
        return f"{value/1_000_000:,.1f}M"  # -260.6M
    elif abs_value >= 1_000:
        return f"{value/1_000:,.0f}K"      # -683K
    else:
        return f"{value:,.0f}"             # -123


def create_global_trend_chart(transport_type):
    years = [2018, 2019, 2020, 2021]
    
    if transport_type == 'air':
        df = world_air
        color = '#1f77b4'
        title = "Global Air Traffic Trends"
    else:
        df = world_rail
        color = '#ff7f0e'
        title = "Global Rail Traffic Trends"
    
    # Conversion systématique en millions
    values = []
    for y in years:
        val = df[df['Year'] == y]['Values'].sum() / 1_000_000
        values.append(val)
    if transport_type == 'air':
    # Pour l'aérien, pic attendu en 2020
        year_min = 2020
    else:
    # Pour le ferroviaire, pic attendu en 2021
        year_min = 2021

# Récupération de la valeur correspondante pour placer l'annotation
    min_value = df[df['Year'] == year_min]['Values'].sum() / 1_000_000
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years,
        y=values,
        mode='lines+markers+text',
        text=[f"{v:,.1f}M" for v in values],  # Format en millions
        textposition="top center",
        line=dict(color=color, width=3),
        marker=dict(size=10, color=color),
        name="Passengers (en millions)"
    ))
    
    # Amélioration de la zone COVID
    fig.add_vrect(
        x0=2019, x1=2021,
        fillcolor="rgba(220, 53, 69, 0.35)",
        layer="below",
        line_width=0,
        #annotation_text="Peak Pandemic",
        #annotation_position="top left",
        #annotation_font_size=12,
        #annotation_font_color="#dc3545"
    )
    
    fig.add_annotation(
        x=year_min,
        y=min_value,  # position de la valeur de la baisse la plus importante
        text="Peak of Pandemic",  # texte de l'annotation
        showarrow=True,  # afficher une flèche pointant vers l'annotation
        arrowhead=2,  # type de flèche
        ax=0,  # décalage de l'annotation sur l'axe x
        ay=-40,  # décalage de l'annotation sur l'axe y
        font=dict(
            size=12,  # taille de la police
            color="rgba(220, 53, 69, 1)"  # couleur de la police (associée à la couleur COVID)
        ),
        align="center",  # aligner l'annotation
        arrowcolor="rgba(220, 53, 69, 0.7)",  # couleur de la flèche
        bgcolor="rgba(255, 255, 255, 0.7)"  # fond de l'annotation pour la lisibilité
    )






    fig.update_layout(
        title=title,
        xaxis_title="Years",
        yaxis_title="Passengers (millions)",
        height=400,
        margin=dict(t=60, b=40),  # Plus d'espace en haut pour la légende
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, tickmode='array', tickvals=years),
        yaxis=dict(showgrid=False)
    )
    
    return fig

def create_continent_info_cards():
    cards = []
    for idx, row in continent_data.iterrows():
        impact_color = get_impact_color(row['global_impact'])
        
        def format_value(row, ttype):
            value = row[f'{ttype}_2021'] if not pd.isna(row[f'{ttype}_2021']) else row.get(f'{ttype}_2020', np.nan)
            return format_passengers(value * (1e6 if row.get(f'{ttype}_unit', 'M') == 'M' else 1e3))
        
        card_content = [
            dbc.CardHeader(
                html.H4(row['Continent'], className="mb-0"),
                style={'backgroundColor': impact_color, 'color': 'white'}
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div("✈️ Air Traffic", className="kpi-label"),
                        html.Div([
                            html.Span(format_value(row, 'air'), className="kpi-value"),
                            html.Span(f"({row['air_decline']:.1f}%)", 
                                     className="kpi-change",
                                     style={'color': '#dc3545' if row['air_decline'] < 0 else '#28a745'})
                        ], className="kpi-stack")
                    ], md=12)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.Div("🚆 Rail Traffic", className="kpi-label"),
                        html.Div([
                            html.Span(format_value(row, 'rail'), className="kpi-value"),
                            html.Span(f"({row['rail_decline']:.1f}%)", 
                                     className="kpi-change",
                                     style={'color': '#dc3545' if row['rail_decline'] < 0 else '#28a745'})
                        ], className="kpi-stack")
                    ], md=12)
                ], className="mb-3"),
                
                html.Div([
                    #html.Small("COVID Impact", className="text-muted d-block mb-1"),
                    dbc.Progress(
                        value=row['global_impact'],
                        max=100,
                        style={'height': '8px', 'marginBottom': '5px'},
                        color="danger" if row['global_impact'] > 50 else "warning"
                    ),
                    html.Small(f"Global Impact: {row['global_impact']:.1f}%", 
                             className="text-muted")
                ])
            ])
        ]
        
        card = dbc.Card(
            card_content,
            className="continent-card mb-4",
            style={
                "borderLeft": f"5px solid {impact_color}",
                "height": "100%"
            }
        )
        cards.append(dbc.Col(card, md=6, lg=4))
    
    return dbc.Row(cards, className="mt-4")



def create_country_map(transport_type, metric_type='relative'):
    # Créer une copie des données
    plot_data = country_data.copy()
    
    if metric_type == 'absolute':
        # Nouvelle logique pour inclure les pays avec données 2020
        has_2021 = plot_data[f'{transport_type}_2021'].notna()
        has_2020 = plot_data[f'{transport_type}_2020'].notna()
        
        # Calcul de l'impact absolu personnalisé
        plot_data['display_value'] = np.where(
            has_2021,
            plot_data[f'{transport_type}_abs_impact'],
            plot_data[f'{transport_type}_2019'] - plot_data[f'{transport_type}_2020']
        )
        
        # Conversion des unités
        if transport_type == 'air':
            plot_data['display_value'] = plot_data['display_value'] / 1000  # Millions
            colorbar_title = "Loss (millions)"
        else:
            colorbar_title = "Loss (thousands)"
            
        col = 'display_value'
        title_suffix = " (Total Passengers Difference)"
        
        # Filtre pour garder seulement les pays avec données valides
        plot_data = plot_data[
            plot_data[f'{transport_type}_2019'].notna() & 
            (has_2021 | has_2020) &
            plot_data['display_value'].notna()
        ].copy()
        
    else:
        # Vue relative (comportement existant)
        col = f'{transport_type}_decline'
        title_suffix = " (Percentage Change vs. 2019)"
        colorbar_title = "Change (%)"
        plot_data = plot_data.dropna(subset=[col]).copy()
    
    # Création de la carte
    fig = px.choropleth(
        plot_data,
        locations="Country Code",
        color=col,
        hover_name="Country Name",
        color_continuous_scale=[[0, 'red'], [0.5, 'yellow'], [1, 'green']],
        range_color=(-100, 20) if metric_type == 'relative' else [plot_data[col].min(), 0],
        labels={col: colorbar_title},
        title = f"COVID-19 Impact on {transport_type.capitalize()} Traffic ({'Percentage' if metric_type == 'relative' else 'Absolute'} Change)",
        projection='natural earth'
    )
    
    # Tooltip minimaliste
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><extra></extra>"
    )
    
    # Ajustement des échelles
    if metric_type == 'absolute':
        fig.update_layout(
            coloraxis=dict(
                cmin=-300 if transport_type == 'air' else -1_000,
                cmax=0
            )
        )
    
    return fig

def create_top10_chart(transport_type, metric_type='relative'):
    # Copie des données et fonction de formatage
    plot_data = country_data.copy()
    
    def format_value(x, transport):
        if pd.isna(x):
            return "N/A"
        if transport == 'air' and x >= 1e6:
            return f"{x/1e6:,.1f}M".replace(",", " ")
        elif x >= 1e3:
            return f"{x/1e3:,.1f}K".replace(",", " ")
        return f"{x:,.0f}"
    
    # Configuration des métriques
    if metric_type == 'absolute':
        if transport_type == 'air':
            plot_data['display_value'] = plot_data[f'{transport_type}_abs_impact'] / 1000
            xaxis_title = "Passenger loss (millions)"
            hover_suffix = "M"
            cmin = -300
        else:
            plot_data['display_value'] = plot_data[f'{transport_type}_abs_impact']
            xaxis_title = "Passenger loss (thousands)"
            hover_suffix = "K"
            cmin = -1000
        
        title_suffix = " (Total passengers difference)"
        cmax = 0
        tickformat = ",.1f" if transport_type == 'air' else ",.0f"
        xaxis_range = [cmin * 1.1, cmax]
        col = 'display_value'
    else:
        col = f'{transport_type}_decline'
        plot_data['display_value'] = plot_data[col] / 100
        title_suffix = "(Percentage Change)"
        xaxis_title = "Change (%)"
        hover_suffix = "%"
        cmin = -1
        cmax = 0
        tickformat = ".0%"
        xaxis_range = [-1.05, 0.05]
    
    # Filtrage et préparation des données
    plot_data = plot_data.dropna(subset=[col]).copy()
    plot_data = plot_data.nsmallest(10, col).sort_values(col, ascending=True)
    
    # Gestion des années de comparaison
    plot_data['use_2020'] = plot_data[f'{transport_type}_2021'].isna()
    plot_data['comparison_value'] = np.where(
        plot_data['use_2020'],
        plot_data[f'{transport_type}_2020'],
        plot_data[f'{transport_type}_2021']
    )
    
    # Formatage des textes
    plot_data['text_2019'] = plot_data[f'{transport_type}_2019'].apply(
        lambda x: format_value(x, transport_type))
    
    plot_data['text_comparison'] = plot_data['comparison_value'].apply(
        lambda x: format_value(x, transport_type))
    
    # Création du graphique
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=plot_data['Country Name'],
        x=plot_data['display_value'],
        orientation='h',
        marker=dict(
            color=plot_data['display_value'],
            colorscale=[[0, 'red'], [0.5, 'yellow'], [1, 'green']],
            cmin=cmin,
            cmax=cmax
        ),
        text=[f"{x*100:.1f}%" if metric_type == 'relative' else f"{x:.1f}{hover_suffix}" 
              for x in plot_data['display_value']],
        textposition='outside',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "2019: %{customdata[0]}<br>"
            "%{customdata[3]}: %{customdata[1]}<br>"
            "Passengers: %{customdata[1]}<br>"
            "Period: %{customdata[4]}<br>"
            "Percentage change: %{customdata[2]:.1f}%<br>"
            "<extra></extra>"
        ),
        customdata=np.stack((
            plot_data['text_2019'],
            plot_data['text_comparison'],
            plot_data[col],
            np.where(plot_data['use_2020'], "2020", "2021"),
            np.where(plot_data['use_2020'], "2019-2020", "2019-2021")
        ), axis=-1)
    ))
    
    # Configuration finale
    fig.update_layout(
        title = f"Top 10 Declines in {transport_type.capitalize()} Traffic{title_suffix}",
        yaxis={'categoryorder':'total ascending'},
        xaxis_title=xaxis_title,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            side='top',
            range=xaxis_range,
            tickformat=tickformat,
            tickvals=[-1, -0.75, -0.5, -0.25, 0] if metric_type == 'relative' else None
        )
    )
    
    '''
    if metric_type == 'relative':
        fig.update_traces(
            marker_colorbar=dict(
                title='Change',
                tickformat=".0%",
                tickvals=[-1, -0.75, -0.5, -0.25, 0]
            )
        )
    '''
    return fig

# ========================================
# 3. APPLICATION DASH
# ========================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
app.title = "COVID-19 Impact on Global Transportation"

app.layout = dbc.Container(fluid=True, children=[
    dcc.Store(id='selected-transport-mode', data='air'),
    dcc.Store(id='selected-metric-type', data='relative'),
    
    # Section Hero
    dbc.Row(
        dbc.Col(
            html.Div(
                className="hero-section", 
                children=[
                    html.H1("The Global transport disruption", className="hero-title",style={
                        'color': 'white',
                        'text-align': 'center',  # Centrage horizontal
                        'margin': '0 auto',      # Centrage complémentaire
                        'width': '100%'          # Nécessaire pour le margin auto
                    }),
                    html.P("Measuring COVID-19's impact on worldwide mobility", className="hero-subtitle",style={'text-align': 'center'}),
                    html.Div([
                        html.Button("✈️ Air", id="btn-air",className="transport-btn",style={
                            'background-color': '#dc3545',  # Rouge vif
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 20px',
                            'margin-right': '10px',
                            'font-weight': 'bold',
                            'border-radius': '5px'
                        }
                                   ,n_clicks=0
                    ),
                        html.Button("🚆 Rail", id="btn-rail",className="transport-btn", style={
                            'background-color': '#6c757d',  # Gris
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 20px',
                            'font-weight': 'bold',
                            'border-radius': '5px'
                        }
                        , n_clicks=0)
                    ], className="mt-3"),
                    html.Div(id='transport-type-indicator', className="mt-2 small", style={'fontWeight': 'bold', 'color': 'white'})
                ]
            ),
            className="mb-4 py-4 bg-primary text-white"
        )
    ),
    
    # Section Évolution mondiale
    dbc.Row(
        dbc.Col(
            html.Div(
                [
                    html.H2("Global Traffic Trends: A Worldwide Overview", className="section-title"),
                    dcc.Graph(id='global-trend-chart', figure=create_global_trend_chart('air'))
                ], 
                className="mb-4"
            )
        )
    ),
    
    # Section Analyse par continent (AJOUTÉE)
    dbc.Row(
        dbc.Col(
            html.Div(
                [
                    html.H2("Continental Analysis", className="section-title"),
                    html.P("Understanding the Global Impact: Comparing 2019 to 2021", className="lead mb-3"),
                    create_continent_info_cards()
                ], 
                className="mb-4"
            )
        )
    ),


    # Section Visualisation géographique
    dbc.Row([
        dbc.Col(
            dcc.RadioItems(
                id='metric-type',
                options=[
                    {'label': ' Percentage change vs. 2019 ', 'value': 'relative'},
                    {'label': ' Total passengers difference', 'value': 'absolute'}
                ],
                value='relative',
                inline=True,
                labelStyle={'margin-right': '20px'},
                className="mb-3"
            ),
            width=12
        )
    ]),
    dbc.Row([
        dbc.Col(
            html.Div(
                [
                    html.H2("Interactive World Map: Pandemic's Travel Impact", className="section-title"),
                    dcc.Graph(
                        id='country-map',
                        figure=create_country_map('air'),
                        config={'displayModeBar': False},
                        className="border rounded bg-white"
                    )
                ], 
                className="mb-4"
            ),
            md=8
        ),
        dbc.Col(
            html.Div(
                id='country-info-panel',
                className="border rounded bg-white p-3",
                style={'height': '580px', 'overflowY': 'auto', 'maxHeight': '580px'}
            ),
            md=4
        )
    ]),
    
    # Section Top 10 (maintenant réactive à la métrique)
    dbc.Row(
        dbc.Col(
            html.Div(
                [
                    html.H2("Top 10 Most Affected Countries", className="section-title"),
                    dcc.Graph(
                        id='top10-chart',
                        figure=create_top10_chart('air'),
                        config={'displayModeBar': False},
                        className="border rounded bg-white"
                    )
                ], 
                className="mb-4"
            )
        )
    )
], style={'maxWidth': '1400px'})

# ========================================
# 4. CALLBACKS
# ========================================

@app.callback(
    [Output('global-trend-chart', 'figure'),
     Output('country-map', 'figure'),
     Output('top10-chart', 'figure'),
     Output('btn-air', 'style'),
     Output('btn-rail', 'style'),
     Output('transport-type-indicator', 'children'),
     Output('selected-transport-mode', 'data')],
    [Input('btn-air', 'n_clicks'),
     Input('btn-rail', 'n_clicks'),
     Input('metric-type', 'value')],
    [State('selected-transport-mode', 'data')]
)
def update_transport_view(btn_air, btn_rail, metric_type, current_transport):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        transport_type = 'air'
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        transport_type = 'air' if trigger_id == 'btn-air' else 'rail' if trigger_id == 'btn-rail' else current_transport or 'air'
    
    # Styles dynamiques
    btn_air_style = {
        'background-color': '#dc3545' if transport_type == 'air' else '#6c757d',
        'color': 'white',
        'border': 'none',
        'padding': '10px 20px',
        'margin-right': '10px',
        'font-weight': 'bold',
        'border-radius': '5px'
    }
    
    btn_rail_style = {
        'background-color': '#28a745' if transport_type == 'rail' else '#6c757d',
        'color': 'white',
        'border': 'none',
        'padding': '10px 20px',
        'font-weight': 'bold',
        'border-radius': '5px'
    }
    
    return [
        create_global_trend_chart(transport_type),
        create_country_map(transport_type, metric_type),
        create_top10_chart(transport_type, metric_type),
        btn_air_style,
        btn_rail_style,
       f"Viewing {'air travel' if transport_type == 'air' else 'rail'}",
        transport_type
    ]

@app.callback(
    Output('country-info-panel', 'children'),
    [Input('country-map', 'hoverData'),
     Input('selected-transport-mode', 'data'),
     Input('metric-type', 'value')]
)
def update_country_info(hover_data, transport_type, metric_type):
    if not hover_data or not hover_data.get('points'):
        return html.Div(
            "Hover over a country to see the details.",
            className="text-muted text-center",
            style={'marginTop': '250px'}
        )

    try:
        country_code = hover_data['points'][0]['location']
        country_info = country_data[country_data['Country Code'] == country_code].iloc[0]
        
        # Initialisation des valeurs par défaut
        values = [np.nan, np.nan, np.nan, np.nan]
        color = '#1f77b4' if transport_type == 'air' else '#ff7f0e'
        title = "Air Traffic" if transport_type == 'air' else "Rail Traffic"
        decline = np.nan
        abs_impact = np.nan
        comparison_period = "N/A"
        
        # Récupération des valeurs selon le type de transport
        if transport_type == 'air':
            if pd.notna(country_info.get('air_2019')):
                values = [
                    country_info.get('air_2018', np.nan)/1e6,
                    country_info.get('air_2019', np.nan)/1e6,
                    country_info.get('air_2020', np.nan)/1e6,
                    country_info.get('air_2021', np.nan)/1e6
                ]
                decline = country_info.get('air_decline', np.nan)
                abs_impact = country_info.get('air_abs_impact', np.nan)/1e3  # Conversion en millions
                comparison_period = "2019-2021"
        else:
            if pd.notna(country_info.get('rail_2019')):
                if pd.notna(country_info.get('rail_2021')):
                    values = [
                        country_info.get('rail_2018', np.nan)/1e6,
                        country_info.get('rail_2019', np.nan)/1e6,
                        country_info.get('rail_2020', np.nan)/1e6,
                        country_info.get('rail_2021', np.nan)/1e6
                    ]
                    comparison_period = "2019-2021"
                elif pd.notna(country_info.get('rail_2020')):
                    values = [
                        country_info.get('rail_2018', np.nan)/1e6,
                        country_info.get('rail_2019', np.nan)/1e6,
                        country_info.get('rail_2020', np.nan)/1e6,
                        np.nan
                    ]
                    comparison_period = "2019-2020"
                
                decline = country_info.get('rail_decline', np.nan)
                abs_impact = country_info.get('rail_abs_impact', np.nan)/1e3  # Conversion en millions

        # Formatage des valeurs
        prev_value = values[1]  # 2019
       # Détermine si on utilise 2020 ou 2021
        use_2020 = comparison_period == "2019-2020"
        current_value = values[2] if use_2020 else values[3]  # 2020 ou 2021
        absolute_value = values[2] -  values[1] if use_2020 else values[3] - values[1]
        # Formatage des valeurs (affiche toujours la valeur disponible)
        prev_formatted = format_passengers(prev_value * 1e6) if pd.notna(prev_value) else "N/A"
        current_formatted = format_passengers(current_value * 1e6) if pd.notna(current_value) else "N/A"
        absolute_formatted = format_passengers(absolute_value * 1e6) if pd.notna(absolute_value) else "N/A"
        year_shown = "2020" if use_2020 else "2021"
        decline_formatted = f"{decline:.1f}%" if pd.notna(decline) else "N/A"
        # Par :
        abs_impact = country_info.get(f'{transport_type}_abs_impact', np.nan)
        if pd.notna(abs_impact):
            if transport_type == 'air':
                abs_impact_formatted = f"{abs_impact/1000:,.0f}M" if abs(abs_impact) >= 1000 else f"{abs_impact:,.0f}K"
            else:
                abs_impact_formatted = f"{abs_impact:,.0f}K"
        else:
            abs_impact_formatted = "N/A"

        restrictions = str(country_info.get('Restrictions', 'Données non disponibles'))
        priority_level = str(country_info.get('Priority level', 'Non spécifié'))

        color_mapping = {
            'Very High': '#ff0000',  # Rouge vif
            'High': '#ff9900',       # Orange vif
            'Medium': '#0066cc',     # Bleu franc
            'Low': '#00aa00',        # Vert clair
        }
        restriction_badge = dbc.Badge(
            priority_level,
            color=color_mapping.get(priority_level, '#666666'),  # Gris moyen
            className="me-1"
        )

        return html.Div([
            html.H4(country_info['Country Name'], className="mb-3"),
            html.P(country_info.get('Continent', ''), className="text-muted mb-4"),
            
            dbc.Card([
                dbc.CardHeader(title, className="text-center"),
                dbc.CardBody([
                    html.H4(
                        decline_formatted,
                        className="text-center mb-2",
                        style={
                            'color': '#dc3545' if (pd.notna(decline) and decline < 0) else '#28a745',
                            'fontSize': '2rem'
                        }
                    ),
                    html.P(
                        f"2019: {prev_formatted} → {year_shown}: {current_formatted}", 
                        className="text-center text-muted mb-0"
                    ),
                    html.P(
                        f"Absolute loss: {absolute_formatted} passengers",
                        className="text-center text-muted mb-0"
                    ),
                    html.Small(
                        f"Comparison period: {comparison_period}",
                        className="text-muted d-block mb-2"
                    ) if transport_type == 'rail' else None
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("COVID Restrictions", className="text-center"),
                dbc.CardBody([
                    html.Div(restriction_badge, className="text-center mb-2"),
                    html.Div(
                        restrictions if len(restrictions) < 500 else restrictions[:500] + "...",
                        className="small",
                        style={'maxHeight': '200px', 'overflowY': 'auto'}
                    )
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Trend 2018–2021", className="text-center"),
                dbc.CardBody([
                    dcc.Graph(
                        figure={
                            'data': [{
                                'x': [2018, 2019, 2020, 2021],
                                'y': values,
                                'mode': 'lines+markers+text',
                                'text': [format_passengers(v * 1e6) if pd.notna(v) else "N/A" for v in values],
                                'textposition': 'top center',
                                'textfont': {'size': 10},
                                'line': {'color': color, 'width': 2},
                                'marker': {'size': 8}
                            }],
                            'layout': {
                                'margin': {'l': 30, 'r': 30, 't': 10, 'b': 30},
                                'height': 200,
                                'plot_bgcolor': 'rgba(0,0,0,0)',
                                'xaxis': {'showgrid': False},
                                'yaxis': {'showgrid': False, 'rangemode': 'tozero'}
                            }
                        },
                        config={'displayModeBar': False},
                        style={'height': '200px'}
                    )
                ], className="py-2")
            ])
        ])
    except Exception as e:
        print(f"Erreur dans la génération du contenu: {str(e)}")
        return html.Div(
            "Erreur lors du chargement des données",
            className="text-danger text-center",
            style={'marginTop': '250px'}
        )

# ========================================
# 5. LANCEMENT DE L'APPLICATION
# ========================================

if __name__ == '__main__':
    app.run(debug=True)