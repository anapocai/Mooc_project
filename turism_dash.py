import dash
from dash import dcc, html
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import webbrowser
from threading import Timer
import pandas as pd
import pycountry

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # Required for deployment

# Data loading and preprocessing
def load_and_preprocess_data():
    # Load tourism data
    df_tourism = pd.read_csv("turism.csv")

    # Debug: print columns to ensure 'DATE' exists
    print("Columns:", df_tourism.columns)

    # Remove leading/trailing spaces from column names
    df_tourism.columns = df_tourism.columns.str.strip()

    # Convert 'DATE' to datetime
    df_tourism['DATE'] = pd.to_datetime(df_tourism['DATE'], errors='coerce')

    # Drop rows where DATE conversion failed
    df_tourism = df_tourism.dropna(subset=['DATE'])

    # Country to ISO3 mapping
    def get_iso3(country_name):
        try:
            return pycountry.countries.lookup(country_name).alpha_3
        except:
            return None

    # Add ISO3 codes
    df_tourism['ISO3'] = df_tourism['Country'].apply(get_iso3)

    # Filter COVID period (June 2019 - Dec 2021)
    covid_period = df_tourism[(df_tourism['DATE'] >= '2019-06-01') & 
                              (df_tourism['DATE'] <= '2021-12-31')].copy()
    covid_period['Year'] = covid_period['DATE'].dt.year
    covid_period['Month'] = covid_period['DATE'].dt.month

    # Group by continent, year, month
    covid_grouped = covid_period.groupby(['CONTINENT', 'Year', 'Month'])['QUANTITY'].sum().reset_index()

    # Prepare sea and air transport data for 2020
    covid_2020 = df_tourism[(df_tourism['DATE'].dt.year == 2020) & 
                            (df_tourism['DATE'].dt.month >= 3)].copy()
    covid_2020['Year'] = covid_2020['DATE'].dt.year
    covid_2020['Month'] = covid_2020['DATE'].dt.month

    # Strip spaces from transport method column (just in case)
    df_tourism['TRANSP. \nMETHOD'] = df_tourism['TRANSP. \nMETHOD'].str.strip()

    sea_data = covid_2020[covid_2020['TRANSP. \nMETHOD'] == 'Sea']
    air_data = covid_2020[covid_2020['TRANSP. \nMETHOD'] == 'Air']

    sea_grouped = sea_data.groupby(['Year', 'Month', 'Country', 'ISO3'])['QUANTITY'].sum().reset_index()
    air_grouped = air_data.groupby(['Year', 'Month', 'Country', 'ISO3'])['QUANTITY'].sum().reset_index()

    return covid_grouped, sea_grouped, air_grouped

# Load data
covid_grouped, sea_covid_monthly_grouped, air_covid_monthly_grouped = load_and_preprocess_data()

# Visualization functions
def create_continent_trends_plot():
    plt.figure(figsize=(12, 6))
    
    for continent in covid_grouped['CONTINENT'].unique():
        continent_data = covid_grouped[covid_grouped['CONTINENT'] == continent]
        plt.plot(continent_data['Year'].astype(str) + '-' + continent_data['Month'].astype(str), 
                 continent_data['QUANTITY'], label=continent)

    plt.xlabel('Year-Month')
    plt.ylabel('Number of Tourists')
    plt.title('Tourism Trends by Continent Before and After COVID-19')
    plt.xticks(rotation=45)
    plt.legend(title='Continent')
    plt.grid(True)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf8')

def create_choropleth(data, title):
    return px.choropleth(
        data,
        locations='ISO3',
        color='QUANTITY',
        hover_name='Country',
        animation_frame='Month',
        color_continuous_scale='Viridis',
        projection='natural earth',
        title=title
    ).update_geos(scope='world')

# Dashboard layout
app.layout = html.Div([
    html.H1("Tourism Analysis During COVID-19", style={'textAlign': 'center'}),

    html.Div([
        html.H2("Tourism Trends by Continent"),
        html.Img(src='data:image/png;base64,{}'.format(create_continent_trends_plot()))
    ], style={'padding': '20px', 'border': '1px solid #ddd', 'margin': '10px'}),

    html.Div([
        html.H2("Sea Transportation Tourism"),
        dcc.Graph(
            id='sea-transport',
            figure=create_choropleth(
                sea_covid_monthly_grouped,
                'Sea Transportation Tourism During COVID-19'
            )
        )
    ], style={'padding': '20px', 'border': '1px solid #ddd', 'margin': '10px'}),

    html.Div([
        html.H2("Air Transportation Tourism"),
        dcc.Graph(
            id='air-transport',
            figure=create_choropleth(
                air_covid_monthly_grouped,
                'Air Transportation Tourism During COVID-19'
            )
        )
    ], style={'padding': '20px', 'border': '1px solid #ddd', 'margin': '10px'})
])

def open_browser():
    try:
        webbrowser.open_new("http://localhost:8050")
    except:
        print("Could not open browser automatically. Please manually navigate to http://localhost:8050")

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run_server(debug=True, port=8050)
