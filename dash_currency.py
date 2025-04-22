import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import os
import matplotlib.pyplot as plt
import io
import base64

# ===== read csv =====
file_names = [
    "2019_data.csv",
    "2020_data.csv",
    "2021_data.csv",
    "2022_data.csv",
    "2023_data.csv",
]
dataframes = [pd.read_csv(f) for f in file_names]
full_df = pd.concat(dataframes, ignore_index=True)

# get currency list
list1 = full_df["Ticker"].unique().tolist()
list2 = [i[:3] for i in list1]

Initial = [
    69.709999,
    109.629997,
    0.870090,
    0.785050,
    1.420100,
    1.339430,
    0.983510,
    6.877600,
    7.832000,
    1.362200,
]

list3 = [full_df[full_df["Ticker"] == ticker].copy() for ticker in list1]

# merge
all_data = []
for i in range(len(list3)):
    df = list3[i].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Currency"] = list2[i]
    df["Origin"] = df["Adj Close"]
    df["Relative"] = df["Origin"] / Initial[i]
    df["Smoothed"] = df["Relative"].rolling(window=20, min_periods=1).mean()
    df["Std"] = df["Relative"].rolling(window=20, min_periods=1).std()
    df["Daily Change"] = df["Origin"].pct_change()
    df["Volatility"] = df["Daily Change"].rolling(window=20, min_periods=1).std()
    all_data.append(df)

all_df = pd.concat(all_data, ignore_index=True)

# ===== create dash =====
app = Dash(__name__)
app.title = "Currency Dashboard"

app.layout = html.Div(
    [
        html.H1("📈 Currency Dashboard", style={"textAlign": "center"}),
        html.H2("1️⃣ Original Exchange Rate (Adj Close)", style={"textAlign": "center"}),
        dcc.Dropdown(
            id="dropdown-origin",
            options=[
                {"label": c, "value": c} for c in sorted(all_df["Currency"].unique())
            ],
            value=list2,
            multi=True,
        ),
        dcc.Graph(id="origin-graph"),
        html.P(
            "Shows raw exchange rates from 2019 to 2023 without normalization. Useful for observing actual market price changes.",
            style={"textAlign": "center"},
        ),
        html.H2("2️⃣ Smoothed Relative Exchange Rate", style={"textAlign": "center"}),
        dcc.Dropdown(
            id="dropdown-relative",
            options=[
                {"label": c, "value": c} for c in sorted(all_df["Currency"].unique())
            ],
            value=list2,
            multi=True,
        ),
        dcc.Graph(id="relative-graph"),
        html.P(
            "Displays normalized exchange rates relative to Jan 1, 2019. Smoothed using a 20-day rolling average to highlight long-term trends.",
            style={"textAlign": "center"},
        ),
        html.H2("3️⃣ Rolling Std Dev of Normalized Rates", style={"textAlign": "center"}),
        dcc.Dropdown(
            id="dropdown-std",
            options=[
                {"label": c, "value": c} for c in sorted(all_df["Currency"].unique())
            ],
            value=list2,
            multi=True,
        ),
        dcc.Graph(id="std-graph"),
        html.P(
            "Shows rolling standard deviation (20-day window) of the normalized rates, indicating long-term volatility.",
            style={"textAlign": "center"},
        ),
        html.H2("4️⃣ Daily % Change Volatility", style={"textAlign": "center"}),
        dcc.Dropdown(
            id="dropdown-volatility",
            options=[
                {"label": c, "value": c} for c in sorted(all_df["Currency"].unique())
            ],
            value=list2,
            multi=True,
        ),
        dcc.Graph(id="volatility-graph"),
        html.P(
            "Displays rolling standard deviation of daily percentage change (20-day window), reflecting short-term market volatility.",
            style={"textAlign": "center"},
        ),
        html.H2(
            "5️⃣ Currency Volatility (2020–2022, Normalized Std Dev)",
            style={"textAlign": "center"},
        ),
        html.Img(id="std-bar-image"),
        html.P(
            "Bar chart comparing total normalized volatility (standard deviation) of each currency during 2020–2022.",
            style={"textAlign": "center"},
        ),
    ]
)


@app.callback(Output("origin-graph", "figure"), Input("dropdown-origin", "value"))
def plot_origin(currencies):
    fig = go.Figure()
    for cur in currencies:
        df = all_df[all_df["Currency"] == cur]
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Origin"], mode="lines", name=cur))
    fig.update_layout(
        title="Original Exchange Rate",
        xaxis_title="Date",
        yaxis_title="Adj Close",
        template="plotly_white",
    )
    return fig


@app.callback(Output("relative-graph", "figure"), Input("dropdown-relative", "value"))
def plot_relative(currencies):
    fig = go.Figure()
    for cur in currencies:
        df = all_df[all_df["Currency"] == cur]
        fig.add_trace(
            go.Scatter(x=df["Date"], y=df["Smoothed"], mode="lines", name=cur)
        )
    fig.update_layout(
        title="Smoothed Relative Exchange Rate",
        xaxis_title="Date",
        yaxis_title="Relative to 2019-01-01",
        template="plotly_white",
    )
    return fig


@app.callback(Output("std-graph", "figure"), Input("dropdown-std", "value"))
def plot_std(currencies):
    fig = go.Figure()
    for cur in currencies:
        df = all_df[all_df["Currency"] == cur]
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Std"], mode="lines", name=cur))
    fig.update_layout(
        title="Rolling Std Dev of Normalized Exchange Rate",
        xaxis_title="Date",
        yaxis_title="Std Dev",
        template="plotly_white",
    )
    return fig


@app.callback(
    Output("volatility-graph", "figure"), Input("dropdown-volatility", "value")
)
def plot_volatility(currencies):
    fig = go.Figure()
    for cur in currencies:
        df = all_df[all_df["Currency"] == cur]
        fig.add_trace(
            go.Scatter(x=df["Date"], y=df["Volatility"], mode="lines", name=cur)
        )
    fig.update_layout(
        title="20-Day Volatility of Daily % Change",
        xaxis_title="Date",
        yaxis_title="Volatility",
        template="plotly_white",
    )
    return fig


@app.callback(Output("std-bar-image", "src"), Input("origin-graph", "figure"))
def update_bar(_):
    subset = all_df[(all_df["Date"] >= "2020-01-01") & (all_df["Date"] <= "2022-12-31")]
    std_vals = subset.groupby("Currency")["Relative"].std().sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(std_vals.index, std_vals.values, color="skyblue")
    plt.xlabel("Standard Deviation (Volatility)")
    plt.title("Currency Volatility (2020–2022, Normalized)")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"data:image/png;base64,{data}"


if __name__ == "__main__":
    app.run(debug=True)
