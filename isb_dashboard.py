import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title='ISB Traffic Dashboard', layout='wide')

st.title('🏫 ISB Website Traffic Forecast Dashboard')
st.markdown('Interactive traffic forecasting powered by Machine Learning')

# ── Load data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('/Users/20214/Desktop/gsc_pages_daily.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# ── Sidebar controls ──────────────────────────────────────────────
st.sidebar.header('Controls')

top_pages = (
    df.groupby('page')['clicks']
    .sum()
    .sort_values(ascending=False)
    .head(30)
    .index.tolist()
)

# Show clean path instead of full URL
def clean_label(url):
    url = url.replace('https://www.isb.ac.th', '')
    url = url.replace('https://ee.isb.ac.th', '[ee]')
    url = url.replace('https://inside.isb.ac.th', '[inside]')
    return url or '/ (homepage)'

page_labels  = [clean_label(p) for p in top_pages]
selected_idx = st.sidebar.selectbox('Select Page', range(len(top_pages)), format_func=lambda i: page_labels[i])
selected_page = top_pages[selected_idx]

forecast_days = st.sidebar.slider('Forecast Days', min_value=30, max_value=180, value=90, step=30)
metric        = st.sidebar.radio('Metric', ['clicks', 'impressions'])

# ── Filter data for selected page ─────────────────────────────────
page_data = df[df['page'] == selected_page]
page_df   = page_data[['date', metric]].copy()
page_df   = page_df.rename(columns={'date': 'ds', metric: 'y'})
page_df   = page_df.sort_values('ds').reset_index(drop=True)

# ── Stats row ─────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric('Total Clicks',      f"{page_data['clicks'].sum():,.0f}")
col2.metric('Total Impressions', f"{page_data['impressions'].sum():,.0f}")
col3.metric('Avg Daily Clicks',  f"{page_data['clicks'].mean():.1f}")
col4.metric('Days of Data',      f"{len(page_df)}")

st.markdown('---')

# ── Train Prophet ─────────────────────────────────────────────────
@st.cache_data
def run_forecast(page, metric, forecast_days):
    page_df = df[df['page'] == page][['date', metric]].copy()
    page_df = page_df.rename(columns={'date': 'ds', metric: 'y'})
    page_df = page_df.sort_values('ds').reset_index(drop=True)

    if len(page_df) < 30:
        return None, None, None, None

    HOLDOUT = 30
    train = page_df.iloc[:-HOLDOUT]
    test  = page_df.iloc[-HOLDOUT:]

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,
    )
    m.fit(train)

    future   = m.make_future_dataframe(periods=HOLDOUT + forecast_days)
    forecast = m.predict(future)
    forecast['yhat']       = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

    preds  = forecast[forecast['ds'].isin(test['ds'])]['yhat'].values
    actual = test['y'].values
    mae    = mean_absolute_error(actual, preds)
    mape   = np.mean(np.abs((actual - preds) / (actual + 1))) * 100

    return page_df, forecast, mae, mape

with st.spinner('Training forecast model...'):
    page_df, forecast, mae, mape = run_forecast(selected_page, metric, forecast_days)

if forecast is None:
    st.warning('Not enough data to forecast this page.')
else:
    last_date   = page_df['ds'].max()
    future_only = forecast[forecast['ds'] > last_date]

    # ── Main forecast chart ───────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=page_df['ds'], y=page_df['y'],
        name='Actual', line=dict(color='black', width=1.5), opacity=0.8
    ))

    fig.add_trace(go.Scatter(
        x=future_only['ds'], y=future_only['yhat'],
        name='Forecast', line=dict(color='steelblue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=pd.concat([future_only['ds'], future_only['ds'][::-1]]),
        y=pd.concat([future_only['yhat_upper'], future_only['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(70,130,180,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Uncertainty range'
    ))

    fig.add_vline(x=str(last_date), line_dash='dash', line_color='red', opacity=0.5)

    fig.update_layout(
        title=f'{metric.capitalize()} Forecast — {clean_label(selected_page)}',
        xaxis_title='Date',
        yaxis_title=metric.capitalize(),
        hovermode='x unified',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Accuracy & summary ────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('MAE',                              f'{mae:.1f} per day')
    col2.metric('MAPE',                             f'{mape:.1f}%')
    col3.metric(f'Predicted {metric} ({forecast_days}d)', f'{future_only["yhat"].sum():,.0f}')
    col4.metric('Avg per day',                      f'{future_only["yhat"].mean():.1f}')

    st.markdown('---')

    # ── Monthly breakdown ─────────────────────────────────────────
    st.markdown('### 📅 Monthly Breakdown')
    monthly = future_only.copy()
    monthly['Month'] = monthly['ds'].dt.strftime('%B %Y')
    monthly_summary  = monthly.groupby('Month').agg(
        Predicted=('yhat',       lambda x: int(round(x.sum()))),
        Lower=    ('yhat_lower', lambda x: int(round(x.sum()))),
        Upper=    ('yhat_upper', lambda x: int(round(x.sum()))),
    ).reset_index()
    st.dataframe(monthly_summary, use_container_width=True)

    st.markdown('---')

    # ── Weekly seasonality chart ──────────────────────────────────
    st.markdown('### 📆 Weekly Traffic Pattern')
    weekly     = forecast[['ds', 'weekly']].copy()
    weekly['day'] = pd.to_datetime(weekly['ds']).dt.day_name()
    avg_weekly = weekly.groupby('day')['weekly'].mean().reindex(
        ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    )

    fig2 = go.Figure(go.Bar(
        x=avg_weekly.index,
        y=avg_weekly.values,
        marker_color=['steelblue' if d not in ['Saturday','Sunday'] else 'coral' for d in avg_weekly.index]
    ))
    fig2.update_layout(
        title='Which days get the most traffic?',
        yaxis_title='Seasonality Effect',
        height=350
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('---')

    # ── Yearly seasonality chart ──────────────────────────────────
    st.markdown('### 📈 Yearly Traffic Pattern')
    yearly     = forecast[['ds', 'yearly']].copy()
    yearly['month'] = pd.to_datetime(yearly['ds']).dt.strftime('%b')
    avg_yearly = yearly.groupby('month')['yearly'].mean().reindex(
        ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    )

    fig3 = go.Figure(go.Bar(
        x=avg_yearly.index,
        y=avg_yearly.values,
        marker_color='steelblue'
    ))
    fig3.update_layout(
        title='Which months get the most traffic?',
        yaxis_title='Seasonality Effect',
        height=350
    )
    st.plotly_chart(fig3, use_container_width=True)