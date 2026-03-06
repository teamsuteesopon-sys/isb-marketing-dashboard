import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='ISB Social Media Dashboard', layout='wide')

st.title('ISB Social Media Performance Dashboard')
st.markdown('Analyze past performance and predict engagement for future posts')

@st.cache_data
def load_data():
    fb = pd.read_csv('/Users/20214/Desktop/facebook_data.csv')
    ig = pd.read_csv('/Users/20214/Desktop/instagram_data.csv')

    fb['datetime'] = pd.to_datetime(fb['Publish time'], format='%m/%d/%Y %H:%M')
    ig['datetime'] = pd.to_datetime(ig['Publish time'], format='%m/%d/%Y %H:%M')

    for df in [fb, ig]:
        df['day_of_week']    = df['datetime'].dt.day_name()
        df['hour']           = df['datetime'].dt.hour
        df['month']          = df['datetime'].dt.month
        df['caption_length'] = df['Description'].fillna('').apply(len)
        df['time_of_day']    = pd.cut(df['hour'],
                                      bins=[-1, 6, 12, 17, 21, 24],
                                      labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Late night'])

    fb['engagement'] = fb['Reactions'] + fb['Comments'] * 2 + fb['Shares'] * 3
    fb['platform']   = 'Facebook'
    fb = fb.rename(columns={'Post type': 'post_type', 'Reach': 'reach', 'Views': 'views'})

    ig['engagement'] = ig['Likes'] + ig['Comments'] * 2 + ig['Saves'] * 3
    ig['platform']   = 'Instagram'
    ig = ig.rename(columns={'Post type': 'post_type', 'Reach': 'reach', 'Views': 'views'})

    return fb, ig

fb, ig = load_data()

st.sidebar.header('Controls')
platform = st.sidebar.radio('Platform', ['Facebook', 'Instagram', 'Both'])

if platform == 'Facebook':
    data = fb.copy()
elif platform == 'Instagram':
    data = ig.copy()
else:
    data = pd.concat([fb, ig], ignore_index=True)

st.markdown('## Performance Overview')
col1, col2, col3, col4 = st.columns(4)
col1.metric('Total Posts',    f"{len(data):,}")
col2.metric('Total Views',    f"{data['views'].sum():,.0f}")
col3.metric('Total Reach',    f"{data['reach'].sum():,.0f}")
col4.metric('Avg Engagement', f"{data['engagement'].mean():.0f}")

st.markdown('---')

col1, col2 = st.columns(2)

with col1:
    st.markdown('### Engagement by Post Type')
    type_perf = data.groupby('post_type')['engagement'].mean().sort_values(ascending=True)
    fig1 = go.Figure(go.Bar(
        x=type_perf.values,
        y=type_perf.index,
        orientation='h',
        marker_color='#2c5f8a'
    ))
    fig1.update_layout(
        xaxis_title='Avg Engagement Score',
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12, color='#2c2c2c'),
        xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown('### Engagement by Day of Week')
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_perf  = data.groupby('day_of_week')['engagement'].mean().reindex(day_order)
    fig2 = go.Figure(go.Bar(
        x=day_perf.index,
        y=day_perf.values,
        marker_color=['#2c5f8a' if d not in ['Saturday', 'Sunday'] else '#c0392b' for d in day_perf.index]
    ))
    fig2.update_layout(
        yaxis_title='Avg Engagement Score',
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12, color='#2c2c2c'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown('### Best Time to Post')
col1, col2 = st.columns(2)

with col1:
    hour_perf = data.groupby('hour')['engagement'].mean()
    fig3 = go.Figure(go.Scatter(
        x=hour_perf.index,
        y=hour_perf.values,
        mode='lines+markers',
        line=dict(color='#2c5f8a', width=2),
        marker=dict(size=6)
    ))
    fig3.update_layout(
        title='Engagement by Hour of Day',
        xaxis_title='Hour (24h)',
        yaxis_title='Avg Engagement',
        height=300,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12, color='#2c2c2c'),
        xaxis=dict(showgrid=True, gridcolor='#f0f0f0', tickmode='linear', dtick=2),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    tod_perf = data.groupby('time_of_day', observed=True)['engagement'].mean()
    fig4 = go.Figure(go.Bar(
        x=tod_perf.index,
        y=tod_perf.values,
        marker_color='#2c5f8a'
    ))
    fig4.update_layout(
        title='Engagement by Time of Day',
        yaxis_title='Avg Engagement',
        height=300,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12, color='#2c2c2c'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown('---')

st.markdown('### Reach vs Engagement')
fig5 = px.scatter(
    data,
    x='reach',
    y='engagement',
    color='post_type',
    hover_data=['datetime', 'post_type'],
    title='Which posts got the most reach AND engagement?',
    height=400,
)
fig5.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='Arial', size=12, color='#2c2c2c'),
    xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
    yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
)
st.plotly_chart(fig5, use_container_width=True)

st.markdown('---')

st.markdown('## Engagement Predictor')
st.markdown('Enter details about a planned post to predict how much engagement it will get.')

@st.cache_data
def train_model(platform_name):
    if platform_name == 'Facebook':
        df = fb.copy()
    elif platform_name == 'Instagram':
        df = ig.copy()
    else:
        df = pd.concat([fb, ig], ignore_index=True)

    df = df.dropna(subset=['engagement', 'post_type', 'day_of_week', 'hour', 'caption_length'])

    le_type = LabelEncoder()
    le_day  = LabelEncoder()

    df['post_type_enc']   = le_type.fit_transform(df['post_type'])
    df['day_of_week_enc'] = le_day.fit_transform(df['day_of_week'])

    X = df[['post_type_enc', 'day_of_week_enc', 'hour', 'caption_length']]
    y = df['engagement']

    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, y)

    return model, le_type, le_day

model, le_type, le_day = train_model(platform)

col1, col2, col3, col4 = st.columns(4)

with col1:
    post_types    = data['post_type'].unique().tolist()
    selected_type = st.selectbox('Post Type', post_types)

with col2:
    day_options  = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    selected_day = st.selectbox('Day of Week', day_options)

with col3:
    selected_hour = st.slider('Hour of Day', 0, 23, 9)

with col4:
    selected_caption = st.slider('Caption Length (chars)', 0, 2000, 300, step=50)

try:
    type_enc = le_type.transform([selected_type])[0]
    day_enc  = le_day.transform([selected_day])[0]
    X_pred   = np.array([[type_enc, day_enc, selected_hour, selected_caption]])
    pred     = max(0, model.predict(X_pred)[0])

    avg_engagement = data['engagement'].mean()
    diff_pct       = ((pred - avg_engagement) / avg_engagement) * 100

    st.markdown('#### Predicted Engagement')
    col1, col2, col3 = st.columns(3)
    col1.metric('Predicted Engagement Score', f'{pred:.0f}', delta=f'{diff_pct:+.1f}% vs average')
    col2.metric('Average Engagement Score',   f'{avg_engagement:.0f}')
    col3.metric('Best Day to Post',           data.groupby('day_of_week')['engagement'].mean().idxmax())

    if diff_pct > 20:
        st.success(f'This looks like a strong post! Expected to perform {diff_pct:.0f}% above average.')
    elif diff_pct < -20:
        st.warning('This post may underperform. Consider changing the day or time.')
    else:
        st.info('This post is expected to perform close to average.')

except Exception as e:
    st.error(f'Prediction error: {e}')

st.markdown('---')

st.markdown('### Top 10 Posts by Engagement')
top_posts = data.nlargest(10, 'engagement')[['datetime', 'post_type', 'views', 'reach', 'engagement', 'platform']].copy()
top_posts['datetime'] = top_posts['datetime'].dt.strftime('%b %d, %Y %H:%M')
top_posts.columns = ['Published', 'Type', 'Views', 'Reach', 'Engagement Score', 'Platform']
st.dataframe(top_posts, use_container_width=True)
