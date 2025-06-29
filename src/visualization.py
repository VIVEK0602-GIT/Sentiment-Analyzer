import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def plot_gauge(compound):
    color = '#28a745' if compound > 0.05 else '#dc3545' if compound < -0.05 else '#6c757d'
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=compound,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Compound Score"},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [-1, -0.05], 'color': '#dc3545'},
                {'range': [-0.05, 0.05], 'color': '#6c757d'},
                {'range': [0.05, 1], 'color': '#28a745'}
            ]
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_breakdown(scores_or_df):
    if isinstance(scores_or_df, dict):
        scores = scores_or_df
        data = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Score': [scores['pos'], scores['neu'], scores['neg']]
        })
    else:
        df = scores_or_df
        data = df['sentiment'].value_counts().reset_index()
        data.columns = ['Sentiment', 'Count']
    fig = px.bar(data, x='Sentiment', y=data.columns[-1], color='Sentiment',
                 color_discrete_map={'Positive': '#28a745', 'Neutral': '#6c757d', 'Negative': '#dc3545',
                                     'positive': '#28a745', 'neutral': '#6c757d', 'negative': '#dc3545'})
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_word_impact(word_impacts):
    if not word_impacts:
        return go.Figure()
    words = [w[0] for w in word_impacts]
    impacts = [w[1] for w in word_impacts]
    colors = ['#28a745' if w[2] == 'positive' else '#dc3545' if w[2] == 'negative' else '#6c757d' for w in word_impacts]
    fig = go.Figure(go.Bar(
        x=impacts,
        y=words,
        orientation='h',
        marker_color=colors
    ))
    fig.update_layout(title="Word Impact Analysis", xaxis_title="Impact", yaxis_title="Word",
                      height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig 