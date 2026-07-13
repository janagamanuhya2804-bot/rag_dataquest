import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any

# Dark theme layout parameters
DARK_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font_color": "#E0E0E0",
    "font_family": "Inter, Roboto, sans-serif",
    "margin": dict(l=20, r=20, t=40, b=20),
    "xaxis": {"gridcolor": "#2D2D2D", "zeroline": False},
    "yaxis": {"gridcolor": "#2D2D2D", "zeroline": False}
}

def generate_category_pie_chart(category_data: Dict[str, float]) -> Any:
    """Generate a pie/donut chart for category breakdown."""
    if not category_data:
        # Return empty placeholder figure
        fig = go.Figure()
        fig.add_annotation(text="No transactions recorded yet.", showarrow=False, font=dict(size=14, color="#888888"))
        fig.update_layout(**DARK_LAYOUT)
        return fig
        
    df = pd.DataFrame(list(category_data.items()), columns=['Category', 'Amount'])
    
    # Custom color palette: Sleek, premium cool tones
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    
    fig = px.pie(
        df, 
        values='Amount', 
        names='Category', 
        hole=0.5,
        color_discrete_sequence=colors,
        title="Spending by Category"
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(**DARK_LAYOUT)
    fig.update_layout(showlegend=False)
    return fig

def generate_daily_trend_chart(daily_data: Dict[str, float]) -> Any:
    """Generate a line/area chart showing daily spending trends."""
    if not daily_data:
        fig = go.Figure()
        fig.add_annotation(text="No spending history available.", showarrow=False, font=dict(size=14, color="#888888"))
        fig.update_layout(**DARK_LAYOUT)
        return fig
        
    # Sort data by date
    sorted_dates = sorted(daily_data.keys())
    sorted_amounts = [daily_data[d] for d in sorted_dates]
    
    df = pd.DataFrame({
        'Date': pd.to_datetime(sorted_dates),
        'Amount': sorted_amounts
    })
    
    # Calculate cumulative spend
    df['Cumulative Spend'] = df['Amount'].cumsum()
    
    fig = go.Figure()
    
    # Cumulative area trace
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Cumulative Spend'],
        fill='tozeroy',
        name='Cumulative Spend (₹)',
        line=dict(color='#00CC96', width=2),
        fillcolor='rgba(0, 204, 150, 0.15)'
    ))
    
    # Daily bar trace
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Amount'],
        name='Daily Spend (₹)',
        marker_color='rgba(99, 110, 250, 0.6)'
    ))
    
    fig.update_layout(
        title="Spending Trend over Time",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **DARK_LAYOUT
    )
    return fig

def generate_comparison_bar_chart(comp_data: Dict[str, Any]) -> Any:
    """Generate a bar chart comparing current vs previous month spending."""
    fig = go.Figure()
    
    categories = [comp_data.get("last_month_name", "Last Month"), comp_data.get("current_month_name", "This Month")]
    amounts = [comp_data.get("last_month_total", 0.0), comp_data.get("current_month_total", 0.0)]
    
    fig.add_trace(go.Bar(
        x=categories,
        y=amounts,
        marker_color=['#636EFA', '#EF553B'],
        text=[f"₹{amt:,.2f}" for amt in amounts],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Month-over-Month Comparison",
        yaxis_title="Total Spend (₹)",
        **DARK_LAYOUT
    )
    return fig
