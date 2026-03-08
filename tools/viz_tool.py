"""
Visualization Tool - Altair Version
Creates interactive HTML charts from data
Drop-in replacement for matplotlib version
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Literal
import pandas as pd
import altair as alt
from langchain_core.tools import tool

# Import config
from config import BASE_DIR

# Output directory for charts
CHARTS_DIR = BASE_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)


@tool
def create_chart(
    data: str, 
    chart_type: Literal["bar", "line", "pie", "scatter", "histogram", "area"] = "bar",
    title: str = "Chart",
    x_label: str = "",
    y_label: str = "",
    filename: str = ""
) -> str:
    """
    Create an interactive HTML chart from data.

    This tool takes data (typically from SQL query results) and creates 
    interactive visualizations using Altair. Charts include zoom, pan, and hover tooltips.

    Args:
        data: JSON string containing the data. Should be a list of dictionaries.
            Example: '[{"month": "Jan", "sales": 1000}, {"month": "Feb", "sales": 1500}]'
            Or dict format: '{"Jan": 1000, "Feb": 1500}'
            - Column names may contain special characters; escape any colons with '\\:'
            - Data types must be inferred or specified: 
                * Quantitative (numeric) -> Q
                * Nominal (categorical) -> N
                * Ordinal (ordered categorical) -> O
                * Temporal (dates/times) -> T
                * Geospatial (maps) -> G

        chart_type: Type of chart to create. Options:
            - "bar": Vertical bar chart (good for comparing categories)
            - "line": Line chart (good for trends over time)
            - "pie": Pie chart (good for showing proportions)
            - "scatter": Scatter plot (good for relationships)
            - "histogram": Histogram (good for distributions)
            - "area": Area chart (good for cumulative trends)

        title: Title for the chart
        x_label: Label for x-axis (optional)
        y_label: Label for y-axis (optional)
        filename: Custom filename (optional, auto-generated if not provided)

    Returns:
        Path to the saved HTML chart file with instructions to open in browser

    Notes:
        - Always map each column to the correct Altair type (Q, O, N, T, G)
        - Escape column names with colons using '\\:'
        - If data is a dict, convert it into a list of dicts for Altair

    Examples:
        create_chart(
            data='[{"customer": "Acme Corp", "revenue": 50000}, {"customer": "TechCo", "revenue": 75000}]',
            chart_type="bar",
            title="Top Customers by Revenue",
            x_label="Customer",
            y_label="Revenue ($)"
        )
    """
    
    try:
        # Parse data
        data_parsed = json.loads(data)
        
        # Convert to pandas DataFrame
        if isinstance(data_parsed, list):
            df = pd.DataFrame(data_parsed)
        elif isinstance(data_parsed, dict):
            df = pd.DataFrame([data_parsed]).T.reset_index()
            df.columns = ['category', 'value']
        else:
            return f"Error: Data must be a JSON list or dict, got {type(data_parsed)}"
        
        if df.empty:
            return "Error: No data provided"
        
        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{chart_type}_{timestamp}.html"
        
        if not filename.endswith('.html'):
            filename = filename.replace('.png', '.html')  # Replace .png if present
            if not filename.endswith('.html'):
                filename += '.html'
        
        filepath = CHARTS_DIR / filename
        
        # Determine column names
        if len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = df.columns[1]
        else:
            x_col = 'index'
            y_col = df.columns[0]
            df[x_col] = df.index
        
        # Set labels
        x_axis_label = x_label if x_label else x_col
        y_axis_label = y_label if y_label else y_col
        
        # Create base chart
        base = alt.Chart(df).properties(
            title=title,
            width=700,
            height=400
        )
        
        # Create chart based on type
        if chart_type == "bar":
            chart = base.mark_bar(color='steelblue').encode(
                x=alt.X(f"{x_col}:O", title=x_axis_label, sort=None),
                y=alt.Y(f"{y_col}:Q", title=y_axis_label),
                tooltip=[x_col, y_col]
            )
        
        elif chart_type == "line":
            chart = base.mark_line(point=True, color='steelblue').encode(
                x=alt.X(f"{x_col}:O", title=x_axis_label, sort=None),
                y=alt.Y(f"{y_col}:Q", title=y_axis_label),
                tooltip=[x_col, y_col]
            )
        
        elif chart_type == "pie":
            chart = base.mark_arc().encode(
                theta=alt.Theta(f"{y_col}:Q"),
                color=alt.Color(f"{x_col}:N", legend=alt.Legend(title=x_axis_label)),
                tooltip=[x_col, y_col]
            )
        
        elif chart_type == "scatter":
            if len(df.columns) < 2:
                return "Error: Scatter plot requires at least 2 columns of data"
            
            chart = base.mark_circle(size=100, color='steelblue', opacity=0.6).encode(
                x=alt.X(f"{x_col}:Q", title=x_axis_label),
                y=alt.Y(f"{y_col}:Q", title=y_axis_label),
                tooltip=[x_col, y_col]
            )
        
        elif chart_type == "histogram":
            # Use the last column for histogram
            data_col = df.columns[-1]
            chart = base.mark_bar(color='steelblue').encode(
                x=alt.X(f"{data_col}:Q", bin=alt.Bin(maxbins=20), title=x_label or data_col),
                y=alt.Y('count()', title=y_label or "Frequency"),
                tooltip=['count()']
            )
        
        elif chart_type == "area":
            chart = base.mark_area(color='steelblue', opacity=0.7).encode(
                x=alt.X(f"{x_col}:O", title=x_axis_label, sort=None),
                y=alt.Y(f"{y_col}:Q", title=y_axis_label),
                tooltip=[x_col, y_col]
            )
        
        else:
            return f"Error: Unknown chart type '{chart_type}'"
        
        # Save chart
        chart.save(str(filepath))
        
        return f"✓ Interactive chart created: {filepath}\n\nOpen {filepath.name} in your browser to view. Chart includes zoom, pan, and hover tooltips."
    
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON data. {str(e)}\nData received: {data[:100]}..."
    
    except Exception as e:
        return f"Error creating chart: {str(e)}"


@tool
def create_multi_series_chart(
    data: str,
    chart_type: Literal["bar", "line"] = "line",
    title: str = "Multi-Series Chart",
    x_label: str = "",
    y_label: str = "",
    filename: str = ""
) -> str:
    """
    Create an interactive chart with multiple data series.
    
    Useful for comparing multiple metrics over time or across categories.
    
    Args:
        data: JSON string with multiple series. Format:
              '[{"month": "Jan", "sales": 1000, "costs": 800}, 
                {"month": "Feb", "sales": 1500, "costs": 900}]'
              
              First column is x-axis, remaining columns are separate series.
        
        chart_type: "line" for multi-line chart, "bar" for grouped bar chart
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        filename: Custom filename (optional)
    
    Returns:
        Path to saved interactive HTML chart
    
    Example:
        create_multi_series_chart(
            data='[{"month": "Jan", "revenue": 10000, "costs": 7000, "profit": 3000},
                   {"month": "Feb", "revenue": 12000, "costs": 7500, "profit": 4500}]',
            chart_type="line",
            title="Revenue, Costs, and Profit Trends",
            x_label="Month",
            y_label="Amount ($)"
        )
    """
    
    try:
        # Parse data
        data_parsed = json.loads(data)
        df = pd.DataFrame(data_parsed)
        
        if df.empty or len(df.columns) < 2:
            return "Error: Need at least 2 columns (x-axis + at least 1 data series)"
        
        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_{chart_type}_{timestamp}.html"
        
        if not filename.endswith('.html'):
            filename = filename.replace('.png', '.html')
            if not filename.endswith('.html'):
                filename += '.html'
        
        filepath = CHARTS_DIR / filename
        
        # Get column names
        x_col = df.columns[0]
        
        # Reshape to long format for Altair
        df_long = df.melt(
            id_vars=[x_col],
            value_vars=list(df.columns[1:]),
            var_name='series',
            value_name='value'
        )
        
        # Create base chart
        base = alt.Chart(df_long).properties(
            title=title,
            width=800,
            height=400
        )
        
        if chart_type == "line":
            chart = base.mark_line(point=True).encode(
                x=alt.X(f"{x_col}:O", title=x_label or x_col),
                y=alt.Y("value:Q", title=y_label or "Value"),
                color=alt.Color("series:N", title="Metric"),
                tooltip=[x_col, "series", "value"]
            )
        
        elif chart_type == "bar":
            chart = base.mark_bar().encode(
                x=alt.X(f"{x_col}:N", title=x_label or x_col),
                y=alt.Y("value:Q", title=y_label or "Value"),
                color=alt.Color("series:N", title="Metric"),
                xOffset="series:N",
                tooltip=[x_col, "series", "value"]
            )
        
        else:
            return f"Error: Unsupported chart type '{chart_type}' for multi-series"
        
        # Save chart
        chart.save(str(filepath))
        
        series_count = len(df.columns) - 1
        return f"✓ Multi-series chart created: {filepath}\n\nComparing {series_count} metrics. Open {filepath.name} in browser to view."
    
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON data. {str(e)}"
    
    except Exception as e:
        return f"Error creating multi-series chart: {str(e)}"
