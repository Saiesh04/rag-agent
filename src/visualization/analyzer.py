"""
Data visualization and analysis module for RAG Q&A System
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

logger = logging.getLogger(__name__)

class DataVisualizer:
    """Handles data visualization and automated analysis"""
    
    def __init__(self, theme: str = "plotly_white"):
        self.theme = theme
        plt.style.use('seaborn-v0_8' if 'seaborn' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def analyze_dataframe(self, df: pd.DataFrame, table_name: str = "Data") -> Dict[str, Any]:
        """Perform automated analysis of a DataFrame"""
        try:
            analysis = {
                "basic_info": self._get_basic_info(df),
                "column_analysis": self._analyze_columns(df),
                "missing_data": self._analyze_missing_data(df),
                "correlations": self._analyze_correlations(df),
                "insights": self._generate_insights(df, table_name)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing DataFrame: {str(e)}")
            return {"error": str(e)}
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the DataFrame"""
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "null_counts": df.isnull().sum().to_dict()
        }
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual columns"""
        column_analysis = {}
        
        for col in df.columns:
            col_data = df[col]
            analysis = {
                "type": str(col_data.dtype),
                "unique_count": col_data.nunique(),
                "null_count": col_data.isnull().sum(),
                "null_percentage": (col_data.isnull().sum() / len(df)) * 100
            }
            
            if pd.api.types.is_numeric_dtype(col_data):
                analysis.update({
                    "mean": col_data.mean(),
                    "median": col_data.median(),
                    "std": col_data.std(),
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "quartiles": col_data.quantile([0.25, 0.5, 0.75]).to_dict()
                })
            elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
                value_counts = col_data.value_counts().head(10)
                analysis.update({
                    "top_values": value_counts.to_dict(),
                    "avg_length": col_data.astype(str).str.len().mean() if not col_data.empty else 0
                })
            
            column_analysis[col] = analysis
        
        return column_analysis
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df)) * 100
        
        return {
            "total_missing": missing_data.sum(),
            "columns_with_missing": missing_data[missing_data > 0].to_dict(),
            "missing_percentages": missing_percentage[missing_percentage > 0].to_dict(),
            "complete_rows": len(df) - df.isnull().any(axis=1).sum()
        }
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {"message": "Not enough numeric columns for correlation analysis"}
        
        corr_matrix = df[numeric_cols].corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": corr_value
                    })
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_correlations
        }
    
    def _generate_insights(self, df: pd.DataFrame, table_name: str) -> List[str]:
        """Generate automated insights from the data"""
        insights = []
        
        # Basic insights
        insights.append(f"Dataset '{table_name}' contains {len(df):,} rows and {len(df.columns)} columns.")
        
        # Missing data insights
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            insights.append(f"Missing data found in {len(missing_cols)} columns: {', '.join(missing_cols[:3])}{'...' if len(missing_cols) > 3 else ''}.")
        
        # Numeric column insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights.append(f"Found {len(numeric_cols)} numeric columns for quantitative analysis.")
            
            # Check for outliers in numeric columns
            for col in numeric_cols[:3]:  # Check first 3 numeric columns
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                if len(outliers) > 0:
                    insights.append(f"Column '{col}' has {len(outliers)} potential outliers ({len(outliers)/len(df)*100:.1f}% of data).")
        
        # Categorical column insights
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns
        if len(categorical_cols) > 0:
            insights.append(f"Found {len(categorical_cols)} categorical columns for qualitative analysis.")
            
            for col in categorical_cols[:2]:  # Check first 2 categorical columns
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.9:
                    insights.append(f"Column '{col}' has high cardinality ({df[col].nunique()} unique values) - possibly an identifier.")
                elif unique_ratio < 0.1:
                    insights.append(f"Column '{col}' has low cardinality ({df[col].nunique()} unique values) - good for grouping.")
        
        return insights
    
    def create_distribution_plot(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Create a distribution plot for a column"""
        try:
            if pd.api.types.is_numeric_dtype(df[column]):
                # Histogram for numeric data
                fig = px.histogram(
                    df, 
                    x=column, 
                    nbins=30,
                    title=f"Distribution of {column}",
                    template=self.theme
                )
                fig.add_vline(x=df[column].mean(), line_dash="dash", line_color="red", 
                             annotation_text=f"Mean: {df[column].mean():.2f}")
                
            else:
                # Bar chart for categorical data
                value_counts = df[column].value_counts().head(20)
                fig = px.bar(
                    x=value_counts.index, 
                    y=value_counts.values,
                    title=f"Distribution of {column}",
                    template=self.theme
                )
                fig.update_xaxis(title=column)
                fig.update_yaxis(title="Count")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating distribution plot: {str(e)}")
            return go.Figure()
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create a correlation heatmap"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) < 2:
                fig = go.Figure()
                fig.add_annotation(text="Not enough numeric columns for correlation analysis",
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            
            corr_matrix = numeric_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                showscale=True
            ))
            
            fig.update_layout(
                title="Correlation Matrix",
                template=self.theme,
                width=600,
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            return go.Figure()
    
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None) -> go.Figure:
        """Create a scatter plot"""
        try:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"{y_col} vs {x_col}",
                template=self.theme,
                hover_data=df.columns.tolist()[:5]  # Show first 5 columns on hover
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            return go.Figure()
    
    def create_box_plot(self, df: pd.DataFrame, column: str, group_by: str = None) -> go.Figure:
        """Create a box plot"""
        try:
            if group_by and group_by in df.columns:
                fig = px.box(
                    df,
                    x=group_by,
                    y=column,
                    title=f"Box Plot of {column} by {group_by}",
                    template=self.theme
                )
            else:
                fig = px.box(
                    df,
                    y=column,
                    title=f"Box Plot of {column}",
                    template=self.theme
                )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating box plot: {str(e)}")
            return go.Figure()
    
    def create_time_series_plot(self, df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
        """Create a time series plot"""
        try:
            # Ensure date column is datetime
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            df_copy = df_copy.sort_values(date_col)
            
            fig = px.line(
                df_copy,
                x=date_col,
                y=value_col,
                title=f"Time Series: {value_col} over {date_col}",
                template=self.theme
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {str(e)}")
            return go.Figure()
    
    def create_summary_dashboard(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Create a comprehensive dashboard with multiple visualizations"""
        try:
            dashboard = {
                "table_name": table_name,
                "plots": {},
                "analysis": self.analyze_dataframe(df, table_name)
            }
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'string']).columns
            
            # Create distribution plots for first few numeric columns
            for col in numeric_cols[:3]:
                dashboard["plots"][f"dist_{col}"] = self.create_distribution_plot(df, col)
            
            # Create distribution plots for first few categorical columns
            for col in categorical_cols[:2]:
                if df[col].nunique() <= 20:  # Only if not too many categories
                    dashboard["plots"][f"dist_{col}"] = self.create_distribution_plot(df, col)
            
            # Create correlation heatmap if enough numeric columns
            if len(numeric_cols) >= 2:
                dashboard["plots"]["correlation"] = self.create_correlation_heatmap(df)
            
            # Create scatter plot if we have at least 2 numeric columns
            if len(numeric_cols) >= 2:
                dashboard["plots"]["scatter"] = self.create_scatter_plot(
                    df, numeric_cols[0], numeric_cols[1]
                )
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating summary dashboard: {str(e)}")
            return {"error": str(e)}
    
    def generate_insights_text(self, analysis: Dict[str, Any]) -> str:
        """Generate a text summary of insights"""
        try:
            insights = analysis.get("insights", [])
            if not insights:
                return "No specific insights generated for this dataset."
            
            return "\n".join([f"• {insight}" for insight in insights])
            
        except Exception as e:
            logger.error(f"Error generating insights text: {str(e)}")
            return "Error generating insights."