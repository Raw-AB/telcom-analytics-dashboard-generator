
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChartConfig:
    """Data class for chart configuration"""
    chart_type: str
    title: str
    x_label: str = ""
    y_label: str = ""
    show_legend: bool = True
    background_color: str = "rgba(255, 255, 255, 1)"
    width: int = 800
    height: int = 400

class ChartGenerator:
    @staticmethod
    def generate_chart(data: Dict, chart_type: str = 'bar', title: str = None) -> str:
        """
        Generate QuickChart URL from data
        
        Args:
            data: Dictionary with 'labels' and 'datasets'
            chart_type: Type of chart ('bar', 'line', 'pie', etc.)
            title: Chart title
            
        Returns:
            QuickChart URL
        """
        config = {
            "type": chart_type,
            "data": data,
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": bool(title),
                        "text": title or ""
                    }
                }
            }
        }
        
        encoded_config = json.dumps(config)
        return f"https://quickchart.io/chart?c={encoded_config}&width=800&height=400"
    
    @staticmethod
    def prepare_chart_data(df: pd.DataFrame, x_col: str, y_cols: List[str]) -> Dict:
        """
        Convert SQL results to QuickChart format
        
        Args:
            df: Pandas DataFrame from SQL query
            x_col: Column name for x-axis
            y_cols: List of columns for y-axis
            
        Returns:
            Dictionary formatted for QuickChart
        """
        chart_data = {
            "labels": df[x_col].tolist(),
            "datasets": []
        }
        
        colors = [
            'rgb(54, 162, 235)',
            'rgb(255, 99, 132)',
            'rgb(75, 192, 192)',
            'rgb(255, 159, 64)',
            'rgb(153, 102, 255)'
        ]
        
        for i, col in enumerate(y_cols):
            chart_data["datasets"].append({
                "label": col,
                "data": df[col].tolist(),
                "backgroundColor": colors[i % len(colors)]
            })
            
        return chart_data
    
    @staticmethod
    def detect_chart_type(question: str) -> str:
        """
        Infer chart type based on question keywords
        
        Args:
            question: Natural language question
            
        Returns:
            Recommended chart type
        """
        question = question.lower()
        if "trend" in question or "over time" in question:
            return "line"
        elif "distribution" in question or "percentage" in question:
            return "pie"
        elif "compare" in question or "vs" in question:
            return "bar"
        elif "top" in question or "most" in question:
            return "horizontalBar"
        return "bar"

    @staticmethod
    def render_chart(question: str, df: pd.DataFrame) -> Optional[str]:
        """
        Complete chart rendering pipeline
        
        Args:
            question: Original question
            df: DataFrame with results
            
        Returns:
            Chart URL or None if failed
        """
        try:
            if df.empty:
                return None
                
            # Auto-detect columns
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            
            x_col = non_numeric_cols[0] if non_numeric_cols else "index"
            if x_col == "index":
                df = df.reset_index()
            y_cols = numeric_cols[:5]  # Limit to 5 series
            
            chart_data = ChartGenerator.prepare_chart_data(df, x_col, y_cols)
            chart_type = ChartGenerator.detect_chart_type(question)
            return ChartGenerator.generate_chart(chart_data, chart_type, question)
            
        except Exception as e:
            print(f"Chart rendering error: {e}")
            return None