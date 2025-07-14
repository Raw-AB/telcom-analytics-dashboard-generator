from azure_sql import run_sql
from sqlcoder_infer import SQLGenerator
from chart_utils import ChartGenerator
import streamlit as st
import pandas as pd
import time
from datetime import datetime
class TelcoAnalytics:
    def __init__(self):
        self.chart_gen = ChartGenerator()
        self.sql_gen = SQLGenerator("./models/sqlcoder-7b.Q4_K_M.gguf")

        # Use Streamlit session state to persist metrics
        if "session_metrics" not in st.session_state:
            st.session_state.session_metrics = {
                "start_time": datetime.now(),
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "query_times": [],
                "response_times": [],
                "common_errors": {},  # e.g., {"Missing JOIN": 5}
                "query_types": {},    # e.g., {"SELECT": 12}
                "accuracy_scores": []

                
            }

    def analyze(self, question: str):
        """End-to-end analysis pipeline with enhanced tracking"""
        st.session_state.session_metrics["total_queries"] += 1
        start_time = time.time()

        try:
            sql = self.sql_gen.generate_sql(question)
            if not sql or not isinstance(sql, str):
                raise ValueError("Invalid SQL generated")

            st.session_state.last_sql = sql
            gen_time = time.time() - start_time
            st.session_state.session_metrics["query_times"].append(gen_time)

            # Run SQL
            try:
                df = run_sql(sql)
                if df.empty:
                    st.session_state.session_metrics["failed_queries"] += 1
                    return None, "No results found", None

                try:
                    chart_url = self.chart_gen.render_chart(question, df)
                    st.session_state.session_metrics["successful_queries"] += 1
                    return df, sql, chart_url

                except Exception as chart_error:
                    st.session_state.session_metrics["failed_queries"] += 1
                    return df, sql, f"Visualization Error: {str(chart_error)}"

            except Exception as sql_error:
                st.session_state.session_metrics["failed_queries"] += 1
                return None, f"Execution Error: {str(sql_error)}", None

        except Exception as gen_error:
            st.session_state.session_metrics["failed_queries"] += 1
            return None, f"Generation Error: {str(gen_error)}", None

    def get_session_stats(self):
        metrics = st.session_state.session_metrics
        avg_time = (
            sum(metrics["query_times"]) / len(metrics["query_times"])
            if metrics["query_times"] else 0
        )

        query_success_log = self.sql_gen.metrics.get("query_success_log", [])
        avg_success_rate = (
            sum(query_success_log) / len(query_success_log) * 100
            if query_success_log else 0
        )

        return {
            "Session Duration": str(datetime.now() - metrics["start_time"]).split('.')[0],
            "Total Queries": metrics["total_queries"],
            "Success Rate": f"{avg_success_rate:.1f}%",
            "Avg Generation Time": f"{avg_time:.2f}s",
            "Model Metrics": {
                "top_error_types": self.sql_gen.metrics.get("top_error_types", []),
                "common_query_types": self.sql_gen.metrics.get("common_query_types", [])
            }
        }


def display_sample_questions():
    """Interactive sample question panel with guaranteed unique keys"""
    with st.sidebar:
        st.header("💡 Sample Questions")
        
        categories = {
            "📊 Product Analysis": [
                ("prod1", "What are the top 5 most expensive products?"),
                ("prod2", "Show products by category and average price")
            ],
            "📈 Usage Trends": [
                ("usage1", "Show monthly data usage trends for the past year"),
                ("usage2", "Show average data usage by device type")
            ],
            "📉 Churn Analysis": [
                ("churn1", "What are the top churn reasons last year?")
            ],
            "👥 Subscriber Insights": [
                ("sub1", "Show subscriber count by state?"),
                ("sub2", "Show subscriber count by gender?")
            ]
        }
        
        for category, questions in categories.items():
            with st.expander(category):
                for qid, q in questions:
                    button_key = f"sample_{category[:3]}_{qid}"
                    if st.button(q, key=button_key):
                        st.session_state.question = q
                        st.session_state.run_analysis = True
def display_debug_info(analytics):
    """Debug information for developers"""
    with st.expander("Developer Tools"):
        tab1, tab2, tab3, tab4 = st.tabs(["Session Stats", "SQL Debug", "Raw Data", "Performance Metrics"])
        
        with tab1:
            stats = analytics.get_session_stats()
            st.subheader("Session Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Queries", stats["Total Queries"])
                st.metric("Session Duration", stats["Session Duration"])
                
            with col2:
                st.metric("Success Rate", stats["Success Rate"])
                st.metric("Avg Generation Time", stats["Avg Generation Time"])
            
            st.subheader("Model Metrics")
            model_metrics = stats["Model Metrics"]
            if isinstance(model_metrics, dict):
                st.write("Top Error Types:")
                for error, count in model_metrics.get("top_error_types", []):
                    st.text(f"{error}: {count}")
                
                st.write("Common Query Types:")
                for qtype, count in model_metrics.get("common_query_types", []):
                    st.text(f"{qtype}: {count}")
            else:
                st.text(model_metrics)
        
        with tab2:
            if 'last_sql' in st.session_state:
                st.subheader("Last Generated SQL")
                st.code(st.session_state.last_sql, language="sql")
            else:
                st.warning("No SQL generated yet")
        
        with tab3:
            st.subheader("Raw Metrics Data")
            st.json(analytics.sql_gen.metrics)
            
        with tab4:
            st.subheader("SQL Generation Performance")
            try:
                metrics = analytics.sql_gen.get_metrics()

                st.metric("Total Queries Processed", metrics["total_queries"])
                st.metric("Success Rate", metrics["success_rate"])
                st.metric("Average Generation Time", metrics["avg_generation_time"])
                st.metric("Average Query Success Rate", metrics.get("avg_query_success_rate", "N/A"))
                st.subheader("Error Analysis")
                if metrics["common_query_types"]:
                    df_qtypes = pd.DataFrame(metrics["common_query_types"], columns=["Query Type", "Count"])
                    if not df_qtypes.empty:
                        st.dataframe(df_qtypes, hide_index=True)
                    else:
                        st.info("No query type metrics yet.")

              
                st.subheader("Query Type Distribution")
                if metrics["common_query_types"]:
                    st.dataframe(
                        pd.DataFrame(metrics["common_query_types"], columns=["Query Type", "Count"]),
                        hide_index=True
                    )
                else:
                    st.info("No query type metrics yet.")
            except Exception as e:
                st.error(f"Error loading performance metrics: {e}")

            
def main():
    # Configure page
    st.set_page_config(
        page_title="Telco Analytics Dashboard",
        layout="wide",
        page_icon="📊"
    )
    
    # Initialize session state
    if 'question' not in st.session_state:
        st.session_state.question = ""
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False
    
    # Initialize analytics engine
    if 'analytics' not in st.session_state:
        st.session_state.analytics = TelcoAnalytics()
    analytics = st.session_state.analytics

    
    # Main header
    st.title("📞 Telco Data Analytics")
    st.markdown("Ask natural language questions about your telecom data")
    
    # Sidebar with sample questions
    display_sample_questions()
    
    # Main question input
    question = st.text_area(
        "Ask your question:",
        value=st.session_state.question,
        height=100,
        key="question_input"
    )
    
    # Update question in session state when changed
    if question != st.session_state.question:
        st.session_state.question = question
    
    # Process question when submitted
    if st.button("Analyze"):
        st.session_state.run_analysis = True
    
    if st.session_state.run_analysis and st.session_state.question:
        with st.spinner("🔍 Analyzing your question..."):
            df, sql, chart = analytics.analyze(st.session_state.question)
        st.session_state.run_analysis = False
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["Visualization", "Data", "SQL"])
        
        with tab1:
            if chart and chart.startswith("http"):
                st.image(chart, use_column_width=True)
            elif chart:
                st.error(chart)
            else:
                st.warning("No visualization generated")
        
        with tab2:
            if df is not None:
                st.dataframe(df, height=600, use_container_width=True)
            else:
                st.error("No data available")
        
        with tab3:
            if sql:
                st.code(sql, language="sql")
            else:
                st.error("No SQL generated")
    
    # Debug information
    display_debug_info(analytics)
    
    # Footer
    st.markdown("---")
    st.caption("Telco Analytics Dashboard v1.0 | Powered by SQLCoder-7B")

if __name__ == "__main__":
    main()