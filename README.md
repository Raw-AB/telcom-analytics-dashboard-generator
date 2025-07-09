# telcom-analytics-dashboard-generator
from NL to SQL to dashboard using LLMs
# System Architecture
User (NL-Query) → Streamlit UI
                →
      SQLCoder (LLM Inference)
                →
          SQL Query Generator
                →
        Azure SQL Database
                →
         Pandas DataFrame
                →
     Chart Generator (QuickChart)
                →
       Final Insight + Chart 

