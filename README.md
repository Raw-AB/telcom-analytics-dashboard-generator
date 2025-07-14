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

<img width="3754" height="1881" alt="Screenshot 2025-06-11 144943" src="https://github.com/user-attachments/assets/70cd03e4-c0af-4293-a745-98a6f20ab086" />
