
from accelerate import init_empty_weights, infer_auto_device_map
from llama_cpp import Llama
import re
from time import time
from collections import defaultdict
import numpy as np


SCHEMA = """
1. [dbo].[dim_sales_channel]
   - Primary Key: sales_channel_key
   - Columns:
     - channel_name
     - channel_type
     - description
     - status

2. [dbo].[dim_device]
   - Primary Key: device_key
   - Columns:
     - device_type
     - brand
     - model
     - specifications
     - purchase_date
     - status

3. [dbo].[dim_product]
   - Primary Key: product_key
   - Columns:
     - product_id
     - product_name
     - category
     - description
     - price
     - launch_date
     - discontinue_date
     - status

4. [dbo].[dim_network]
   - Primary Key: network_key
   - Columns:
     - network_id
     - network_type
     - provider
     - coverage_area
     - performance_metrics
     - launch_date
     - status

5. [dbo].[dim_geography]
   - Primary Key: geography_key
   - Columns:
     - geography_id
     - country
     - state
     - city
     - region
     - postal_code
     - latitude
     - longitude

6. [dbo].[dim_time]
   - Primary Key: time_key
   - Columns:
     - date
     - day
     - month
     - month_name
     - quarter
     - year
     - week
     - day_of_week
     - is_weekend
     - fiscal_period
     - holiday_flag

7. [dbo].[dim_subscriber]
   - Primary Key: subscriber_key
   - Columns:
     - subscriber_id
     - first_name
     - last_name
     - email
     - phone_number
     - subscription_date
     - status
     - gender
     - date_of_birth
     - address
     - city
     - state
     - postal_code
     - country
   - Links to: sales_channel, device, product, network, geography, time

8. [dbo].[fact_usage]
   - Primary Key: usage_key
   - Columns:
     - data_consumed_mb
     - call_minutes
     - sms_count
     - roaming_minutes
     - roaming_data_mb
     - international_calls
     - usage_date
     - usage_time
     - total_usage_cost
   - Links to: subscriber, time, product, device, network, geography, sales_channel

9. [dbo].[fact_billing]
   - Primary Key: billing_key
   - Columns:
     - invoice_number
     - billing_cycle
     - total_charges
     - discounts
     - taxes
     - total_due
     - due_date
     - paid_date
     - status
     - payment_method
     - notes
   - Links to: subscriber, time

10. [dbo].[fact_payment]
    - Primary Key: payment_key
    - Columns:
      - payment_id
      - invoice_key
      - payment_amount
      - payment_method
      - payment_status
      - transaction_reference
      - payment_notes
    - Links to: subscriber, time

11. [dbo].[fact_churn]
    - Primary Key: churn_key
    - Columns:
      - churn_date
      - churn_reason
      - churn_category
      - status
      - feedback
      - last_payment_date
      - lifetime_value
      - contracts_cancelled
    - Links to: subscriber, time
"""

FEW_SHOTS_EXAMPLE = """
# Example 1: Top Products (Matches dim_product)
Question: What are the top 5 most expensive products?

SQL:
SELECT TOP 5
    [product_name],
    MAX([price]) AS max_price
FROM 
    [dbo].[dim_product]
GROUP BY
    [product_name]
ORDER BY 
    max_price DESC;


# Example 2: Usage Analysis (Matches fact_usage)
Question: Show data usage by device model

SQL:
SELECT 
    dd.[model] AS [device_model],
    SUM(du.[data_consumed_mb]) AS [total_data_mb]
FROM 
    [dbo].[fact_usage] du
JOIN 
    [dbo].[dim_device] dd ON du.[device_key] = dd.[device_key]
GROUP BY 
    dd.[model]
ORDER BY 
    [total_data_mb] DESC;


# Example 3: Monthly Revenue (Matches fact_billing)
Question: Show monthly revenue trends last year

SQL:
SELECT 
    dt.[month_name] AS [month], 
    SUM(fb.[total_due]) AS [revenue] 
FROM 
    [dbo].[fact_billing] fb 
JOIN 
    [dbo].[dim_time] dt ON fb.[time_key] = dt.[time_key] 
WHERE 
    YEAR(dt.[date]) = YEAR(DATEADD(year, -1, GETDATE()))  
GROUP 
    BY dt.[month_name] 
ORDER BY 
    MIN(dt.[date]);

# Example 4: Subscriber Geography (Matches dim_subscriber)
Question: Show subscriber count by state

SQL:
SELECT 
    [state],
    COUNT([subscriber_key]) AS [subscriber_count]
FROM 
    [dbo].[dim_subscriber]
GROUP BY 
    [state]
ORDER BY 
    [subscriber_count] DESC;

# Example 5: Churn Analysis (Matches fact_churn)
Question: What are the top churn reasons last year?

SQL:
SELECT TOP 5
    [churn_reason],
    COUNT([churn_key]) AS [churn_count]
FROM 
    [dbo].[fact_churn] fc
JOIN 
    [dbo].[dim_time] dt ON fc.[time_key] = dt.[time_key]
WHERE 
    dt.[year] = YEAR(DATEADD(year, -1, GETDATE()))
GROUP BY 
    [churn_reason]
ORDER BY 
    [churn_count] DESC;

"""

class SchemaParser:
    def __init__(self, schema_text):
        self.schema = {
            'tables': {},
            'relationships': []
        }
        self.parse_schema(schema_text)
    
    def parse_schema(self, schema_text):
        current_table = None
        for line in schema_text.splitlines():
            line = line.strip()
            
            # Table detection
            if line.startswith(tuple(f"{i}." for i in range(1, 12))):
                match = re.match(r"\d+\. \[dbo\]\.\[([^\]]+)\]", line)
                if match:
                    current_table = match.group(1)
                    self.schema['tables'][current_table] = {
                        'columns': set(),
                        'primary_key': None,
                        'foreign_keys': []
                    }
            
            # Primary key detection
            elif "Primary Key:" in line and current_table:
                pk = line.split("Primary Key:")[1].strip()
                self.schema['tables'][current_table]['primary_key'] = pk
            
            # Column detection
            elif "-" in line and current_table and "Columns:" not in line:
                col_parts = line.split(":")
                if len(col_parts) >= 1:
                    col_name = col_parts[0].strip("- ").strip()
                    if col_name:
                        self.schema['tables'][current_table]['columns'].add(col_name)
            
            # Relationship detection
            elif "Links to:" in line and current_table:
                related_tables = [t.strip() for t in line.split("Links to:")[1].split(",")]
                for rel_table in related_tables:
                    if rel_table in self.schema['tables']:
                        self.schema['relationships'].append((current_table, rel_table))

class SQLValidator:
    @staticmethod
    def fix_sql_syntax(sql: str) -> str:
        
        # Fix common SQL syntax issues
        sql = re.sub(r"AS\s+\/([^\]]+)\]", r"AS [\1]", sql)
        sql = re.sub(r"/(\w+)]", r"[\1]", sql)
        
        sql = re.sub(r"(?<!\[)(\b\w+)\]", r"[\1]", sql)
        
        sql = sql.replace("/]", "]").replace("AS /", "AS [").replace("]/", "]")
        
        sql = re.sub(r"FROM\s+dbo\]\.\[", "FROM [dbo].[", sql)
        sql = re.sub(r"JOIN\s+dbo\]\.\[", "JOIN [dbo].[", sql)
        
        sql = re.sub(r"SELECT\s+", "SELECT ", sql)
        
        join_pattern = r"JOIN\s+\[dbo\]\.\[\w+\]\s+\w+\s+ON\s+\w+\s*;"
        sql = re.sub(join_pattern, "-- FIX NEEDED: JOIN without condition", sql)
        
        return sql.strip()
    
    @staticmethod
    def extract_tables_and_columns(sql):
        tables = set(re.findall(r"\[dbo\]\.\[([^\]]+)\]", sql))
        columns = set(re.findall(r"(?<!\[dbo\]\.\[)\[([^\]]+)\]", sql))
        return tables, columns
    
    def validate(self, sql, schema):
        sql = self.fix_sql_syntax(sql)
        tables, columns = self.extract_tables_and_columns(sql)
        
        
        missing_tables = [t for t in tables if t not in schema['tables']]
        
        missing_columns = []
        for col in columns:
            found = False
            for table in tables:
                if table in schema['tables'] and col in schema['tables'][table]['columns']:
                    found = True
                    break
            if not found:
                missing_columns.append(col)
        
        join_issues = []
        join_matches = re.finditer(
            r"JOIN\s+\[dbo\]\.\[([^\]]+)\]\s+ON\s+(.+?)(?=\s+(?:JOIN|WHERE|GROUP|ORDER|$))", 
            sql, 
            re.IGNORECASE
        )
        
        for match in join_matches:
            table = match.group(1)
            condition = match.group(2)
            join_cols = re.findall(r"\[([^\]]+)\]", condition)
            
            for col in join_cols:
                if not any(col in schema['tables'][t]['columns'] for t in tables if t in schema['tables']):
                    join_issues.append(f"Invalid column '{col}' in JOIN condition for table {table}")
        
        
        corrections = []
        corrected_sql = sql
        
        
        for table in missing_tables:
            related_tables = [
                t for t in tables 
                if (t, table) in schema['relationships'] or (table, t) in schema['relationships']
            ]
            
            if related_tables:
                join_table = related_tables[0]
                join_cond = f"[{join_table}].[{join_table}_key] = [{table}].[{join_table}_key]"
                corrections.append(f"Added missing JOIN to {table}")
                corrected_sql = re.sub(
                    r"(FROM\s+\[dbo\]\.\[[^\]]+\])", 
                    f"\\1\nJOIN [dbo].[{table}] ON {join_cond}", 
                    corrected_sql
                )
        
        return {
            "corrected_sql": corrected_sql,
            "original_sql": sql,
            "missing_tables": missing_tables,
            "missing_columns": missing_columns,
            "join_issues": join_issues,
            "corrections": corrections
        }

class SQLGenerator:
    def __init__(self, model_path):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=35,
            n_batch=512,
            verbose=False
        )
        self.schema_parser = SchemaParser(SCHEMA)
        self.validator = SQLValidator()
        self.metrics = {
            "total_queries": 0,
            "successful_executions": 0,
            "total_generation_time": 0.0,
            "accuracy_scores": [],
            "common_errors": defaultdict(int),
            "query_types": defaultdict(int),
            "response_times": [],
            "query_success_log": []
            
        }
    
    def build_prompt(self, question):
        
        schema = self.schema_parser.schema
        
        table_summaries = []
        for table, details in schema['tables'].items():
            summary = f"- [dbo].[{table}] (PK: {details['primary_key']})"
            if details['columns']:
                summary += f"\n  Columns: {', '.join(sorted(details['columns']))}"
            
            
            rels = [r[1] for r in schema['relationships'] if r[0] == table]
            if rels:
                summary += f"\n  Relates to: {', '.join(rels)}"
            
            table_summaries.append(summary)
        
        
        prompt_sections = {
            "Database": "Azure SQL (T-SQL Syntax)",
            "Schema Overview": "\n".join(table_summaries),
            "Key Relationships": "\n".join(
                f"- [dbo].[{a}] ↔ [dbo].[{b}] (join on {a}_key = {b}_key)"
                for a, b in schema['relationships']
            ),
            "Rules": """
1. Always use [dbo].[table] format for table references
2. Use explicit JOIN conditions with ON clause.
3. For date filtering: WHERE [date_col] BETWEEN @start AND @end.
4. Use TOP instead of LIMIT for row limiting.
5. When aggregating, include non-aggregated columns in GROUP BY.
6. For text searches: WHERE [col] LIKE '%pattern%'.
7. For NULL checks: WHERE [col] IS [NOT] NULL.
8. Prefer INNER JOIN unless outer join is needed.
9. For date formatting: FORMAT([date_col], 'yyyy-MM-dd').
10. Always include a WHERE clause when filtering.
11. If the question mentions product category, use the column [category] from the [dbo].[dim_product] table.
12. When the question mentions "plan" or "plans", it refers to the [dbo].[dim_product] table.
13. When the question says "mobile model", map it to the column [model].
14. When asked about "network provider", use the column [provider] from [dbo].[dim_network].
15. When asked about "quarter", use the column [fiscal_period] from [dbo].[dim_time].
16. The terms "customer" or "client" refer to subscribers in [dbo].[dim_subscriber].
17. Ensure every JOIN includes a valid ON condition like: 'ON table1.key = table2.key'
18. If the question contains phrases like "top", "most", or "highest", use GROUP BY and ORDER BY with an aggregate function like COUNT or SUM.
19. To calculate subscription duration, subtract [subscription_date] from [churn_date]:DATEDIFF(DAY, [subscription_date], [churn_date]) for number of days, or use MONTH or YEAR for longer periods.
20. Never use PostgreSQL functions like date_trunc() or GENERATE_SERIES().
21. Always join [fact_usage] with [dim_time] using time_key for time-based analysis.
22. Avoid implicit joins or WHERE-based joins.

""",
            "Examples": FEW_SHOTS_EXAMPLE,
            "Question": question,
            "Instructions": """
Before writing SQL:
1. Identify needed tables
2. Determine table relationships
3. Plan necessary joins
4. Consider filtering conditions
5. Decide on aggregation

Now generate the SQL query:
"""
        }
        
        return "\n\n".join(f"### {k}:\n{v}" for k, v in prompt_sections.items())
    
    def classify_question(self, question):
        """Categorize question type for better handling"""
        question_lower = question.lower()
        if any(w in question_lower for w in ["top", "most", "highest", "lowest"]):
            return "ranking"
        elif any(w in question_lower for w in ["trend", "over time", "monthly", "quarterly"]):
            return "temporal"
        elif any(w in question_lower for w in ["compare", "vs", "versus", "difference"]):
            return "comparison"
        elif any(w in question_lower for w in ["count", "number", "how many"]):
            return "aggregation"
        return "general"
    
    def generate_sql(self, question: str, execute_func=None) -> str:
        start_time = time()
        self.metrics["total_queries"] += 1
        success_flag = 0  # Track success per query for avg success rate

        try:
            prompt = self.build_prompt(question)
            output = self.llm(
                prompt,
                max_tokens=350,
                stop=["###", "\n\n"],
                temperature=0.1,
                top_p=0.95
            )
            raw_sql = output['choices'][0]['text'].strip()
            formatted_sql = self.format_sql(raw_sql)
            validation_result = self.validator.validate(formatted_sql, self.schema_parser.schema)
            final_sql = validation_result['corrected_sql']

            if not self._is_valid_sql(final_sql):
                raise ValueError("Generated SQL failed validation checks")

            if execute_func:
                try:
                    execute_func(final_sql)
                    self.metrics["successful_executions"] += 1
                    success_flag = 1
                except Exception as e:
                    self.metrics["common_errors"][str(e)] += 1
                    raise ValueError(f"Execution failed: {str(e)}")
            else:
                self.metrics["successful_executions"] += 1
                success_flag = 1

            return final_sql

        except Exception as e:
            self.metrics["common_errors"][str(e)] += 1
            raise ValueError(f"SQL generation failed: {str(e)}")

        finally:
            gen_time = time() - start_time
            self.metrics["total_generation_time"] += gen_time
            self.metrics["response_times"].append(gen_time)
            self.metrics["query_types"][self.classify_question(question)] += 1
            self.metrics.setdefault("query_success_log", []).append(success_flag)




    def _is_valid_sql(self, sql: str) -> bool:
        
        if not sql or not isinstance(sql, str):
            return False
            
        sql = sql.upper().strip()
        return (
            len(sql) > 20 and  
            sql.startswith(("SELECT ", "WITH ")) and
            "FROM " in sql and
            not any(keyword in sql for keyword in ["DROP", "DELETE", "TRUNCATE", "INSERT", "UPDATE"])
        )

    def format_sql(self, sql: str) -> str:
        
        replacements = {
            r"date_trunc\('month', ([^)]+)\)": r"DATEFROMPARTS(YEAR(\1), MONTH(\1), 1)",
            r"date_trunc\('year', ([^)]+)\)": r"DATEFROMPARTS(YEAR(\1), 1, 1)",
            r"TO_CHAR\(": "FORMAT(",
            r"NOW\(\)": "GETDATE()",
            r"::date": "CAST(\1 AS DATE)",
            r"true": "1",
            r"false": "0",
            r"\|\|": "+",
            r"ILIKE": "LIKE",
            r"LIMIT (\d+)": r"TOP \1",
            r"CURRENT_DATE\s*-\s*interval\s*'1 year'":"DATEADD(YEAR, -1, GETDATE())",
        }

        for pattern, replacement in replacements.items():
            sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
        
        return sql.strip()

    
    def get_metrics(self):
        
        total = self.metrics["total_queries"]
        success = self.metrics["successful_executions"]
            
        query_success_log = self.metrics.get("query_success_log", [])
        avg_success_rate = (
            sum(query_success_log) / len(query_success_log) * 100
            if query_success_log else 0
        )

        return {
            "total_queries": self.metrics.get("total_queries", 0),
            "successful_executions": self.metrics.get("successful_executions", 0),
            "avg_generation_time": (
                self.metrics["total_generation_time"] / self.metrics["total_queries"]
                if self.metrics["total_queries"] else 0
            ),
            "success_rate": (
                self.metrics["successful_executions"] / self.metrics["total_queries"] * 100
                if self.metrics["total_queries"] else 0
            ),
            "avg_query_success_rate": f"{avg_success_rate:.1f}%",
            "common_query_types": sorted(
                self.metrics.get("query_types", {}).items(), key=lambda x: -x[1]
            ),
            "top_error_types": sorted(
                self.metrics.get("common_errors", {}).items(), key=lambda x: -x[1]
            )
        }


# Example usage
# if __name__ == "__main__":
#     print("Initializing SQL generator...")
#     generator = SQLGenerator("./models/sqlcoder-7b.Q4_K_M.gguf")
    
#     questions = [
#         "What are the top 5 most expensive products?",
#         "Show average call minutes by device type",
#         "What are the monthly revenue trends for the last year?",
#         "Which states have the highest number of subscribers?"
#     ]
    
#     for question in questions:
#         print(f"\nQuestion: {question}")
#         sql = generator.generate_sql(question)
    
#     print("\nPerformance Report:")
#     print(generator.get_metrics())