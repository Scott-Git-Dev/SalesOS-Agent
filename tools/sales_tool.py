"""
Sales Database Tool
Enhanced text-to-SQL with better error handling and query refinement
"""
import sqlite3
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
# Import config
import sys
from pathlib import Path
from config import SALES_DB_PATH, LLAMA_SERVER_URL, MODEL_NAME, DEBUG_MODE, SQL_PRINTING_ENABLED

sys.path.append(str(Path(__file__).parent.parent))


_SCHEMA_CACHE = None
_SQL_LLM = None

def _get_schema_cached():
    """Get database schema (cached after first call)"""
    global _SCHEMA_CACHE
    
    if _SCHEMA_CACHE is None:
        
        conn = sqlite3.connect(f'file:{SALES_DB_PATH}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        _SCHEMA_CACHE = _get_database_schema(cursor)
        conn.close()
          
    return _SCHEMA_CACHE

def _get_sql_llm():
    """Get or create cached SQL generation LLM"""

    global _SQL_LLM

    if _SQL_LLM is None:
        _SQL_LLM = ChatOpenAI(
            model=MODEL_NAME,
            temperature=0.0,
            base_url=LLAMA_SERVER_URL,
            max_tokens=4500
        )
    return _SQL_LLM


@tool
def query_sales_database(question: str) -> str:
    """
    Query the company sales database for business intelligence using natural language.
    
    Use for questions about: 
    - Sales performance, revenue, trends
    - Customer data, orders, purchasing patterns  
    - Product performance, bestsellers
    - Regional performance
    - Sales rep performance
    - Time-based analysis (monthly, quarterly, YTD)
    
    This tool CANNOT answer questions about:
    - Goals, targets, quotas (use search_local_docs instead)
    - Strategies, plans, objectives (use search_local_docs instead)
    - Future projections or forecasts (use search_local_docs for planning docs)

    Examples:
    - "What were total sales last quarter?"
    - "Who are our top 5 customers by revenue?"
    - "Which products sell best in Europe?"
    - "Show me monthly revenue trend for 2024"
    
    Args:
        question: Natural language question about sales data

    Returns:
        Query results formatted as structured data, or error message with guidance
    """
    
    try:
        # Connect to database
        conn = sqlite3.connect(f'file:{SALES_DB_PATH}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        schema = _get_schema_cached()  # ← Uses cache
        
        # Use LLM to generate SQL
        sql_query = _generate_sql_with_llm(question, schema)
        
        # Handle scope errors (question asks for data not in DB)
        if sql_query.startswith("SCOPE_ERROR:"):
            conn.close()
            error_message = sql_query.replace("SCOPE_ERROR:", "").strip()
            return f"⚠️ This question requires data not in the sales database.\n\n{error_message}\n\n💡 Try using search_local_docs to find this information in company documents."
        
        # Handle other SQL generation errors
        if sql_query.startswith("ERROR:"):
            conn.close()
            return sql_query
        
        sql_query = sql_query

        if SQL_PRINTING_ENABLED or DEBUG_MODE:
            print(f"[SQL QUERY] {sql_query}")
        
        # Validate SQL for safety
        _validate_sql(sql_query)

        # Execute query
        try:
            cursor.execute(sql_query)
            results = cursor.fetchall()
        except sqlite3.Error as sql_error:
            # Try to refine query if execution fails
            if DEBUG_MODE:
                print(f"[SQL ERROR] {sql_error}")
                print("[SQL REFINEMENT] Attempting to fix query...")
            
            refined_sql = _refine_failed_query(question, sql_query, str(sql_error), schema)
            
            if refined_sql and not refined_sql.startswith("ERROR:"):
                if DEBUG_MODE:
                    print(f"[SQL REFINED] {refined_sql}")
                
                cursor.execute(refined_sql)
                results = cursor.fetchall()
            else:
                raise sql_error
        
        conn.close()
        
        if not results:
            return f"✓ Query executed successfully but returned no results.\n\nQuestion: {question}\n\nThis might mean:\n- No data matches the criteria\n- The date range is outside available data\n- The filter conditions are too restrictive"
        
        # Return structured results for agent to interpret
        rows = [dict(row) for row in results]
        
        # Format nicely
        return _format_results_structured(rows, question)
    
    except sqlite3.Error as e:
        return f"❌ Database error: {str(e)}\n\nThe generated SQL may be invalid. Try rephrasing your question or check if the data exists in the database."
    
    except Exception as e:
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        return f"❌ Error: {str(e)}"


def _get_database_schema(cursor) -> str:
    """Get detailed database schema with sample data and relationships"""
    
    schema_parts = []
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    
    for table in tables:
        # Get column info
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        
        schema_parts.append(f"\n{'='*60}")
        schema_parts.append(f"Table: {table}")
        schema_parts.append(f"{'='*60}")
        schema_parts.append("Columns:")
        for col in columns:
            pk_marker = " [PRIMARY KEY]" if col[5] else ""
            schema_parts.append(f"  - {col[1]} ({col[2]}){pk_marker}")
        
        # Get sample row
        cursor.execute(f"SELECT * FROM {table} LIMIT 1")
        sample = cursor.fetchone()
        if sample:
            schema_parts.append("\nSample row:")
            sample_dict = dict(sample)
            for key, value in list(sample_dict.items())[:6]:
                schema_parts.append(f"  {key}: {value}")
    
    schema_text = "\n".join(schema_parts)
    
    # Enhanced guidelines
    guidelines = """

═══════════════════════════════════════════════════════════
RELATIONSHIPS & QUERY PATTERNS
═══════════════════════════════════════════════════════════

Key Foreign Key Relationships:
- customers.region_id → regions.region_id
- sales.customer_id → customers.customer_id
- sales_items.sale_id → sales.sale_id
- sales_items.product_id → products.product_id

Critical SQL Rules:
1. ALWAYS filter by status = 'Completed' when calculating revenue/totals
2. Use sales.total_amount for aggregate revenue (it's pre-calculated with discounts)
3. For product-level detail, join through sales_items
4. ALWAYS use explicit JOINs, never implicit joins
5. Use table aliases for readability

Date/Time Patterns:
- Month grouping: strftime('%Y-%m', sale_date)
- Year grouping: strftime('%Y', sale_date)
- Last quarter start: date('now', 'start of year', '+' || ((cast(strftime('%m', 'now') as integer) - 1) / 3 * 3 - 3) || ' months')
- Last quarter end:   date('now', 'start of year', '+' || ((cast(strftime('%m', 'now') as integer) - 1) / 3 * 3) || ' months', '-1 day')
- Last year: date('now', '-1 year')
- This year: strftime('%Y', sale_date) = strftime('%Y', 'now')
- Last 30 days: date('now', '-30 days')

Common Query Templates:
═══════════════════════

1. Total Revenue (Simple):
   SELECT SUM(total_amount) as total_revenue
   FROM sales
   WHERE status = 'Completed'

2. Top Customers:
   SELECT c.company, SUM(s.total_amount) as total_revenue
   FROM sales s
   JOIN customers c ON s.customer_id = c.customer_id
   WHERE s.status = 'Completed'
   GROUP BY c.company
   ORDER BY total_revenue DESC
   LIMIT 10

3. Product Performance:
   SELECT p.product_name, 
          SUM(si.quantity) as units_sold,
          SUM(si.quantity * si.unit_price * (1 - si.discount)) as revenue
   FROM sales_items si
   JOIN products p ON si.product_id = p.product_id
   JOIN sales s ON si.sale_id = s.sale_id
   WHERE s.status = 'Completed'
   GROUP BY p.product_name
   ORDER BY revenue DESC

4. Monthly Trend:
   SELECT strftime('%Y-%m', sale_date) as month,
          SUM(total_amount) as revenue
   FROM sales
   WHERE status = 'Completed'
   GROUP BY month
   ORDER BY month

5. Regional Analysis:
   SELECT r.region_name,
          COUNT(DISTINCT c.customer_id) as customer_count,
          SUM(s.total_amount) as total_revenue
   FROM sales s
   JOIN customers c ON s.customer_id = c.customer_id
   JOIN regions r ON c.region_id = r.region_id
   WHERE s.status = 'Completed'
   GROUP BY r.region_name

6. Last Quarter Revenue:
   SELECT SUM(total_amount) as total_revenue
   FROM sales
   WHERE status = 'Completed'
     AND sale_date >= date('now', 'start of year', '+' || ((cast(strftime('%m', 'now') as integer) - 1) / 3 * 3 - 3) || ' months')
     AND sale_date <  date('now', 'start of year', '+' || ((cast(strftime('%m', 'now') as integer) - 1) / 3 * 3) || ' months')

Customer Name Convention:
- customers.company = Business/organization name (use for "top customers", "best customers")
- customers.customer_name = Contact person name (use only for "contact name", "point of contact")

LIMIT Guidelines:
- DO NOT use LIMIT for: totals, averages, aggregations, time series, grouped data
- DO use LIMIT for: top N lists, recent transactions, sample data
"""
    
    return schema_text + guidelines

def _validate_sql(query: str) -> None:
    """Validate SQL query for security"""
    
    query_upper = query.upper()
    
    # Block dangerous keywords
    forbidden = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 
                 'CREATE', 'TRUNCATE', 'REPLACE', 'PRAGMA']
    
    for keyword in forbidden:
        if keyword in query_upper:
            raise ValueError(f"Query contains forbidden keyword: {keyword}")
        

def _generate_sql_with_llm(question: str, schema: str) -> str:
    """
    Generate SQL using LLM with improved prompt engineering.
    
    Returns:
        - Valid SQL query string
        - "SCOPE_ERROR: <message>" if question asks for data not in database
        - "ERROR: <message>" for other failures
    """
    
    sql_llm = _get_sql_llm()  # Use cached LLM
    
    # IMPROVED: Structured prompt with system message
    system_message = """You are an expert SQL query generator for a sales analytics database.

Your ONLY job is to convert natural language questions into valid SQLite queries.

CRITICAL RULES:
1. Return ONLY the SQL query - no markdown, no explanations, no code blocks
2. Generate exactly ONE query
3. Do NOT add semicolons at the end
4. Use explicit JOINs only
5. ALWAYS filter by status = 'Completed' when calculating revenue/metrics
6. Use appropriate aggregations (SUM, COUNT, AVG)
7. Use meaningful aliases for readability

SCOPE CHECKING:
If the question asks for data NOT in the schema below, respond with:
SCOPE_ERROR: [explain what data is missing and suggest alternative]

Examples of out-of-scope questions:
- "sales goals", "targets", "quotas" → SCOPE_ERROR: Sales goals and targets are not stored in this database. These are typically found in strategic planning documents.
- "customer satisfaction" → SCOPE_ERROR: Customer satisfaction scores are not tracked in this database.
- "future projections" → SCOPE_ERROR: This database contains historical data only, not forecasts.

LIMIT USAGE:
- DO NOT use LIMIT for: aggregations, totals, trends, time series, grouped data
- DO use LIMIT for: top N lists, recent individual records, sample data"""

    user_message = f"""DATABASE SCHEMA:
        {schema}

        USER QUESTION: {question}

        Generate the SQL query:"""
    
    try:
        response = sql_llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ])
        
        sql_query = response.content.strip()
        
        # Clean up formatting
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        sql_query = sql_query.rstrip(';')
        
        # Check for scope errors
        if sql_query.startswith("SCOPE_ERROR:"):
            return sql_query
        
        # Basic validation
        if not sql_query.upper().startswith("SELECT"):
            return "ERROR: Generated query must be a SELECT statement"
        
        return sql_query
    
    except Exception as e:
        if DEBUG_MODE:
            print(f"[ERROR] SQL generation failed: {e}")
        return f"ERROR: Failed to generate SQL query: {str(e)}"


def _refine_failed_query(question: str, failed_sql: str, error: str, schema: str) -> str | None:
    """
    Attempt to refine a failed SQL query by providing error feedback to LLM.
    
    This implements a simple self-correction loop.
    """
    
    sql_llm = _get_sql_llm()
    
    refinement_prompt = f"""The following SQL query failed with an error. Generate a corrected version.

ORIGINAL QUESTION: {question}

FAILED SQL QUERY:
{failed_sql}

ERROR MESSAGE:
{error}

SCHEMA:
{schema}

Common issues and fixes:
- Column doesn't exist → Check schema for correct column names
- Ambiguous column → Add table aliases
- Syntax error → Check JOIN syntax, WHERE clause placement
- Type mismatch → Ensure proper type conversions

Generate the CORRECTED SQL query (no explanations, just the query):"""
    
    try:
        response = sql_llm.invoke([HumanMessage(content=refinement_prompt)])
        refined_sql = response.content.strip()
        refined_sql = refined_sql.replace("```sql", "").replace("```", "").strip()
        refined_sql = refined_sql.rstrip(';')
        
        # Validate it's different from original
        if refined_sql.upper() == failed_sql.upper():
            return None
            
        return refined_sql
    
    except Exception as e:
        if DEBUG_MODE:
            print(f"[ERROR] Query refinement failed: {e}")
        return None


def _format_results_structured(rows: list, question: str) -> str:
    """
    Format query results in a structured, parseable format.
    
    Returns results that are:
    - Easy for the agent to interpret
    - Include metadata for context
    - Properly formatted for downstream use (e.g., charting)
    """
    
    if not rows:
        return "No results"
    
    # Single aggregate result (one row, typically aggregations)
    if len(rows) == 1 and len(rows[0]) <= 3:
        parts = ["📊 Query Result:"]
        for key, value in rows[0].items():
            formatted_value = _format_value(key, value)
            parts.append(f"  • {key}: {formatted_value}")
        return "\n".join(parts)
    
    # Multiple rows - return structured data
    result_parts = [
        f"📊 Query returned {len(rows)} row(s):",
        ""
    ]
    
    # Show first few rows in readable format
    display_rows = min(10, len(rows))
    
    for i, row in enumerate(rows[:display_rows], 1):
        row_parts = []
        for key, value in row.items():
            formatted_value = _format_value(key, value)
            row_parts.append(f"{key}: {formatted_value}")
        result_parts.append(f"  [{i}] {' | '.join(row_parts)}")
    
    if len(rows) > display_rows:
        result_parts.append(f"\n  ... and {len(rows) - display_rows} more rows")
    
    # Add data summary for context
    result_parts.extend([
        "",
        "💡 Data Summary:",
        f"  • Total rows: {len(rows)}",
        f"  • Columns: {', '.join(rows[0].keys())}"
    ])
    
    # If data looks suitable for visualization, suggest it
    if len(rows) >= 3 and len(rows[0]) == 2:
        result_parts.append("  • 💭 This data could be visualized with create_chart")
    
    return "\n".join(result_parts)


def _format_value(key: str, value) -> str:
    """Format a single value based on its type and context"""
    
    if value is None:
        return "N/A"
    
    # Money fields
    money_keywords = ["revenue", "amount", "spent", "value", "price", "cost", "sales"]
    if any(keyword in key.lower() for keyword in money_keywords):
        if isinstance(value, (int, float)):
            return f"${value:,.2f}"
    
    # Large numbers
    if isinstance(value, int) and value > 1000:
        return f"{value:,}"
    
    # Floats
    if isinstance(value, float):
        return f"{value:.2f}"
    
    return str(value)
