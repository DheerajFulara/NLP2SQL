import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import io
import json
import re
import logging
from typing import Dict, List, Optional, Union, TypedDict
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.colored_header import colored_header
import streamlit_nested_layout
import numpy as np
from streamlit_extras.dataframe_explorer import dataframe_explorer
import src.database.DB_Config as DB_Config
from src.prompts.Base_Prompt import SYSTEM_MESSAGE
from src.api.LLM_Config import get_completion_from_messages
import hashlib
from datetime import datetime
from time import time
from collections import defaultdict
from jsonschema import validate as json_validate, ValidationError

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SUPPORTED_CHART_TYPES = {
    "Bar Chart": "A chart that presents categorical data with rectangular bars.",
    "Line Chart": "A chart that displays information as a series of data points called 'markers' connected by straight line segments.",
    "Scatter Plot": "A plot that displays values for typically two variables for a set of data.",
    "Area Chart": "A chart that displays quantitative data visually, using the area below the line.",
    "Histogram": "A graphical representation of the distribution of numerical data.",
    "Pie Chart": "A chart that shows proportions of a whole using slices.",
    "Box Plot": "A chart that shows the distribution of data based on quartiles."
}

# Page Configuration with dark theme details
st.set_page_config(
    page_icon="🗃️",
    page_title="NLP2SQL",
    layout="wide"
)

def apply_custom_theme():
    custom_css = f"""
    <style>
    /* Global Styles */
    body, .stApp {{
        background-color: #1e1e1e;
        color: #64ffda;
        font-family: sans-serif;
    }}
    /* Sidebar */
    .css-1d391kg, .stSidebar .sidebar-content {{
        background-color: #333333;
    }}
    /* Buttons */
    .stButton>button {{
        background-color: #00ADB5;
        color: #fff;
        border: none;
    }}
    /* Expander */
    .stExpander {{
        background-color: #333333;
        border: none;
        border-radius: 8px;
        padding: 0.5rem;
    }}
    .stExpander .stExanderHeader, .stExpander .stExanderContent {{
        color: #64ffda;
    }}
    /* Tabs */
    .stTabs [data-baseweb="tab"] {{
        background-color: #333333;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        color: #64ffda;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: #00ADB5;
        color: #fff;
    }}
    /* Code Blocks */
    pre {{
        background-color: #333333;
        color: #64ffda;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Apply the custom theme early
apply_custom_theme()

load_dotenv()

@st.cache_resource
def load_system_message(schemas: dict) -> str:
    """Load and format the system message with JSON-serialized schemas."""
    return SYSTEM_MESSAGE.format(schemas=json.dumps(schemas, indent=2))

# Add input validation to prevent SQL injection and other security vulnerabilities

def validate_sql_query(query: str) -> bool:
    """
    Ensure the SQL query is valid and safe (select queries only).

    Parameters:
    - query (str): The SQL query to validate.

    Returns:
    - bool: True if the query is valid and safe, False otherwise.
    """
    if not isinstance(query, str):
        return False

    disallowed_keywords = r'\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|EXEC)\b'

    if re.search(disallowed_keywords, query, re.IGNORECASE):
        return False

    if not query.strip().lower().startswith(('select', 'with')):
        return False

    if query.count('(') != query.count(')'):
        return False

    return True

# --- New helper: Validate that query uses existent tables/columns ---
def validate_query_tables(query: str, schemas: dict) -> bool:
    """
    Very basic check: warn if any known schema table name is missing in the query.
    This is a heuristic check.
    """
    lower_query = query.lower()
    missing = []
    for table in schemas.keys():
        if table.lower() not in lower_query:
            missing.append(table)
    if missing:
        logging.warning(f"LLM query does not mention these tables from the schema: {', '.join(missing)}")
        return False
    return True

def get_data(query: str, db_name: str, db_type: str, host: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> pd.DataFrame:
    """Run the specified query and return the complete resulting DataFrame."""
    if not validate_sql_query(query):
        logger.error("Invalid or unsafe SQL query.")
        return pd.DataFrame()
    # Removed pagination limit and offset
    return DB_Config.query_database(query, db_name, db_type, host, user, password)

def save_temp_file(uploaded_file) -> str:
    """Saves an uploaded file to a temporary location."""
    temp_file_path = "temp_database.db"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    return temp_file_path

# Step 1: Define Type Classes
class Path(TypedDict):
    description: str
    tables: List[str]
    columns: List[List[str]]
    score: int

class TableColumn(TypedDict):
    table: str
    columns: List[str]
    reason: str

class DecisionLog(TypedDict):
    query_input_details: List[str]
    preprocessing_steps: List[str]
    path_identification: List[Path]
    ambiguity_detection: List[str]
    resolution_criteria: List[str]
    chosen_path_explanation: List[TableColumn]
    generated_sql_query: str
    alternative_paths: List[str]
    execution_feedback: List[str]
    final_summary: str
    visualization_suggestion: Optional[str]

DECISION_LOG_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The generated SQL query"},
        "error": {"type": ["string", "null"], "description": "Error message if query generation failed"},
        "decision_log": {
            "type": "object",
            "properties": {
                "query_input_details": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Details about the input query"
                },
                "preprocessing_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Steps taken to preprocess the query"
                },
                "path_identification": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "tables": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "columns": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "score": {"type": "integer"}
                        },
                        "required": ["description", "tables", "columns", "score"]
                    }
                },
                "ambiguity_detection": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "resolution_criteria": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "chosen_path_explanation": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "reason": {"type": "string"}
                        },
                        "required": ["table", "columns", "reason"]
                    }
                },
                "generated_sql_query": {"type": "string"},
                "alternative_paths": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "execution_feedback": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "final_summary": {"type": "string"},
                "visualization_suggestion": {"type": ["string", "null"]}
            },
            "required": [
                "query_input_details",
                "preprocessing_steps",
                "path_identification",
                "ambiguity_detection",
                "resolution_criteria",
                "chosen_path_explanation",
                "generated_sql_query",
                "alternative_paths",
                "execution_feedback",
                "final_summary"
            ]
        }
    },
    "required": ["query", "decision_log"]
}

# Step 3: Implement the Modified generate_sql_query Function
def generate_sql_query(user_message: str, schemas: dict, max_attempts: int = 1) -> dict:
    """Generate a SQL query using LLM responses and validate output structure."""
    formatted_system_message = f"""
    {load_system_message(schemas)}

    IMPORTANT: Your response must be valid JSON matching this schema:
    {json.dumps(DECISION_LOG_SCHEMA, indent=2)}

    Ensure all responses strictly follow this format. Include a final_summary and visualization_suggestion in the decision_log.
    """

    for attempt in range(max_attempts):
        try:
            response = get_completion_from_messages(formatted_system_message, user_message)
            # Strip any triple-backtick fences
            response = re.sub(r'^```json\s*', '', response.strip())
            response = re.sub(r'```$', '', response.strip())
            json_response = json.loads(response)
            try:
                json_validate(instance=json_response, schema=DECISION_LOG_SCHEMA)
            except ValidationError as ve:
                logger.warning(f"JSON schema validation error: {ve.message}. Attempt: {attempt + 1}")
                continue

            # Validate referenced tables in the generated SQL query
            if not validate_query_tables(json_response.get('query', ''), schemas):
                logger.warning("Generated SQL query contains non-existent tables/columns.")

            return {
                "query": json_response.get('query'),
                "error": json_response.get('error'),
                "decision_log": json_response['decision_log'],
                "visualization_recommendation": json_response['decision_log'].get('visualization_suggestion')
            }

        except json.JSONDecodeError as e:
            logger.exception(f"Invalid JSON response: {response}, Error: {e}")
            continue
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            continue

    return {
        "error": "Failed to generate a valid SQL query after multiple attempts.",
        "decision_log": {
            "execution_feedback": ["Failed to generate a valid response after multiple attempts."],
            "final_summary": "Query generation failed."
        }
    }

# Step 4: Implement Response Validation
def validate_response_structure(response: dict) -> bool:
    """Check if the LLM response follows the expected JSON schema."""
    try:
        if not all(key in response for key in ["query", "decision_log"]):
            return False

        decision_log = response["decision_log"]
        required_sections = [
            "query_input_details",
            "preprocessing_steps",
            "path_identification",
            "ambiguity_detection",
            "resolution_criteria",
            "chosen_path_explanation",
            "generated_sql_query",
            "alternative_paths",
            "execution_feedback",
            "final_summary"
        ]

        if not all(key in decision_log for key in required_sections):
            return False

        for path in decision_log["path_identification"]:
            if not all(key in path for key in ["description", "tables", "columns", "score"]):
                return False

        for explanation in decision_log["chosen_path_explanation"]:
            if not all(key in explanation for key in ["table", "columns", "reason"]):
                return False

        return True

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False

def build_markdown_decision_log(decision_log: Dict) -> str:
    """Convert the decision log into a markdown-formatted string."""
    markdown_log = []

    # Query Input Details
    if query_details := decision_log.get("query_input_details"):
        markdown_log.extend([
            "### Query Input Analysis",
            "\n".join(f"- {detail}" for detail in query_details),
            ""
        ])

    # Preprocessing Steps
    if preprocessing := decision_log.get("preprocessing_steps"):
        markdown_log.extend([
            "### Preprocessing Steps",
            "\n".join(f"- {step}" for step in preprocessing),
            ""
        ])

    # Path Identification
    if paths := decision_log.get("path_identification"):
        markdown_log.extend([
            "### Path Identification",
            "\n".join([
                f"**Path {i+1}** (Score: {path['score']})\n"
                f"- Description: {path['description']}\n"
                f"- Tables: {', '.join(path['tables'])}\n"
                f"- Columns: {', '.join([', '.join(cols) for cols in path['columns']])}"
                for i, path in enumerate(paths)
            ]),
            ""
        ])

    # Ambiguity Detection
    if ambiguities := decision_log.get("ambiguity_detection"):
        markdown_log.extend([
            "### Ambiguity Analysis",
            "\n".join(f"- {ambiguity}" for ambiguity in ambiguities),
            ""
        ])

    # Resolution Criteria
    if criteria := decision_log.get("resolution_criteria"):
        markdown_log.extend([
            "### Resolution Criteria",
            "\n".join(f"- {criterion}" for criterion in criteria),
            ""
        ])

    # Chosen Path Explanation
    if chosen_path := decision_log.get("chosen_path_explanation"):
        markdown_log.extend([
            "### Selected Tables and Columns",
            "\n".join([
                f"**{table['table']}**\n"
                f"- Columns: {', '.join(table['columns'])}\n"
                f"- Reason: {table['reason']}"
                for table in chosen_path
            ]),
            ""
        ])

    # Generated SQL Query
    if sql_query := decision_log.get("generated_sql_query"):
        markdown_log.extend([
            "### Generated SQL Query",
            f"```sql\n{sql_query}\n```",
            ""
        ])

    # Alternative Paths
    if alternatives := decision_log.get("alternative_paths"):
        markdown_log.extend([
            "### Alternative Approaches",
            "\n".join(f"- {alt}" for alt in alternatives),
            ""
        ])

    # Execution Feedback
    if feedback := decision_log.get("execution_feedback"):
        markdown_log.extend([
            "### Execution Feedback",
            "\n".join(f"- {item}" for item in feedback),
            ""
        ])

    # Final Summary
    if summary := decision_log.get("final_summary"):
        markdown_log.extend([
            "### Summary",
            summary,
            ""
        ])

    # Visualization Suggestion
    if viz_suggestion := decision_log.get("visualization_suggestion"):
        markdown_log.extend([
            "### Visualization Recommendation",
            f"Suggested visualization type: {repr(viz_suggestion)}",
            ""
        ])

    # Join with proper line breaks and clean up any extra spaces
    return "\n".join(line.rstrip() for line in markdown_log)

def create_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str) -> Optional[any]:
    """Construct a Plotly chart without color column and trendline options."""
    try:
        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, title=f"Bar Chart of {y_col} by {x_col}")
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart of {y_col} by {x_col}", markers=True)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot of {y_col} vs {x_col}", hover_data=[x_col, y_col])
        elif chart_type == "Area Chart":
            fig = px.area(df, x=x_col, y=y_col, title=f"Area Chart of {y_col} by {x_col}")
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, nbins=30, title=f"Histogram of {x_col}")
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_col, values=y_col, title=f"Pie Chart of {x_col}")
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot of {y_col} grouped by {x_col}")
        else:
            st.warning("Chart type not recognized.")
            return None

        fig.update_layout(autosize=True, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    except Exception as e:
        st.error(f"Error generating the chart: {e}")
        logger.error(f"Error generating chart: {e}")
        return None

def display_summary_statistics(df: pd.DataFrame) -> None:
    """Show optimized summary statistics, filtering out unnecessary metrics."""

    if df.empty:
        st.warning("The DataFrame is empty. Unable to display summary statistics.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime']).columns

    # Tabs for organized statistics
    tab1, tab2, tab3 = st.tabs(["Numeric Summary", "Categorical Analysis", "Missing Data & Correlations"])

    # --- NUMERIC SUMMARY ---
    with tab1:
        st.markdown("### Numeric Summary Statistics")
        filtered_stats = df[numeric_cols].describe().T

        # Drop meaningless statistics
        filtered_stats = filtered_stats.drop(columns=["count"], errors="ignore")

        # Add only necessary statistics
        filtered_stats["median"] = df[numeric_cols].median()
        filtered_stats["iqr"] = filtered_stats["75%"] - filtered_stats["25%"]
        filtered_stats["std"] = df[numeric_cols].std()

        # Filter out columns with no variance (constant values)
        filtered_stats = filtered_stats.loc[filtered_stats["std"] > 0]

        # Format output
        st.dataframe(filtered_stats.style.format("{:.2f}").highlight_max(axis=0, color="lightgreen"))

        # Histograms for meaningful distributions
        for col in numeric_cols:
            if df[col].nunique() > 1:
                st.markdown(f"**Distribution of {col}**")
                st.plotly_chart(px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}"), use_container_width=True)

    # --- CATEGORICAL ANALYSIS ---
    with tab2:
        st.markdown("### Categorical Data Insights")

        for col in categorical_cols:
            value_counts = df[col].value_counts()
            unique_count = value_counts.shape[0]

            # Only show if the column has meaningful variability
            if unique_count < len(df) * 0.8:
                st.markdown(f"**{col}:** {unique_count} unique values")
                freq_table = value_counts.reset_index()
                freq_table.columns = ["Category", "Count"]
                freq_table["Percentage"] = (freq_table["Count"] / len(df) * 100).round(2)
                st.table(freq_table.style.format({"Percentage": "{:.2f}%"}))

                if unique_count <= 10:
                    st.plotly_chart(px.pie(freq_table, names="Category", values="Count", title=f"Pie Chart for {col}"), use_container_width=True)
                else:
                    st.plotly_chart(px.bar(freq_table, x="Category", y="Count", title=f"Bar Chart for {col}"), use_container_width=True)

    # --- MISSING DATA & CORRELATIONS ---
    with tab3:
        st.markdown("### Missing Data Analysis")

        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            missing_df = missing_data.reset_index()
            missing_df.columns = ["Column", "Missing Values"]
            missing_df["Percentage"] = (missing_df["Missing Values"] / len(df) * 100).round(2)
            st.table(missing_df.style.format({"Percentage": "{:.2f}%"}))
        else:
            st.success("No missing data detected.")

        st.markdown("### Correlation Matrix")
        if len(numeric_cols) >= 2:
            correlation_matrix = df[numeric_cols].corr()
            heat_fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
            st.plotly_chart(heat_fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation analysis.")

def perform_advanced_analysis(df: pd.DataFrame) -> None:
    """Perform advanced statistical analysis on the dataset."""
    st.markdown("## 📊 Advanced Statistical Analysis")

    # Create tabs for different analyses
    tabs = st.tabs(["Distribution Analysis", "Outlier Detection", "Time Series Analysis", "Feature Relationships"])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns

    with tabs[0]:
        st.markdown("### 📈 Distribution Analysis")
        if len(numeric_cols) > 0:
            col = st.selectbox("Select column for distribution analysis", numeric_cols)

            # Calculate statistical measures
            skewness = stats.skew(df[col].dropna())
            kurtosis = stats.kurtosis(df[col].dropna())

            # Create distribution plot
            fig = ff.create_distplot([df[col].dropna()], [col], bin_size=0.2)
            st.plotly_chart(fig, use_container_width=True)

            # Display statistical measures
            col1, col2, col3 = st.columns(3)
            col1.metric("Skewness", f"{skewness:.2f}")
            col2.metric("Kurtosis", f"{kurtosis:.2f}")
            col3.metric("Normality Test p-value", f"{stats.normaltest(df[col].dropna())[1]:.4f}")

    with tabs[1]:
        st.markdown("### 🔍 Outlier Detection")
        if len(numeric_cols) > 0:
            col = st.selectbox("Select column for outlier detection", numeric_cols, key="outlier_col")

            # Calculate outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]

            # Create box plot
            fig = go.Figure()
            fig.add_trace(go.Box(y=df[col], name=col))
            st.plotly_chart(fig, use_container_width=True)

            if not outliers.empty:
                st.markdown(f"**Found {len(outliers)} outliers:**")
                st.dataframe(outliers)

    with tabs[2]:
        st.markdown("### ⏳ Time Series Analysis")
        if len(datetime_cols) > 0:
            date_col = st.selectbox("Select date column", datetime_cols)
            value_col = st.selectbox("Select value column", numeric_cols)

            # Ensure data is sorted by date
            ts_data = df[[date_col, value_col]].sort_values(date_col)
            ts_data = ts_data.set_index(date_col)

            # Automatically detect the period based on the frequency of the date column
            period = st.number_input("Enter the period for seasonal decomposition (default is 12)", min_value=1, value=12)

            # Perform seasonal decomposition
            try:
                decomposition = seasonal_decompose(ts_data[value_col], period=period)

                # Plot components
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.trend, name='Trend'))
                fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.seasonal, name='Seasonal'))
                fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.resid, name='Residual'))
                fig.update_layout(title='Time Series Decomposition')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Could not perform seasonal decomposition. Ensure enough data points and regular intervals.")

    with tabs[3]:
        st.markdown("### 🔗 Feature Relationships")
        if len(numeric_cols) >= 2:
            # Correlation analysis
            correlation = df[numeric_cols].corr()

            # Heatmap
            fig = px.imshow(correlation,
                          labels=dict(color="Correlation"),
                          title="Feature Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

            # VIF Analysis
            if st.checkbox("Show Variance Inflation Factor (VIF) Analysis"):
                if len(numeric_cols) < 2:
                    st.warning("At least two numeric columns are required to calculate VIF.")
                else:
                    try:
                        X = df[numeric_cols].dropna()
                        vif_data = pd.DataFrame()
                        vif_data["Feature"] = numeric_cols
                        vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                                         for i in range(X.shape[1])]
                        st.dataframe(vif_data.sort_values('VIF', ascending=False))
                    except Exception as e:
                        st.warning("Could not calculate VIF. Check for multicollinearity or missing values.")

def assess_data_quality(df: pd.DataFrame) -> None:
    """Assess the quality of the dataset and provide detailed insights."""
    st.markdown("## 🔍 Data Quality Assessment")

    # Create tabs for different quality checks
    tabs = st.tabs(["Overview", "Missing Values", "Duplicates", "Consistency", "Anomalies"])

    with tabs[0]:
        st.markdown("### 📊 Data Quality Overview")

        # Basic statistics
        total_rows = len(df)
        total_cols = len(df.columns)
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # in MB

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{total_rows:,}")
        col2.metric("Total Columns", total_cols)
        col3.metric("Memory Usage", f"{memory_usage:.2f} MB")
        col4.metric("Data Types", len(df.dtypes.unique()))

        # Data type distribution
        dtype_counts = df.dtypes.value_counts()
        fig = px.pie(values=dtype_counts.values,
                    names=dtype_counts.index.astype(str),
                    title="Column Data Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.markdown("### ❌ Missing Values Analysis")

        # Calculate missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing Percentage': missing_pct.values
        }).sort_values('Missing Percentage', ascending=False)

        # Display missing values
        if missing_df['Missing Count'].sum() > 0:
            st.dataframe(missing_df)

            # Visualize missing values
            fig = px.bar(missing_df,
                        x='Column',
                        y='Missing Percentage',
                        title="Missing Values by Column")
            st.plotly_chart(fig, use_container_width=True)

            # Missing patterns
            if st.checkbox("Show Missing Value Patterns"):
                fig = go.Figure()
                for col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index[df[col].isnull()],
                        y=[col] * df[col].isnull().sum(),
                        mode='markers',
                        name=col
                    ))
                fig.update_layout(title="Missing Value Patterns Across Records")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found in the dataset!")

    with tabs[2]:
        st.markdown("### 🔄 Duplicate Analysis")

        # Check for duplicate rows
        duplicates = df.duplicated()
        duplicate_count = duplicates.sum()

        if duplicate_count > 0:
            st.warning(f"Found {duplicate_count} duplicate rows ({(duplicate_count/len(df)*100):.2f}% of data)")

            # Show duplicate rows
            if st.checkbox("Show Duplicate Rows"):
                st.dataframe(df[duplicates])

            # Analyze column-wise duplicates
            col_duplicates = {col: df[col].duplicated().sum() for col in df.columns}
            col_dup_df = pd.DataFrame({
                'Column': col_duplicates.keys(),
                'Duplicate Count': col_duplicates.values(),
                'Duplicate Percentage': [v/len(df)*100 for v in col_duplicates.values()]
            }).sort_values('Duplicate Count', ascending=False)

            st.markdown("### Column-wise Duplicates")
            st.dataframe(col_dup_df)
        else:
            st.success("No duplicate rows found in the dataset!")

    with tabs[3]:
        st.markdown("### 📏 Data Consistency Checks")

        consistency_issues = []

        # Check numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Check for values outside reasonable range
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

            if len(outliers) > 0:
                consistency_issues.append({
                    'Column': col,
                    'Issue': 'Extreme Values',
                    'Count': len(outliers),
                    'Details': f"Values outside range [{lower_bound:.2f}, {upper_bound:.2f}]"
                })

        # Check string columns
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            # Check for mixed case values
            if df[col].str.isupper().any() and df[col].str.islower().any():
                consistency_issues.append({
                    'Column': col,
                    'Issue': 'Mixed Case',
                    'Count': len(df[col].unique()),
                    'Details': 'Contains both upper and lower case values'
                })

            # Check for mixed data types
            numeric_conversion = pd.to_numeric(df[col], errors='coerce')
            if numeric_conversion.isnull().any() and not numeric_conversion.notnull().all():
                consistency_issues.append({
                    'Column': col,
                    'Issue': 'Mixed Types',
                    'Count': len(df[col].unique()),
                    'Details': 'Contains non-numeric values'
                })

        if consistency_issues:
            st.dataframe(pd.DataFrame(consistency_issues))
        else:
            st.success("No major consistency issues found!")

    with tabs[4]:
        st.markdown("### 🎯 Anomaly Detection")

        if len(numeric_cols) > 0:
            col = st.selectbox("Select column for anomaly detection", numeric_cols, key="anomaly_col")

            # Z-score based anomaly detection
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            anomalies = df[col][z_scores > 3]

            if not anomalies.empty:
                st.warning(f"Found {len(anomalies)} potential anomalies using Z-score method")

                # Visualize anomalies
                fig = go.Figure()
                fig.add_trace(go.Box(y=df[col], name=col))
                fig.add_trace(go.Scatter(
                    x=[0]*len(anomalies),
                    y=anomalies,
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=10)
                ))
                st.plotly_chart(fig, use_container_width=True)

                # Show anomalous values
                st.dataframe(anomalies)
            else:
                st.success("No significant anomalies detected!")

def handle_query_response(response: dict, db_name: str, db_type: str, host: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> None:
    """Process LLM-generated SQL query, display results, and handle visualizations."""
    try:
        query = response.get('query', '')
        error = response.get('error', '')
        decision_log = response.get('decision_log', {})
        visualization_recommendation = response.get('visualization_recommendation', None)

        if error:
            detailed_error = generate_detailed_error_message(error)
            st.error(f"Error generating SQL query: {detailed_error}")
            return

        if not query:
            st.warning("No query generated. Please refine your message.")
            return

        st.success("SQL Query generated successfully!")
        colored_header("SQL Query and Summary", color_name="blue-70", description="")
        st.code(query, language="sql")

        if decision_log:
            with st.expander("Decision Log", expanded=False):
                display_decision_log_widgets(decision_log)

        # --- For PostgreSQL: Run EXPLAIN ANALYZE before executing query ---
        if db_type.lower() == 'postgresql':
            with DB_Config.get_connection(db_name, db_type, host, user, password) as conn:
                if conn:
                    try:
                        cur = conn.cursor()
                        cur.execute(f"EXPLAIN ANALYZE {query}")
                        plan = "\n".join(row[0] for row in cur.fetchall())
                        with st.expander("Query Optimization Hints", expanded=True):
                            st.markdown("### Execution Plan (EXPLAIN ANALYZE)")
                            st.code(plan, language="sql")
                            if "Seq Scan" in plan:
                                st.warning("The plan shows a sequential scan. Consider adding indexes on frequently queried columns.")
                    except Exception as ex:
                        st.error("Failed to run EXPLAIN ANALYZE.")
                        logger.exception(f"EXPLAIN ANALYZE failed: {ex}")

        # --- Measure execution time ---
        start_time = time()
        sql_results = get_data(query, db_name, db_type, host, user, password)
        execution_time = time() - start_time

        if sql_results.empty:
            no_result_reason = "The query executed successfully but did not match any records in the database."
            if 'no valid SQL query generated' in decision_log.get("execution_feedback", []):
                no_result_reason = "The query was not generated due to insufficient or ambiguous input."
            elif 'SQL query validation failed' in decision_log.get("execution_feedback", []):
                no_result_reason = "The query failed validation checks and was not executed."
            st.warning(f"The query returned no results because: {no_result_reason}")
            return

        if sql_results.columns.duplicated().any():
            st.error("The query returned a DataFrame with duplicate column names. Please modify your query to avoid this.")
            return

        for col in sql_results.select_dtypes(include=['object']):
            try:
                sql_results[col] = pd.to_datetime(sql_results[col], format='%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                pass

        # --- Handling Large Datasets: Prompt sampling if needed ---
        if sql_results.shape[0] > 10000:
            sample_choice = st.checkbox("Large dataset detected. Sample data for visualization?", value=True)
            if sample_choice:
                sql_results = sql_results.sample(10000)
                st.info("Data has been sampled to 10,000 rows for visualization.")

        colored_header("Query Results and Filter", color_name="blue-70", description="")
        filtered_results = dataframe_explorer(sql_results, case=False)
        st.dataframe(filtered_results, use_container_width=True, height=600)

        colored_header("Summary Statistics", color_name="blue-70", description="")
        display_summary_statistics(filtered_results)

        # Add Advanced Analysis section
        colored_header("Advanced Analysis", color_name="blue-70", description="")
        perform_advanced_analysis(filtered_results)

        # Add Data Quality Assessment section
        colored_header("Data Quality Assessment", color_name="blue-70", description="")
        assess_data_quality(filtered_results)

        performance_metrics = analyze_query_performance(
            query,
            execution_time,
            len(sql_results)
        )

        with st.expander("🔍 Query Performance Analysis", expanded=True):
            st.markdown("### Performance Metrics")

            # Display execution metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Execution Time", f"{performance_metrics['execution_time']:.3f}s")
            with col2:
                st.metric("Rows Returned", performance_metrics['rows_returned'])
            with col3:
                st.metric("Rows/Second", f"{performance_metrics['rows_per_second']:.0f}")
            with col4:
                st.metric("Performance", performance_metrics['performance_class'])

            # Display optimization suggestions
            if performance_metrics['suggestions']:
                st.markdown("### Optimization Suggestions")
                for suggestion in performance_metrics['suggestions']:
                    if suggestion['type'] == 'error':
                        st.error(suggestion['message'])
                    elif suggestion['type'] == 'warning':
                        st.warning(suggestion['message'])
                    else:
                        st.info(suggestion['message'])

        # Add a colored header to separate the Visualization section from Summary Statistics
        colored_header("Visualization Section", color_name="blue-70", description="")

        if len(filtered_results.columns) >= 2:
            with st.sidebar.expander("📊 Visualization Options", expanded=True):
                numerical_cols = filtered_results.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = filtered_results.select_dtypes(include=['object', 'category']).columns.tolist()

                # Suggest default X and Y columns
                suggested_x = numerical_cols[0] if numerical_cols else filtered_results.columns[0]
                suggested_y = numerical_cols[1] if len(numerical_cols) > 1 else (filtered_results.columns[1] if len(filtered_results.columns) > 1 else filtered_results.columns[0])

                x_options = [f"{col} ⭐" if col == suggested_x else col for col in filtered_results.columns]
                y_options = [f"{col} ⭐" if col == suggested_y else col for col in filtered_results.columns]

                x_col = st.selectbox("Select X-axis Column", options=x_options, index=x_options.index(f"{suggested_x} ⭐") if f"{suggested_x} ⭐" in x_options else 0, key="x_axis")
                y_col = st.selectbox("Select Y-axis Column", options=y_options, index=y_options.index(f"{suggested_y} ⭐") if f"{suggested_y} ⭐" in y_options else 0, key="y_axis")
                x_col_clean = x_col.replace(" ⭐", "")
                y_col_clean = y_col.replace(" ⭐", "")

                chart_type_options = ["None", "Bar Chart", "Line Chart", "Scatter Plot", "Area Chart", "Histogram", "Pie Chart", "Box Plot"]
                suggested_chart_type = visualization_recommendation if visualization_recommendation in chart_type_options else ("Bar Chart" if numerical_cols else "None")
                chart_type_display = [f"{chart} ⭐" if chart == suggested_chart_type else chart for chart in chart_type_options]

                try:
                    default_chart_index = chart_type_display.index(f"{suggested_chart_type} ⭐")
                except ValueError:
                    default_chart_index = 0

                chart_type = st.selectbox(
                    "Select Chart Type",
                    options=chart_type_display,
                    index=default_chart_index,
                    help=f"Recommended Chart Type: {suggested_chart_type}",
                    key="chart_type"
                )
                chart_type_clean = chart_type.replace(" ⭐", "")

            if chart_type_clean != "None" and x_col_clean and y_col_clean:
                chart = create_chart(filtered_results, chart_type_clean, x_col_clean, y_col_clean)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)

        export_format = st.selectbox("Select Export Format", options=["CSV", "Excel", "JSON"], key="export_format")
        export_results(filtered_results, export_format)

        if "query_history" not in st.session_state:
            st.session_state.query_history = []
            st.session_state.query_timestamps = []

        st.session_state.query_history.append(query)
        st.session_state.query_timestamps.append(pd.Timestamp.now())

    except Exception as e:
        detailed_error = generate_detailed_error_message(str(e))
        st.error(f"An unexpected error occurred: {detailed_error}")
        logger.exception(f"Unexpected error: {e}")

def export_results(sql_results: pd.DataFrame, export_format: str) -> None:
    """Allow the user to download query results in CSV, Excel, or JSON format."""
    if export_format == "CSV":
        st.download_button(
            label="📥 Download Results as CSV",
            data=sql_results.to_csv(index=False),
            file_name='query_results.csv',
            mime='text/csv'
        )
    elif export_format == "Excel":
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            sql_results.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_buffer.seek(0)
        st.download_button(
            label="📥 Download Results as Excel",
            data=excel_buffer,
            file_name='query_results.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    elif export_format == "JSON":
        st.download_button(
            label="📥 Download Results as JSON",
            data=sql_results.to_json(orient='records'),
            file_name='query_results.json',
            mime='application/json'
        )
    else:
        st.error("⚠️ Selected export format is not supported.")

def analyze_dataframe_for_visualization(df: pd.DataFrame) -> list:
    """Propose suitable chart types based on numeric and categorical column analysis."""
    suggestions = set()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    logger.debug(f"Numerical Columns: {numerical_cols}")
    logger.debug(f"Categorical Columns: {categorical_cols}")

    if len(numerical_cols) == 1:
        suggestions.update(["Histogram", "Box Plot"])
    if len(categorical_cols) == 1:
        suggestions.update(["Bar Chart", "Pie Chart"])

    if len(numerical_cols) >= 2:
        suggestions.update(["Scatter Plot", "Line Chart"])
    elif len(numerical_cols) == 1 and len(categorical_cols) == 1:
        suggestions.update(["Bar Chart"])

    if len(numerical_cols) > 2:
        suggestions.add("Scatter Plot")

    time_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if time_cols:
        suggestions.add("Line Chart")

    ordered_suggestions = [chart for chart in SUPPORTED_CHART_TYPES.keys() if chart in suggestions]
    logger.debug(f"Ordered Suggestions: {ordered_suggestions}")
    return ordered_suggestions

def generate_detailed_error_message(error_message: str) -> str:
    """Use the LLM to produce a user-friendly explanation of any encountered error."""
    prompt = f"Provide a detailed and user-friendly explanation for the following error message:\n\n{error_message}"
    detailed_error = get_completion_from_messages(SYSTEM_MESSAGE, prompt)
    return detailed_error.strip() if detailed_error else error_message

def display_decision_log_widgets(decision_log: Dict) -> None:
    """
    Display the complete decision log with enhanced visual organization and styling.
    Each section of the decision log is displayed in its own tab with appropriate formatting
    and visual hierarchy. Only shows tabs that have data.
    """
    # Create custom CSS for better tab styling
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Define all possible tabs and their data requirements
    tab_data = [
        ("Input Analysis", bool(decision_log.get("query_input_details") or decision_log.get("preprocessing_steps"))),
        ("Paths", bool(decision_log.get("path_identification"))),
        ("Ambiguities", bool(decision_log.get("ambiguity_detection"))),
        ("Resolution", bool(decision_log.get("resolution_criteria"))),
        ("Selected Path", bool(decision_log.get("chosen_path_explanation"))),
        ("SQL Query", bool(decision_log.get("generated_sql_query"))),
        ("Alternatives", bool(decision_log.get("alternative_paths"))),
        ("Feedback", bool(decision_log.get("execution_feedback"))),
        ("Summary", bool(decision_log.get("final_summary") or decision_log.get("visualization_suggestion")))
    ]

    # Filter out tabs with no data
    available_tabs = [tab for tab, has_data in tab_data if has_data]

    if not available_tabs:
        st.info("No decision log data available.")
        return

    # Create tabs only for sections with data
    tabs = st.tabs(available_tabs)

    # Create a mapping of tab names to their indices
    tab_indices = {name: idx for idx, name in enumerate(available_tabs)}

    # Input Details Tab
    if "Input Analysis" in tab_indices:
        with tabs[tab_indices["Input Analysis"]]:
            st.markdown("### Query Input Details")
            for detail in decision_log.get("query_input_details", []):
                st.info(detail)

            if preprocessing_steps := decision_log.get("preprocessing_steps"):
                st.markdown("### Preprocessing Steps")
                for step in preprocessing_steps:
                    st.markdown(f"```\n{step}\n```")

    # Paths Tab
    if "Paths" in tab_indices:
        with tabs[tab_indices["Paths"]]:
            st.markdown("### Path Identification")
            for i, path in enumerate(decision_log.get("path_identification", []), 1):
                with st.expander(f"Path {i} (Score: {path['score']})", expanded=i == 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Description**:")
                        st.markdown(f"_{path['description']}_")
                    with col2:
                        st.metric("Score", path['score'])

                    st.divider()

                    col3, col4 = st.columns(2)
                    with col3:
                        st.markdown("**Tables**")
                        for table in path['tables']:
                            st.markdown(f"- `{table}`")
                    with col4:
                        st.markdown("**Columns**")
                        for cols in path['columns']:
                            st.markdown(f"- `{', '.join(cols)}`")

    # Ambiguities Tab
    if "Ambiguities" in tab_indices:
        with tabs[tab_indices["Ambiguities"]]:
            st.markdown("### Ambiguity Analysis")
            if ambiguities := decision_log.get("ambiguity_detection"):
                for ambiguity in ambiguities:
                    st.warning(ambiguity)

    # Resolution Tab
    if "Resolution" in tab_indices:
        with tabs[tab_indices["Resolution"]]:
            st.markdown("### Resolution Criteria")
            if criteria := decision_log.get("resolution_criteria"):
                for i, criterion in enumerate(criteria, 1):
                    st.markdown(f"**{i}.** {criterion}")
                    st.divider()

    # Chosen Path Tab
    if "Selected Path" in tab_indices:
        with tabs[tab_indices["Selected Path"]]:
            st.markdown("### Selected Tables and Columns")
            if chosen_path := decision_log.get("chosen_path_explanation"):
                for item in chosen_path:
                    with st.expander(f"{item['table']}", expanded=True):
                        st.markdown("#### Selected Columns:")
                        cols = st.columns(min(3, len(item['columns'])))
                        for i, col in enumerate(item['columns']):
                            with cols[i % len(cols)]:
                                st.code(col)

                        st.markdown("#### Selection Rationale:")
                        st.info(item['reason'])

    # SQL Query Tab
    if "SQL Query" in tab_indices:
        with tabs[tab_indices["SQL Query"]]:
            st.markdown("### Generated SQL Query")
            if sql_query := decision_log.get("generated_sql_query"):
                st.code(sql_query, language="sql")
                if st.button("Copy Query"):
                    st.write("Query copied to clipboard!")
                    st.session_state['clipboard'] = sql_query

    # Alternatives Tab
    if "Alternatives" in tab_indices:
        with tabs[tab_indices["Alternatives"]]:
            st.markdown("### Alternative Approaches")
            if alternatives := decision_log.get("alternative_paths"):
                for i, alt in enumerate(alternatives, 1):
                    with st.expander(f"Alternative {i}", expanded=False):
                        st.markdown(alt)

    # Feedback Tab
    if "Feedback" in tab_indices:
        with tabs[tab_indices["Feedback"]]:
            st.markdown("### Execution Feedback")
            if feedback := decision_log.get("execution_feedback"):
                for item in feedback:
                    if "error" in item.lower():
                        st.error(item)
                    elif "warning" in item.lower():
                        st.warning(item)
                    else:
                        st.success(item)

    # Summary Tab
    if "Summary" in tab_indices:
        with tabs[tab_indices["Summary"]]:
            st.markdown("### Analysis Summary")
            if summary := decision_log.get("final_summary"):
                st.markdown(f"**Key Findings:**")
                st.markdown(f"_{summary}_")

                if viz_suggestion := decision_log.get("visualization_suggestion"):
                    st.divider()
                    st.markdown("### Visualization Recommendation")
                    st.success(f"Suggested visualization type: **{viz_suggestion}**")
                    st.markdown("_This chart type was selected based on the data structure and analysis goals._")

def analyze_query_performance(query: str, execution_time: float, row_count: int) -> dict:
    """Analyze query performance and suggest optimizations."""
    performance_metrics = {
        "execution_time": execution_time,
        "rows_returned": row_count,
        "rows_per_second": row_count / execution_time if execution_time > 0 else 0,
        "suggestions": []
    }

    # Performance classification
    if execution_time < 0.1:
        performance_metrics["performance_class"] = "Excellent"
    elif execution_time < 0.5:
        performance_metrics["performance_class"] = "Good"
    elif execution_time < 2.0:
        performance_metrics["performance_class"] = "Fair"
    else:
        performance_metrics["performance_class"] = "Poor"
        performance_metrics["suggestions"].append({
            "type": "warning",
            "message": f"Query execution time ({execution_time:.2f}s) is high"
        })

    return performance_metrics

# Database Setup
db_type = st.sidebar.selectbox("Select Database Type 🗄️", options=["SQLite", "PostgreSQL"])

if db_type == "SQLite":
    uploaded_file = st.sidebar.file_uploader("Upload SQLite Database 📂", type=["db", "sqlite", "sql"])

    if uploaded_file:
        db_file = save_temp_file(uploaded_file)
        schemas = DB_Config.get_all_schemas(db_file, db_type='sqlite')
        table_names = list(schemas.keys())

        if not schemas:
            st.error("Could not load any schemas please check the database file")

        if table_names:
            options = ["Select All"] + table_names
            selected_tables = st.sidebar.multiselect("Select Tables 📋", options=options, key="sqlite_tables")
            if "Select All" in selected_tables:
                if len(selected_tables) < len(options):
                    selected_tables = table_names
                else:
                    selected_tables = options
            selected_tables = [table for table in selected_tables if table != "Select All"]
            colored_header(f"🔍 Selected Tables: {', '.join(selected_tables)}", color_name="blue-70", description="")
            if len(selected_tables) > 3:

                    with st.expander("View All Table Schemas 📖", expanded=False):
                        for table in selected_tables:
                            with st.expander(f"Schema: {table}", expanded=False):
                                st.json(schemas[table])
            else:
                for table in selected_tables:
                    with st.expander(f"View Schema: {table} 📖", expanded=False):
                        st.json(schemas[table])

            user_message = st.text_input(placeholder="Type your SQL query here...", key="user_message", label="Your Query 💬", label_visibility="hidden")
            if user_message:
                selected_schemas = {table: schemas[table] for table in selected_tables}
                logger.debug(f"Schemas being passed to `generate_sql_query`: {selected_schemas}")
                with st.spinner('🧠 Generating SQL query...'):
                    response = generate_sql_query(user_message, selected_schemas)
                handle_query_response(response, db_file, db_type='sqlite')

        else:
            st.info("📭 No tables found in the database.")
    else:
        st.info("📥 Please upload a database file to start.")

elif db_type == "PostgreSQL":
    with st.sidebar.expander("🔐 PostgreSQL Connection Details", expanded=True):
        postgres_host = st.text_input("Host 🏠", placeholder="PostgreSQL Host")
        postgres_db = st.text_input("DB Name 🗄️", placeholder="Database Name")
        postgres_user = st.text_input("Username 👤", placeholder="Username")
        postgres_password = st.text_input("Password 🔑", type="password", placeholder="Password")

    if all([postgres_host, postgres_db, postgres_user, postgres_password]):
        schemas = DB_Config.get_all_schemas(postgres_db, db_type='postgresql', host=postgres_host, user=postgres_user, password=postgres_password)
        table_names = list(schemas.keys())

        if table_names:
            options = ["Select All"] + table_names
            selected_tables = st.sidebar.multiselect("Select Tables 📋", options=options, key="postgresql_tables")
            if "Select All" in selected_tables:
                if len(selected_tables) < len(options):
                    selected_tables = table_names
                else:
                    selected_tables = options
            selected_tables = [table for table in selected_tables if table != "Select All"]
            colored_header("🔍 Selected Tables:", color_name="blue-70", description="")
            for table in selected_tables:
                with st.expander(f"View Schema: {table} 📖", expanded=False):
                    st.json(schemas[table])

            user_message = st.text_input(placeholder="Type your SQL query here...", key="user_message_pg", label="Your Query 💬", label_visibility="hidden")
            if user_message:
                with st.spinner('🧠 Generating SQL query...'):
                    selected_schemas = {table: schemas[table] for table in selected_tables}
                    logger.debug(f"Schemas being passed to `generate_sql_query`: {selected_schemas}")
                    response = generate_sql_query(user_message, selected_schemas)
                handle_query_response(response, postgres_db, db_type='postgresql', host=postgres_host, user=postgres_user, password=postgres_password)
        else:
            st.info("📭 No tables found in the database.")
    else:
        st.info("🔒 Please fill in all PostgreSQL connection details to start.")

# Query history
with st.sidebar.expander(" Query History", expanded=False):
    if st.session_state.get("query_history"):
        st.write("### 📝 Saved Queries")

        search_query = st.text_input("Search Queries 🔍", key="search_query")
        query_history_df = pd.DataFrame({
            "Query": st.session_state.query_history,
            "Timestamp": pd.to_datetime(st.session_state.query_timestamps)
        })

        if search_query:
            query_history_df = query_history_df[query_history_df['Query'].str.contains(search_query, case=False, na=False)]

        queries_per_page = 5
        total_queries = len(query_history_df)
        num_pages = max((total_queries // queries_per_page) + (total_queries % queries_per_page > 0), 1)
        current_page = st.number_input("Page 📄", min_value=1, max_value=num_pages, value=1)

        start_index = (current_page - 1) * queries_per_page
        end_index = start_index + queries_per_page
        page_queries = query_history_df.iloc[start_index:end_index]

        for i, row in page_queries.iterrows():
            with st.expander(f"🗂️ Query {i + 1}: {row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                st.write("**SQL Query:**")
                st.code(row['Query'], language="sql")

                if st.button(f"🔄 Re-run Query {i + 1}", key=f"rerun_query_{i}"):
                    user_message = row['Query']
                    with st.spinner('🔄 Re-running the saved SQL query...'):
                        selected_schemas = {table: schemas[table] for table in selected_tables}
                        response = generate_sql_query(user_message, selected_schemas)
                        handle_query_response(
                            response,
                            db_file if db_type == "SQLite" else postgres_db,
                            db_type,
                            host=postgres_host if db_type == "PostgreSQL" else None,
                            user=postgres_user if db_type == "PostgreSQL" else None,
                            password=postgres_password if db_type == "PostgreSQL" else None
                        )

        st.write(f"Page {current_page} of {num_pages}")

    else:
        st.info("📭 No query history available.")
