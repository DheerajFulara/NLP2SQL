import sqlite3
from typing import Optional, Dict, Any, Union, List
import psycopg2
from psycopg2 import OperationalError, sql
import pandas as pd
import logging
import json
from contextlib import contextmanager
from abc import ABC, abstractmethod
import psycopg2.pool
import re

# Configure logging for improved debug and info messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store PostgreSQL connection pool
pg_pool = None

def create_pg_pool(minconn: int, maxconn: int, dbname: str, user: str,
                   password: str, host: str) -> None:
    """
    Creates or reinitializes a PostgreSQL connection pool to manage database connections.

    :param minconn: Minimum number of connections in the pool.
    :param maxconn: Maximum number of connections in the pool.
    :param dbname:  Name of the database.
    :param user:    Username for the database.
    :param password:Password for the database.
    :param host:    Host address for the database.
    """
    global pg_pool
    try:
        pg_pool = psycopg2.pool.SimpleConnectionPool(
            minconn,
            maxconn,
            dbname=dbname,
            user=user,
            password=password,
            host=host
        )
        if pg_pool:
            logger.info("PostgreSQL connection pool created successfully.")
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL connection pool: {e}")

@contextmanager
def get_pg_connection():
    """
    Context manager for obtaining a connection from the global PostgreSQL connection pool.
    Safely yields a connection and returns it back to the pool.
    """
    conn = None
    try:
        conn = pg_pool.getconn()
        yield conn
    except Exception as e:
        logger.error(f"Error while getting PostgreSQL connection: {e}")
        raise
    finally:
        if conn:
            pg_pool.putconn(conn)

@contextmanager
def get_connection(
    db_name: str,
    db_type: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None
):
    """
    Context manager for creating and managing database connections.

    :param db_name: Name of the database.
    :param db_type: Type of the database ('sqlite' or 'postgresql').
    :param host:    Host address for PostgreSQL.
    :param user:    Username for PostgreSQL.
    :param password:Password for PostgreSQL.

    :yields: A database connection object.
    """
    conn = None
    try:
        if db_type.lower() == 'postgresql':
            # Create pool if not already created
            if not pg_pool:
                create_pg_pool(
                    minconn=1,
                    maxconn=10,
                    dbname=db_name,
                    user=user if user else '',
                    password=password if password else '',
                    host=host if host else 'localhost'
                )
            conn = pg_pool.getconn()
            logger.info("Connected to PostgreSQL database using connection pool.")
        elif db_type.lower() == 'sqlite':
            conn = sqlite3.connect(db_name)
            logger.info("Connected to SQLite database.")
        else:
            logger.error(f"Unsupported database type: {db_type}")
            yield None
            return
        yield conn
    except OperationalError as e:
        logger.error(f"Operational error while connecting to the database: {e}")
        yield None
    except Exception as e:
        logger.exception(f"Unexpected error while connecting to the database: {e}")
        yield None
    finally:
        # Safely close the connection if it's not None
        if conn:
            # If we used pg_pool, we should not close directly, but release the connection
            if db_type.lower() == 'postgresql' and pg_pool:
                pg_pool.putconn(conn)
                logger.info("PostgreSQL connection returned to pool.")
            else:
                conn.close()
                logger.info("SQLite connection closed.")

def query_database(
    query: str,
    db_name: str,
    db_type: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    limit: int = None,
    offset: int = 0
) -> pd.DataFrame:
    """
    Executes an SQL query on the specified database and returns the results as a Pandas DataFrame.
    If a limit is provided and the query is a SELECT without a LIMIT clause,
    automatically appends the LIMIT and OFFSET.

    :param query:   SQL query string to execute.
    :param db_name: Database name.
    :param db_type: Database type ('sqlite' or 'postgresql').
    :param host:    Database host (PostgreSQL only).
    :param user:    Database user (PostgreSQL only).
    :param password:Database password (PostgreSQL only).
    :param limit:   Maximum number of rows to return.
    :param offset:  Number of rows to skip before starting to return rows.

    :return:        A Pandas DataFrame containing the query results, or an empty DataFrame on error.
    """
    with get_connection(db_name, db_type, host, user, password) as conn:
        if conn is None:
            logger.error("Database connection failed. Returning empty DataFrame.")
            return pd.DataFrame()

        modified_query = query
        if db_type.lower() in ['sqlite', 'postgresql'] and query.strip().lower().startswith('select'):
            if limit is not None and "limit" not in query.lower():
                modified_query = f"{query.rstrip(';')} LIMIT {limit} OFFSET {offset};"
                logger.warning("Query truncated with LIMIT 100 for performance. Use pagination for full results.")
        try:
            df = pd.read_sql_query(modified_query, conn)
            logger.info("Query executed successfully.")
            return df
        except Exception as e:
            logger.exception(f"Unexpected error executing query: {e}")
            return pd.DataFrame()

# --- Abstract Schema Extractor Class ---

class SchemaExtractor(ABC):
    """
    Abstract base class for extracting schema information from a database.

    Implementing classes must provide methods to get all table names and
    detailed schema info for each table.
    """
    def __init__(self, connection):
        self.conn = connection

    @abstractmethod
    def get_tables(self) -> List[str]:
        """
        Returns a list of all table names in the database schema.
        """
        pass

    @abstractmethod
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Returns a dictionary describing the schema of the specified table.
        """
        pass

# --- SQLite Schema Extraction ---

def get_sqlite_table_info(cursor, table_name: str) -> Dict[str, Any]:
    """
    Retrieve detailed schema information for a given SQLite table, including columns,
    primary keys, foreign keys, indexes, triggers, and sample data.

    :param cursor:    SQLite cursor object.
    :param table_name:Name of the table to extract schema information.
    :return:          Dictionary containing table schema details.
    """
    table_info = {
        'columns': {},
        'foreign_keys': [],
        'indexes': [],
        'sample_data': [],
        'primary_keys': [],
        'constraints': [],
        'triggers': []
    }

    # Fetch column metadata
    cursor.execute(f"PRAGMA table_info('{table_name}');")
    columns = cursor.fetchall()
    for col in columns:
        col_id, col_name, col_type, not_null, default_val, pk = col
        table_info['columns'][col_name] = {
            'type': col_type,
            'nullable': (not not_null),
            'default': default_val,
            'primary_key': bool(pk)
        }
        if pk:
            table_info['primary_keys'].append(col_name)

    # Fetch foreign key constraints
    cursor.execute(f"PRAGMA foreign_key_list('{table_name}');")
    fkeys = cursor.fetchall()
    for fk in fkeys:
        # columns: (id, seq, table, from, to, on_update, on_delete, match)
        _, _, ref_table, from_col, to_col, on_update, on_delete, _ = fk
        table_info['foreign_keys'].append({
            'from_column': from_col,
            'to_table': ref_table,
            'to_column': to_col,
            'on_update': on_update,
            'on_delete': on_delete
        })

    # Fetch indexes
    cursor.execute(f"PRAGMA index_list('{table_name}');")
    indexes = cursor.fetchall()
    for idx in indexes:
        idx_id, idx_name, unique_flag, _ = idx[0], idx[1], idx[2], idx[3] if len(idx) > 3 else None
        cursor.execute(f"PRAGMA index_info('{idx_name}');")
        index_columns = cursor.fetchall()
        table_info['indexes'].append({
            'name': idx_name,
            'unique': bool(unique_flag),
            'columns': [col[2] for col in index_columns]
        })

    # Fetch sample data
    try:
        cursor.execute(f"SELECT * FROM '{table_name}' LIMIT 5;")
        rows = cursor.fetchall()
        if rows:
            column_names = [desc[0] for desc in cursor.description]
            table_info['sample_data'] = [dict(zip(column_names, row)) for row in rows]
    except Exception as e:
        logger.warning(f"Unable to retrieve sample data for table {table_name}: {e}")

    # SQLite triggers
    # There's no simple PRAGMA for triggers, so we can check sqlite_master
    cursor.execute(f"SELECT name, sql FROM sqlite_master WHERE type='trigger' AND tbl_name='{table_name}';")
    trigger_rows = cursor.fetchall()
    for tr in trigger_rows:
        tr_name, tr_sql = tr
        table_info['triggers'].append({
            'name': tr_name,
            'definition': tr_sql
        })

    return table_info

class SQLiteSchemaExtractor(SchemaExtractor):
    """
    Extracts schema information from a SQLite database.
    """
    def get_tables(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        return tables

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        return get_sqlite_table_info(cursor, table_name)

# --- PostgreSQL Schema Extraction ---

def get_postgresql_table_info(cursor, table_name: str) -> Dict[str, Any]:
    """
    Retrieve detailed schema information for a given PostgreSQL table, including columns,
    primary keys, foreign keys, indexes, triggers, and sample data.

    :param cursor:    PostgreSQL cursor object.
    :param table_name:Name of the table to extract schema information.
    :return:          Dictionary containing table schema details.
    """
    table_info = {
        'columns': {},
        'foreign_keys': [],
        'indexes': [],
        'sample_data': [],
        'primary_keys': [],
        'constraints': [],
        'triggers': []
    }

    # Get column information
    cursor.execute(
        """
        SELECT
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length
        FROM information_schema.columns
        WHERE table_name = %s AND table_schema = 'public';
        """,
        [table_name]
    )
    columns = cursor.fetchall()
    for col_name, data_type, is_nullable, default_val, char_len in columns:
        table_info['columns'][col_name] = {
            'type': data_type,
            'nullable': (is_nullable.upper() == 'YES'),
            'default': default_val,
            'max_length': char_len,
            'primary_key': False
        }

    # Primary key info
    cursor.execute(
        """
        SELECT kcu.column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_name = %s
          AND tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema = 'public';
        """,
        [table_name]
    )
    pk_columns = [row[0] for row in cursor.fetchall()]
    for pk_col in pk_columns:
        if pk_col in table_info['columns']:
            table_info['columns'][pk_col]['primary_key'] = True
            table_info['primary_keys'].append(pk_col)

    # Foreign key info
    cursor.execute(
        """
        SELECT
            kcu.column_name AS from_column,
            ccu.table_name AS to_table,
            ccu.column_name AS to_column,
            rc.update_rule AS on_update,
            rc.delete_rule AS on_delete
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
          ON tc.constraint_name = ccu.constraint_name
        JOIN information_schema.referential_constraints AS rc
          ON tc.constraint_name = rc.constraint_name
        WHERE tc.table_name = %s
          AND tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = 'public';
        """,
        [table_name]
    )
    fkeys = cursor.fetchall()
    for from_col, to_table, to_col, on_update, on_delete in fkeys:
        table_info['foreign_keys'].append({
            'from_column': from_col,
            'to_table': to_table,
            'to_column': to_col,
            'on_update': on_update,
            'on_delete': on_delete
        })

    # Indexes
    cursor.execute(
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = 'public' AND tablename = %s;
        """,
        [table_name]
    )
    indexes = cursor.fetchall()
    for idx_name, idx_def in indexes:
        # Attempt to parse the columns from the index definition
        idx_columns = []
        try:
            # Typically looks like: CREATE [UNIQUE] INDEX indexname ON tablename USING btree (col1, col2)
            start = idx_def.index('(')
            end = idx_def.rindex(')')
            cols_part = idx_def[start + 1:end]
            idx_columns = [c.strip() for c in cols_part.split(',')]
        except ValueError:
            pass
        is_unique = 'UNIQUE' in idx_def.upper()
        table_info['indexes'].append({
            'name': idx_name,
            'unique': is_unique,
            'columns': idx_columns
        })

    # Triggers
    # Query pg_trigger joined with pg_class to get triggers for the table
    cursor.execute(
        """
        SELECT tgname, pg_get_triggerdef(t.oid)
        FROM pg_trigger t
        JOIN pg_class c ON t.tgrelid = c.oid
        WHERE c.relname = %s
          AND NOT t.tgisinternal;
        """,
        [table_name]
    )
    triggers = cursor.fetchall()
    for tr_name, tr_def in triggers:
        table_info['triggers'].append({
            'name': tr_name,
            'definition': tr_def
        })

    # Sample data
    try:
        cursor.execute(sql.SQL("SELECT * FROM {} LIMIT 5;").format(sql.Identifier(table_name)))
        sample_data = cursor.fetchall()
        if sample_data:
            column_names = [desc[0] for desc in cursor.description]
            table_info['sample_data'] = [dict(zip(column_names, row)) for row in sample_data]
    except Exception as e:
        logger.warning(f"Unable to retrieve sample data for table {table_name}: {e}")

    return table_info

class PostgreSQLSchemaExtractor(SchemaExtractor):
    """
    Extracts schema information from a PostgreSQL database.
    """
    def get_tables(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public';
            """
        )
        return [row[0] for row in cursor.fetchall()]

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        return get_postgresql_table_info(cursor, table_name)

# --- Utility for generating a structured JSON string (optional) ---
def generate_json_schema(table_name: str, table_info: Dict[str, Any]) -> str:
    """
    Generates a JSON representation of a single table's schema.

    :param table_name: Name of the table.
    :param table_info: A dictionary containing the table's schema details.
    :return:           A JSON string with the formatted schema.
    """
    schema = {
        "object": table_name,
        "columns": table_info.get('columns', {}),
        "primary_keys": table_info.get('primary_keys', []),
        "foreign_keys": table_info.get('foreign_keys', []),
        "indexes": table_info.get('indexes', []),
        "triggers": table_info.get('triggers', []),
        "constraints": table_info.get('constraints', []),
        "sample_data": table_info.get('sample_data', [])
    }
    return json.dumps(schema, indent=2)

# --- Main entry point for schema retrieval ---
def get_all_schemas(
    db_name: str,
    db_type: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Retrieves schema information for all tables in the given database and returns a nested dictionary.
    Each table name maps to a dictionary of schema components.

    :param db_name:  Name of the database.
    :param db_type:  Type of the database ('sqlite' or 'postgresql').
    :param host:     Host address (PostgreSQL only).
    :param user:     User name (PostgreSQL only).
    :param password: Password (PostgreSQL only).

    :return: A dictionary keyed by table name, where each value is a dictionary containing schema details.
    """
    schemas = {}

    with get_connection(db_name, db_type, host, user, password) as conn:
        if not conn:
            logger.error("Database connection failed. Returning empty schema.")
            return {}

        # Decide which extractor class to use
        if db_type.lower() == 'sqlite':
            extractor = SQLiteSchemaExtractor(conn)
        elif db_type.lower() == 'postgresql':
            extractor = PostgreSQLSchemaExtractor(conn)
        else:
            logger.error(f"Unsupported database type: {db_type}")
            return {}

        # Iterate through all tables and retrieve schema info
        for table in extractor.get_tables():
            schemas[table] = extractor.get_table_info(table)

    return schemas
