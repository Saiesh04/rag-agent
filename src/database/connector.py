"""
Database connectivity module for RAG Q&A System
"""
import logging
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
import mysql.connector
import psycopg2
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class DatabaseConnector:
    """Handles database connections and queries for multiple database types"""
    
    def __init__(self):
        self.connection = None
        self.engine = None
        self.db_type = None
        self.connection_params = {}
    
    def connect_sqlite(self, db_path: str) -> bool:
        """Connect to SQLite database"""
        try:
            self.connection = sqlite3.connect(db_path)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            self.db_type = "sqlite"
            self.connection_params = {"db_path": db_path}
            
            # Create SQLAlchemy engine for pandas compatibility
            self.engine = create_engine(f"sqlite:///{db_path}")
            
            logger.info(f"Connected to SQLite database: {db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to SQLite: {str(e)}")
            return False
    
    def connect_mysql(self, host: str, port: int, database: str, username: str, password: str) -> bool:
        """Connect to MySQL database"""
        try:
            self.connection = mysql.connector.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password
            )
            self.db_type = "mysql"
            self.connection_params = {
                "host": host,
                "port": port,
                "database": database,
                "username": username
            }
            
            # Create SQLAlchemy engine
            connection_string = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
            self.engine = create_engine(connection_string)
            
            logger.info(f"Connected to MySQL database: {database}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MySQL: {str(e)}")
            return False
    
    def connect_postgresql(self, host: str, port: int, database: str, username: str, password: str) -> bool:
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password
            )
            self.db_type = "postgresql"
            self.connection_params = {
                "host": host,
                "port": port,
                "database": database,
                "username": username
            }
            
            # Create SQLAlchemy engine
            connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            self.engine = create_engine(connection_string)
            
            logger.info(f"Connected to PostgreSQL database: {database}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {str(e)}")
            return False
    
    def connect_from_url(self, database_url: str) -> bool:
        """Connect using a database URL"""
        try:
            parsed = urlparse(database_url)
            scheme = parsed.scheme.lower()
            
            if scheme == 'sqlite':
                db_path = database_url.replace('sqlite:///', '').replace('sqlite://', '')
                return self.connect_sqlite(db_path)
            elif scheme in ['mysql', 'mysql+mysqlconnector']:
                return self.connect_mysql(
                    host=parsed.hostname,
                    port=parsed.port or 3306,
                    database=parsed.path.lstrip('/'),
                    username=parsed.username,
                    password=parsed.password
                )
            elif scheme in ['postgresql', 'postgres']:
                return self.connect_postgresql(
                    host=parsed.hostname,
                    port=parsed.port or 5432,
                    database=parsed.path.lstrip('/'),
                    username=parsed.username,
                    password=parsed.password
                )
            else:
                logger.error(f"Unsupported database scheme: {scheme}")
                return False
                
        except Exception as e:
            logger.error(f"Error parsing database URL: {str(e)}")
            return False
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the database connection"""
        if not self.connection:
            return {"success": False, "error": "No connection established"}
        
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            elif self.db_type == "mysql":
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            elif self.db_type == "postgresql":
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                
            return {"success": True, "db_type": self.db_type}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_tables(self) -> List[str]:
        """Get list of tables in the database"""
        if not self.engine:
            return []
        
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            return tables
            
        except Exception as e:
            logger.error(f"Error getting tables: {str(e)}")
            return []
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a specific table"""
        if not self.engine:
            return {}
        
        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name)
            
            # Get sample data
            sample_query = f"SELECT * FROM {table_name} LIMIT 5"
            sample_data = pd.read_sql(sample_query, self.engine)
            
            # Get row count
            count_query = f"SELECT COUNT(*) AS count FROM {table_name}"
            count_result = pd.read_sql(count_query, self.engine)
            row_count = count_result['count'].iloc[0]
            
            return {
                "columns": [{"name": col["name"], "type": str(col["type"])} for col in columns],
                "row_count": row_count,
                "sample_data": sample_data.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error getting table info: {str(e)}")
            return {}
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute a SQL query and return results"""
        if not self.engine:
            return {"success": False, "error": "No database connection"}
        
        try:
            # Use pandas for better data handling
            df = pd.read_sql(query, self.engine)
            
            return {
                "success": True,
                "data": df.to_dict('records'),
                "columns": df.columns.tolist(),
                "row_count": len(df),
                "dataframe": df  # Keep DataFrame for analysis
            }
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_database_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the database"""
        if not self.connection:
            return {"success": False, "error": "No connection established"}
        
        try:
            tables = self.get_tables()
            table_summaries = {}
            
            for table in tables:
                table_info = self.get_table_info(table)
                table_summaries[table] = {
                    "columns": len(table_info.get("columns", [])),
                    "rows": table_info.get("row_count", 0),
                    "column_details": table_info.get("columns", [])
                }
            
            return {
                "success": True,
                "db_type": self.db_type,
                "connection_params": {k: v for k, v in self.connection_params.items() if k != "password"},
                "total_tables": len(tables),
                "tables": table_summaries
            }
            
        except Exception as e:
            logger.error(f"Error getting database summary: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def close_connection(self):
        """Close the database connection"""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                
            if self.engine:
                self.engine.dispose()
                self.engine = None
                
            self.db_type = None
            self.connection_params = {}
            
            logger.info("Database connection closed")
            
        except Exception as e:
            logger.error(f"Error closing connection: {str(e)}")
    
    def is_connected(self) -> bool:
        """Check if database is connected"""
        if not self.connection:
            return False
        
        test_result = self.test_connection()
        return test_result.get("success", False)
    
    def generate_natural_language_query(self, question: str, context: str = "") -> str:
        """Generate SQL query from natural language (basic implementation)"""
        # This is a simplified implementation
        # In production, you might want to use a more sophisticated NL-to-SQL model
        
        question_lower = question.lower()
        
        # Basic keyword mapping
        if "count" in question_lower or "how many" in question_lower:
            return "SELECT COUNT(*) FROM [TABLE_NAME]"
        elif "all" in question_lower or "list" in question_lower:
            return "SELECT * FROM [TABLE_NAME] LIMIT 100"
        elif "average" in question_lower or "avg" in question_lower:
            return "SELECT AVG([COLUMN_NAME]) FROM [TABLE_NAME]"
        elif "sum" in question_lower or "total" in question_lower:
            return "SELECT SUM([COLUMN_NAME]) FROM [TABLE_NAME]"
        elif "max" in question_lower or "maximum" in question_lower:
            return "SELECT MAX([COLUMN_NAME]) FROM [TABLE_NAME]"
        elif "min" in question_lower or "minimum" in question_lower:
            return "SELECT MIN([COLUMN_NAME]) FROM [TABLE_NAME]"
        else:
            return "SELECT * FROM [TABLE_NAME] WHERE [CONDITION] LIMIT 100"