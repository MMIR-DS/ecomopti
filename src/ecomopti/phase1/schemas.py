# src/ecomopti/phase1/schemas.py
"""
Phase 1 Data Schemas: Validate raw input and processed output.

Pandera provides runtime validation to catch data drift early.
Using lazy=True reports ALL schema violations at once (better for debugging
than failing on first error).
"""
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

def validate_cleaned(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate raw telco.csv AFTER cleaning (TotalCharges is dropped, E/T created).
    
    Raises:
        SchemaErrors: If data doesn't match expected structure
        
    Returns:
        Validated DataFrame (unchanged if valid)
    """
    schema = DataFrameSchema({
        "customerID": Column(str, nullable=False),
        "gender": Column(str, nullable=False),
        "SeniorCitizen": Column(int, Check.isin([0, 1]), nullable=False),
        "Partner": Column(str, nullable=False),
        "Dependents": Column(str, nullable=False),
        "tenure": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "PhoneService": Column(str, nullable=False),
        "MultipleLines": Column(str, nullable=True),
        "InternetService": Column(str, nullable=False),
        "OnlineSecurity": Column(str, nullable=True),
        "OnlineBackup": Column(str, nullable=True),
        "DeviceProtection": Column(str, nullable=True),
        "TechSupport": Column(str, nullable=True),
        "StreamingTV": Column(str, nullable=True),
        "StreamingMovies": Column(str, nullable=True),
        "Contract": Column(str, nullable=False),
        "PaperlessBilling": Column(str, nullable=False),
        "PaymentMethod": Column(str, nullable=False),
        "MonthlyCharges": Column(float, nullable=False),
        "Churn": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "E": Column(int, Check.isin([0, 1]), nullable=False),
        "T": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
    }, strict="filter")  # Allow extra columns if they exist (forward compatibility)
    
    return schema.validate(df, lazy=True)

def validate_processed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate processed split after one-hot encoding.
    
    Ensures metadata columns (E, T, customerID) are present and all 
    other columns are numeric (encoded features).
    """
    schema = DataFrameSchema({
        "E": Column(int, Check.isin([0, 1]), nullable=False),
        "T": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "customerID": Column(str, nullable=False),
        # All other columns are numeric/binary features from one-hot encoding
    }, strict="filter")  # Allow encoded feature columns (unknown count)
    
    return schema.validate(df, lazy=True)