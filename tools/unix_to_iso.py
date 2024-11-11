import pandas as pd
from datetime import datetime
import argparse
import os
import sys

def convert_unix_to_iso(unix_timestamp):
    """
    Convert Unix timestamp to ISO 8601 format.
    
    Args:
        unix_timestamp (int or float): Unix timestamp.
        
    Returns:
        str: ISO 8601 formatted date-time string.
    """
    try:
        # If the timestamp is in milliseconds, uncomment the next line
        # unix_timestamp = unix_timestamp / 1000
        return datetime.utcfromtimestamp(unix_timestamp).isoformat() + 'Z'
    except Exception as e:
        print(f"Error converting timestamp {unix_timestamp}: {e}")
        return None

def replace_unix_with_iso(input_file, output_file, timestamp_column):
    """
    Read a CSV file, convert Unix timestamps in a specified column to ISO format,
    and write the result to a new CSV file.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        timestamp_column (str): Name of the column containing Unix timestamps.
    """
    if not os.path.isfile(input_file):
        print(f"Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading '{input_file}': {e}")
        sys.exit(1)
    
    if timestamp_column not in df.columns:
        print(f"Column '{timestamp_column}' not found in the CSV file.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Apply the conversion to the specified column
    df[timestamp_column] = df[timestamp_column].apply(convert_unix_to_iso)
    
    try:
        df.to_csv(output_file, index=False)
        print(f"Successfully wrote the updated CSV to '{output_file}'.")
    except Exception as e:
        print(f"Error writing to '{output_file}': {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Replace Unix timestamps in a CSV with ISO 8601 formatted dates.")
    parser.add_argument('input_file', help='Path to the input CSV file.')
    parser.add_argument('output_file', help='Path to the output CSV file.')
    parser.add_argument('timestamp_column', help='Name of the column containing Unix timestamps.')
    
    args = parser.parse_args()
    
    replace_unix_with_iso(args.input_file, args.output_file, args.timestamp_column)

if __name__ == "__main__":
    main()
