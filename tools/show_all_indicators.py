import sys
import os

def get_script_directory():
    """
    Returns the directory where the current script is located.
    """
    return os.path.dirname(os.path.abspath(__file__))

def write_to_file(file_path, content):
    """
    Writes the given content to the specified file path.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Indicators successfully written to {file_path}\n")
    except Exception as e:
        print(f"Failed to write to file {file_path}: {e}\n")

def print_pandas_ta_indicators():
    """
    Retrieves all available indicators in pandas_ta using multiple methods.
    Returns a string containing the list of indicators.
    """
    indicators_output = ""
    try:
        import pandas_ta as ta
    except ImportError as e:
        indicators_output += f"Error importing pandas_ta: {e}\nPlease ensure pandas_ta is installed and updated correctly.\n\n"
        return indicators_output

    # Check pandas_ta version
    try:
        version = ta.__version__
        indicators_output += f"pandas_ta version: {version}\n\n"
    except AttributeError:
        indicators_output += "Could not determine pandas_ta version.\n\n"

    # Method 1: Using ta.indicators() method
    try:
        indicators = ta.indicators()
        indicators_output += "=== pandas_ta Indicators (Method 1: ta.indicators()) ===\n"
        for indicator in indicators:
            indicators_output += f"{indicator}\n"
        indicators_output += f"\nTotal pandas_ta indicators: {len(indicators)}\n\n"
        return indicators_output  # Exit after successful method
    except AttributeError:
        indicators_output += "Method 1 Failed: 'indicators()' method not found in pandas_ta.\n"
    except Exception as e:
        indicators_output += f"Method 1 Failed: An unexpected error occurred: {e}\n"

    # Method 2: Using ta.indicators attribute
    try:
        indicators = ta.indicators
        if isinstance(indicators, list):
            indicators_output += "=== pandas_ta Indicators (Method 2: ta.indicators) ===\n"
            for indicator in indicators:
                indicators_output += f"{indicator}\n"
            indicators_output += f"\nTotal pandas_ta indicators: {len(indicators)}\n\n"
            return indicators_output  # Exit after successful method
    except AttributeError:
        indicators_output += "Method 2 Failed: 'indicators' attribute not found in pandas_ta.\n"
    except Exception as e:
        indicators_output += f"Method 2 Failed: An unexpected error occurred: {e}\n"

    # Method 3: Inspecting callable functions
    try:
        import inspect
        attributes = dir(ta)
        # Filter callable functions that are likely indicators
        indicators = [
            attr for attr in attributes
            if callable(getattr(ta, attr)) and not attr.startswith('_')
        ]
        indicators_output += "=== pandas_ta Indicators (Method 3: Inspect Callables) ===\n"
        for indicator in indicators:
            indicators_output += f"{indicator}\n"
        indicators_output += f"\nTotal pandas_ta indicators (callables): {len(indicators)}\n\n"
    except Exception as e:
        indicators_output += f"Method 3 Failed: An unexpected error occurred: {e}\n"

    return indicators_output

def print_talib_indicators():
    """
    Retrieves all available indicators in TA-Lib.
    Returns a string containing the list of indicators.
    """
    indicators_output = ""
    try:
        import talib
    except ImportError as e:
        indicators_output += f"Error importing TA-Lib: {e}\nPlease ensure TA-Lib is installed correctly.\n\n"
        return indicators_output

    try:
        indicators = talib.get_functions()
        indicators_output += "=== TA-Lib Indicators ===\n"
        for indicator in indicators:
            indicators_output += f"{indicator}\n"
        indicators_output += f"\nTotal TA-Lib indicators: {len(indicators)}\n\n"
    except Exception as e:
        indicators_output += f"An unexpected error occurred with TA-Lib: {e}\n\n"

    return indicators_output

def main():
    """
    Main function to retrieve indicators from pandas_ta and TA-Lib,
    and write them to a text file.
    """
    indicators_content = ""

    # Get pandas_ta indicators
    indicators_content += print_pandas_ta_indicators()

    # Get TA-Lib indicators
    indicators_content += print_talib_indicators()

    # Define the output file path
    script_dir = get_script_directory()
    output_file = os.path.join(script_dir, "indicators.txt")

    # Write the indicators to the file
    write_to_file(output_file, indicators_content)

if __name__ == "__main__":
    main()
