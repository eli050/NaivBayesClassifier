import pandas as pd


class LoadCSV:
    """Class for reading a CSV file and returning it as a DataFrame"""

    def __init__(self, file_path):
        """Initialize with the path to the CSV file"""
        self.file_path = file_path

    def get_data(self):
        """Attempts to read the CSV file and return a DataFrame; prints the error if reading fails"""
        try:
            df = pd.read_csv(self.file_path)
            return df
        except Exception as e:
            print(f"------{e}----------")
            raise

