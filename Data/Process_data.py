import json
import os

class Process_data:
    Data = [map]
    data_path: str = ""
    
    def getData(self):
        """
        Method to read JSON array data (proper JSON format)
        """
        try:
            if not self.data_path:
                raise ValueError("data_path is not specified")

            # Get absolute path and fix backslashes for Windows
            absolute_path = os.path.abspath(self.data_path)
            print(f"Reading from: {absolute_path}")

            # Check if file exists
            if not os.path.exists(absolute_path):
                raise FileNotFoundError(f"File not found: {absolute_path}")

            with open(absolute_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)  # Parse the entire JSON file
            
            # If it's a list, use it directly
            if isinstance(json_data, list):
                self.Data = json_data
            else:
                # If it's a single object, wrap it in a list
                self.Data = [json_data]
                
            print(f"Successfully loaded {len(self.Data)} documents")

        except FileNotFoundError:
            print(f"Error: File not found at path '{self.data_path}'")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in file '{self.data_path}': {str(e)}")
        except Exception as e:
            print(f"Error reading data: {str(e)}")

# Usage with raw string to handle Windows backslashes
preprocess = Process_data()
preprocess.data_path = r"C:\Users\farah\OneDrive\Documents\KG\knowledge-graph-llms\Data\news_data_docs.json"
preprocess.getData()
Data=preprocess.Data
print(f"Total documents: {len(preprocess.Data)}")
if preprocess.Data:
    print("First document:", preprocess.Data[0]['tags_split'][0])