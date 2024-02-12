from sqlalchemy import *
from sqlalchemy.schema import *
from sqlalchemy.engine import create_engine
from google.cloud import bigquery
import datetime
import pandas as pd
import pandas_gbq
import logging
import string
import re
from tqdm import tqdm
from modules.validation_module import DataValidator

logging.basicConfig(level=logging.INFO)  # Set logging level

class DqEngine:
    def __init__(self, engine: str):
        self.engine = engine

    def perform_eda(self, query: str=None, df=None, cloud=True):
        """Processes data based on a query and writes results to BigQuery."""

        try:
            if query is not None:
                dataframe = pd.read_sql_query(query, self.engine)
            elif df is not None:
                dataframe = df.copy()
            else:
                logging.error('No data loaded')
                raise ValueError('No data loaded')

            for index, row in tqdm(dataframe.iterrows(),total=dataframe.shape[0]):
                schema = row['input_schema']
                table = row['table_name']
                uids = row['uids'].split(",")
                target_attribute = row['target_attribute']
                kwargs = self.process_args(row['parameters'])
                error_id = row['error_id']
                error_code = row['error_code']
                function_name = row['function_name']
                output_schema = row['output_schema']

                logging.info(f"\n**Validating attribute: {target_attribute}**\n**Running check: {function_name}**")

                data_validator = DataValidator(self.engine, schema, table)
                function_map = {function_name: getattr(data_validator, function_name)}
                
                if kwargs is not None:
                    result_df = function_map[function_name](uids, target_attribute, **kwargs)
                else:
                    result_df = function_map[function_name](uids, target_attribute)

                if result_df.empty:
                    pass
                else:
                    result_df['TARGET_ATTRIBUTE_model_value'] = None
                    result_df['TARGET_ATTRIBUTE_confidence_score'] = 1.0
                    result_df['TARGET_ATTRIBUTE_source_attributes'] = None
                    result_df = self.write_dqresults(result_df, uids, target_attribute, function_name, error_id, error_code, output_schema, cloud=cloud)
            if cloud==False:
                return result_df

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e  # Re-raise the exception to propagate it

    def process_args(self, args_str):
        """Processes the args string into a list of values or None."""
        if args_str is not None:
            return {k.strip():v.strip() for k, v in (item.split('=') for item in args_str.split('|'))}
        else:
            return {}

    def write_dqresults(self, result_df, uids, target_attribute, function_name, error_id, error_code, output_schema, cloud=True):
        """Writes results to a BigQuery table."""
        
        result_df['BATCH_ID'] = f'{target_attribute}_{function_name}_'+''.join([str(i) for i in datetime.datetime.utcnow().timetuple()][0:6])
        result_df['TARGET_ATTRIBUTE'] = f'{target_attribute}'
        result_df['TARGET_ATTRIBUTE_error_id'] = error_id
        result_df['TARGET_ATTRIBUTE_error_codes'] = error_code
        
        result_df = result_df[['BATCH_ID']+
                            [i for i in uids]+
                            ['TARGET_ATTRIBUTE',
                            'TARGET_ATTRIBUTE_catalog_value',
                            'TARGET_ATTRIBUTE_model_value',
                            'TARGET_ATTRIBUTE_error_id',
                            'TARGET_ATTRIBUTE_error_codes',
                            'TARGET_ATTRIBUTE_confidence_score',
                            'TARGET_ATTRIBUTE_source_attributes'
                           ]]
        
        type_dict = {'BATCH_ID': 'string',
                    'TARGET_ATTRIBUTE': 'string',                    
                    'TARGET_ATTRIBUTE_catalog_value': 'string',
                    'TARGET_ATTRIBUTE_model_value': 'string',                    
                    'TARGET_ATTRIBUTE_error_id': 'int64',
                    'TARGET_ATTRIBUTE_error_codes': 'string',
                    'TARGET_ATTRIBUTE_confidence_score': 'float64',
                    'TARGET_ATTRIBUTE_source_attributes': 'string'}

        uid_dict = {i: 'string' for i in uids}
        type_dict.update(uid_dict)
        result_df = result_df.astype(type_dict)
        result_df= result_df.reset_index(drop=True)
        if cloud==False:
            return result_df
        else:
            logging.info(f"  **Writing results to BigQuery table: {result_df['BATCH_ID'][0]}**")
            project_id = str(self.engine).split('//')[1].strip('([])')
            pandas_gbq.to_gbq(result_df, f"{output_schema}.{result_df['BATCH_ID'][0]}", project_id=project_id,if_exists='replace')
            return None
    

    def clean_and_compare_columns(self, df, original, new):
        if pd.api.types.is_numeric_dtype(df[original]):
            df = df[~df[original].isna()]  # Handle numeric dtypes
        else:
            df = df[df[original].notna()]  # Handle non-numeric dtypes

        if df[original].dtype == object and df[new].dtype == object:
            # Create temporary columns within the conditional block
            df['modified_original_col'] = df[original].apply(lambda x: ''.join(e for e in str(x) if e.isalnum()).strip().lower())
            df['modified_new_col'] = df[new].apply(lambda x: ''.join(e for e in str(x) if e.isalnum()).strip().lower())

            filtered = df[df['modified_original_col'] != df['modified_new_col']]
            filtered = filtered.drop(columns=['modified_original_col', 'modified_new_col'])
            return filtered
        else:
            return df[df[original] != df[new]]

    def cleanse_duplicate_asins_using_bq(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Removes rows from a DataFrame where 'asin' values are present in a BigQuery table,
        excluding duplicates from the retrieved asins.

        If the table doesn't exist, returns the original DataFrame without filtering.

        Args:
            df (pd.DataFrame): The DataFrame to filter.
            query (str): The SQL query to fetch data from BigQuery.

        Returns:
            pd.DataFrame: The filtered DataFrame, or the original DataFrame if the table doesn't exist.
        """

        try:
            asins_to_exclude = pd.read_sql_query(query, self.engine)['asin'].unique()
            return df[~df['asin'].isin(asins_to_exclude)].reset_index(drop=True)
            
        except exc.DatabaseError as e: 
            print(f"Table not found in BigQuery. Returning original DataFrame.")
            return df  # Return original DataFrame if table doesn't exist
