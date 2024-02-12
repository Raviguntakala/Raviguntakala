from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
import re
import string
import pandas as pd
import ast

class DataValidator:
    def __init__(self, engine, schema_name, table_name):
        """
        Initializes the DataValidator with connection details and validation target.

        Parameters:
            engine (sqlalchemy.engine.Engine): The SQLAlchemy engine for BigQuery.
            schema_name (str): The schema name of the table.
            table_name (str): The name of the table to validate.
            column (str): The name of the column to validate.
        """
        self.engine = engine
        self.schema_name = schema_name
        self.table_name = table_name
    

    def is_complete(self, uids, target_attribute, required=None):
        """
        Retrieve rows where the specified column is NULL.

        Parameters:
        - uids (list): The list of unique identifiers.
        - valid_list: The set of valid values for the specified column.
        - required (str): column identifying where attribute is 'required'. If set, only retrieve rows where target_attribute is NULL and attribute is 'required'.

        Returns:
        - pd.DataFrame: DataFrame containing rows where the specified column is NULL.
        """
        if self.engine: 
            try: 
                if required!=None:
                    data_frame = pd.read_sql_query(f"SELECT {','.join(uids)},{target_attribute} AS TARGET_ATTRIBUTE_catalog_value \
                                                    FROM {self.schema_name}.{self.table_name} \
                                                    where ({target_attribute} IS NULL) AND LOWER({required})='required';",self.engine)
                else :
                    data_frame = pd.read_sql_query(f"SELECT {','.join(uids)},{target_attribute} AS TARGET_ATTRIBUTE_catalog_value \
                                                    FROM {self.schema_name}.{self.table_name} \
                                                    where ({target_attribute} IS NULL);",self.engine)
                return data_frame     
            except Exception as e: 
                    print(f"Failed with error: {str(e)}")
                    return { 
                      'status': False, 
                       'err_msg': str(e)}  
        else:
            print('BigQuery Connection Issue: Engine is not available.')
            return { 
                  'status': False, 
                   'err_msg':'BigQuery Connection Issue'}  
    

    def is_valid(self, uids, target_attribute, valid_list):
        """
        Filter rows where the values in the specified column are not in the valid_set.

        Parameters:
        - uids (list): The list of unique identifiers.
        - target_attribute (str): The name of the column to validate.
        - valid_list (list): The set of valid values for the specified column.
        
        Returns:
        - pd.DataFrame: DataFrame containing rows with values not in the valid_set.
        """ 
        try:
            valid_list= ast.literal_eval(valid_list)
        except:
            raise ValueError('valid_list argument is not a valid list')

        if self.engine: 
            try:
                data_frame = pd.read_sql_query(f"SELECT {','.join(uids)},{target_attribute} AS TARGET_ATTRIBUTE_catalog_value \
                                                FROM {self.schema_name}.{self.table_name} \
                                                WHERE {target_attribute} IS NOT NULL;",self.engine)
                return data_frame[~data_frame['TARGET_ATTRIBUTE_catalog_value'].isin(valid_list)]

            except Exception as e: 
                    print(f"Failed with error: {str(e)}")
                    return { 
                      'status': False, 
                       'err_msg': str(e)} 
        else:
            print('BigQuery Connection Issue: Engine is not available.')
            return { 
                  'status': False, 
                   'err_msg':'BigQuery Connection Issue'} 


    def is_number(self, uids, target_attribute, min_digits=1, max_digits=9, int_only=False, non_zero=False, non_negative=False):
            """
            Verify if a string contains only numbers with a minimum of 1 digit, a maximum of 9 digits.

            Parameters:
            - uids (list): The list of unique identifiers.
            - target_attribute (str): The name of the column to validate.
            - min_digits (int): Minimum number of digits.
            - max_digits (int): Maximum number of digits.
            - int_only (bool): If True, check if the value is an integer.
            - non_zero (bool): If True, check if the value is not zero.
            - non_negative (bool): If True, check if the value is not negative. 

            Returns:
            - pd.DataFrame: DataFrame containing rows where the column is not a valid number.
            """
            try:
                min_digits = int(min_digits)
                max_digits = int(max_digits)
            except:
                raise ValueError('min_digits and max_digits arguments are not valid integers')
            
            for i in [int_only, non_zero, non_negative]:
                if str(i)=='True':
                    pass
                elif str(i)=='False':
                    pass
                else:
                    raise ValueError(f'{i} is not a valid boolean')
            
            if self.engine:
                try:
                    data_frame = pd.read_sql_query(f"SELECT {','.join(uids)},{target_attribute} AS TARGET_ATTRIBUTE_catalog_value \
                                                    FROM {self.schema_name}.{self.table_name} \
                                                    WHERE {target_attribute} IS NOT NULL;",self.engine)

                    def check_numeric(value):
                        try:
                            float(value)
                            return True
                        except:
                            return False

                    # check if value is within min/max digits 
                    final_result = data_frame[~(data_frame['TARGET_ATTRIBUTE_catalog_value'].apply(lambda value: min_digits <= len(str(value)) <= max_digits))]
                    data_frame = data_frame[(data_frame['TARGET_ATTRIBUTE_catalog_value'].apply(lambda value: min_digits <= len(str(value)) <= max_digits))]
                    
                    # check if value is a number
                    result = data_frame[~(data_frame['TARGET_ATTRIBUTE_catalog_value'].apply(lambda value: check_numeric(value)))]
                    if final_result.empty:
                        final_result = result
                    else:
                        final_result = pd.concat([final_result, result])
                    data_frame = data_frame[data_frame['TARGET_ATTRIBUTE_catalog_value'].apply(lambda value: check_numeric(value))]
                    data_frame['TARGET_ATTRIBUTE_catalog_value'] = data_frame['TARGET_ATTRIBUTE_catalog_value'].astype(float)

                    # check if value is an integer
                    if ast.literal_eval(str(int_only)):
                        result = data_frame[~(data_frame['TARGET_ATTRIBUTE_catalog_value'].apply(lambda value: value%1 == 0))]
                        if final_result.empty:
                            final_result = result
                        else:
                            final_result = pd.concat([final_result, result])
                        data_frame = data_frame[(data_frame['TARGET_ATTRIBUTE_catalog_value'].apply(lambda value: value%1 == 0))]
                    
                    #check if value is not zero
                    if ast.literal_eval(str(non_zero)):
                        result = data_frame[~(data_frame['TARGET_ATTRIBUTE_catalog_value'].apply(lambda value: value != 0))]
                        if final_result.empty:
                            final_result = result
                        else:    
                            final_result = pd.concat([final_result, result])
                        data_frame = data_frame[(data_frame['TARGET_ATTRIBUTE_catalog_value'].apply(lambda value: value != 0))]
                    
                    #check if value is not negative
                    if ast.literal_eval(str(non_negative)):
                        result = data_frame[~(data_frame['TARGET_ATTRIBUTE_catalog_value'].apply(lambda value: value >= 0))]
                        if final_result.empty:
                            final_result = result
                        else:      
                            final_result = pd.concat([final_result, result])
                        data_frame = data_frame[(data_frame['TARGET_ATTRIBUTE_catalog_value'].apply(lambda value: value >= 0))]
                        
                    return final_result.reset_index(drop=True)

                except Exception as e:
                    print(f"Failed with error: {str(e)}")
                    return {
                        'status': False,
                        'err_msg': str(e)
                    } 
            else:
                print('BigQuery Connection Issue: Engine is not available.')
                return { 
                    'status': False, 
                    'err_msg':'BigQuery Connection Issue'}        


    def is_text(self, uids, target_attribute, min_char=0, max_char=5000):
        """
        Checks the validity of a string based on specific conditions

        Parameters:
        - uids (list): The list of unique identifiers.
        - target_attribute (str): The name of the column to validate.
        - min_char (int): Minimum number of characters.
        - max_char (int): Maximum number of characters.

        Returns:
        - pd.DataFrame: DataFrame containing rows where the column is not a valid string.
        
        **Validity Conditions:**

        - Must not be empty.
        - Must have a length between 1 and 5000 characters (inclusive).
        - Cannot contain only numbers.
        - Cannot contain only punctuation.
        """
        try:
            min_char = int(min_char)
            max_char = int(max_char)
        except:
            raise ValueError('min_char and max_char must be integers')
        
        if self.engine: 
            try: 
                data_frame = pd.read_sql_query(f"SELECT {','.join(uids)},{target_attribute} AS TARGET_ATTRIBUTE_catalog_value \
                                                FROM {self.schema_name}.{self.table_name} \
                                                WHERE {target_attribute} IS NOT NULL ",self.engine)
                return data_frame[~(data_frame['TARGET_ATTRIBUTE_catalog_value'].apply(lambda value: True if min_char < len(str(value)) <= max_char and bool(re.search(r'[a-zA-Z]', str(value))) else False))]
            except Exception as e: 
                    print(f"Failed with error: {str(e)}")
                    return { 
                      'status': False, 
                       'err_msg': str(e)} 
        else:
            print('BigQuery Connection Issue: Engine is not available.')
            return { 
                  'status': False, 
                   'err_msg':'BigQuery Connection Issue'}
    
                                
    def is_formated(self,uids,target_attribute, regex=None):
        """
        Check if the given string is alphanumeric, containing at least one letter (a-z or A-Z) and one numeric digit (0-9).

        Parameters:
            - uids (list): The list of unique identifiers.
            - target_attribute (str): The name of the column to validate.
            - regex (str): The regular expression to use for validation.

        Returns:
        - pd.DataFrame: DataFrame containing rows where the column is not alphanumeric.
        """
        try:
            regex=re.compile(regex)
        except:
            raise ValueError('regex must be a valid regular expression')

        if self.engine: 
            try: 
                data_frame = pd.read_sql_query(f"SELECT {','.join(uids)},{target_attribute} AS TARGET_ATTRIBUTE_catalog_value \
                                                FROM {self.schema_name}.{self.table_name} \
                                                WHERE {target_attribute} IS NOT NULL",self.engine)
                return data_frame[~(data_frame['TARGET_ATTRIBUTE_catalog_value'].apply(lambda value: bool(re.fullmatch(regex, str(value)))))]
            except Exception as e: 
                    print(f"Failed with error: {str(e)}")
                    return { 
                      'status': False, 
                       'err_msg': str(e)} 
        else:
            print('BigQuery Connection Issue: Engine is not available.')
            return { 
                  'status': False, 
                   'err_msg':'BigQuery Connection Issue'}
                   
if __name__ == "__main__":
    # this code leverages an engine from sqlalchemy to connect to cloud sql databases
    # to utilize this code you will need to create an engine
    # for example: 
    engine = create_engine('bigquery://us-gcp-ame-con-52dbb-sbx-1')                                  

    # #1
    # Create an instance of the class with the necessary information
    dv = DataValidator(engine,'data_quality','meta_Appliances')

    invalid_categories = dv.is_valid(uids=['asin'],target_attribute='main_cat',categories=['Appliances'])
    invalid_categories.head()