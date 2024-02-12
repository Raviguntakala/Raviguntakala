import time
import pandas as pd
from sklearn.ensemble import IsolationForest
from vertexai.language_models import TextEmbeddingModel
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
import re
import string

class TextOutlierDetector:
    def __init__(self, engine):
        """
        Initializes an instance of the OutlierDetector class with the specified parameters.

        Parameters:
            engine: SQLAlchemy engine for database connection.
            schema_name (str): Name of the database schema.
            product_column (str): The column name in the DataFrame containing product information.
            output_directory (str): Directory where output files will be saved.
        """
        self.engine = engine
        self.category_column = None
        self.output_directory = None
        self.target_attribute_train = None
        self.target_attribute_test = None

    def load_train(self, schema_name, table_name, uids, target_attribute, embeddings, category_column=[]):
        """
        Retrieve rows where the specified column is NULL.

        Parameters:
            table_name (str): The name of the table.
            uids (list): List of column names used as unique identifiers.
            target_attribute (str): The column to be retrieved.
            category_column (str): The column containing category information to split the data for training & inference.

        Returns:
            pd.DataFrame: DataFrame containing training data.
        """
        self.train_embeddings = embeddings

        if len(category_column)>0:
            self.category_column=category_column
            uids += [category_column]

        if self.engine:
            try:
                train_data = pd.read_sql_query(
                    f"""SELECT {','.join(uids)},{target_attribute} AS TARGET_ATTRIBUTE_catalog_value,{embeddings}
                    FROM {schema_name}.{table_name}
                    WHERE ARRAY_LENGTH({embeddings})!=0;""",
                    self.engine
                    )

                return train_data
            except Exception as e:
                print(f"Failed with error: {str(e)}")
                return {
                    'status': False,
                    'err_msg': str(e)
                }
        else:
            print('Connection Issue: Engine is not available.')
            return {
                'status': False,
                'err_msg': 'Connection Issue'
            }


    def load_test(self, schema_name, table_name, uids, target_attribute, embeddings, exclude_scored=False):
        """
        Retrieve rows where the specified column is NULL.

        Parameters:
            uids (list): List of column names used as unique identifiers.
            target_attribute (str): The column to be retrieved.
            table_name (str): The name of the table.

        Returns:
            pd.DataFrame: DataFrame containing data for inference.
        """
        self.test_embeddings = embeddings
        
        if exclude_scored==False:
            exclusion = ';'
        else:
            exclusion = f"""AND NOT EXISTS (
                SELECT {uids[0]}
                FROM `{exclude_scored}` as RESULTS
                WHERE RESULTS.TARGET_ATTRIBUTE="{target_attribute}"
                AND test_table.{uids[0]}=RESULTS.{uids[0]});"""
            
        if self.category_column is not None:
            uids += [self.category_column]

        if self.engine:
            try:
                test_data = pd.read_sql_query(
                    f"""SELECT {','.join(uids)},{target_attribute} AS TARGET_ATTRIBUTE_catalog_value,{embeddings}
                        FROM {schema_name}.{table_name} AS test_table
                        WHERE ARRAY_LENGTH({embeddings})!=0
                        {exclusion}""",
                        self.engine
                        )

                return test_data
            except Exception as e:
                print(f"Failed with error: {str(e)}")
                return {
                    'status': False,
                    'err_msg': str(e)
                }
        else:
            print('Connection Issue: Engine is not available.')
            return {
                'status': False,
                'err_msg': 'Connection Issue'
            }


    def train_single_model(self, train_table, categories=[], contamination=0.01):
        """
        Trains an isolation forest model using the train data.

        Parameters:
            train_embeddings (pd.DataFrame): DataFrame containing the training data.
            categories (list): List of categories to include for training the model.
            contamination (float): Proportion of outliers expected in the data (default is 0.01).

        Returns:
            odm: Trained isolation forest Outlier Detection Model.
        """
        if (self.category_column is not None) and (len(categories)>0):
            train_table = train_table[train_table[self.category_column].isin(categories)]

        odm = IsolationForest(n_estimators=500,random_state=0, contamination=contamination)
        odm.fit(list(train_table[self.train_embeddings]))
        return odm
        

    def check_outliers(self, train_table, test_table, contamination=0.01, output_directory=None, outlier_threshold=None):
        """
        Performs modeling and outlier detection for specified product types.

        Parameters:
            train_embeddings (pd.DataFrame): DataFrame used as the training set for modeling.
            test_embeddings (pd.DataFrame): DataFrame containing the data to be analyzed.
            category (list): List of product types for which to perform modeling.
            contamination (float): Proportion of outliers expected in the data (default is 0.01).
        """
        self.output_directory = output_directory
        odm = IsolationForest(n_estimators=500,random_state=0, contamination=contamination)

        if self.category_column is None:
            odm.fit(list(train_table[self.train_embeddings]))
            return self._perform_outlier_detection(test_table, odm, 'text_outlier_', 'predictions', outlier_threshold)
        else:
            categories=train_table[self.category_column].unique()
            combined_data = pd.DataFrame()
            for cat in categories:
                cat_train_data = train_table[train_table[self.category_column] == cat].reset_index(drop=True)
                cat_test_data = test_table[test_table[self.category_column] == cat].reset_index(drop=True)

                odm.fit(list(cat_train_data[self.train_embeddings]))
                modified_data = self._perform_outlier_detection(cat_test_data, odm, cat, 'predictions', outlier_threshold)
                if modified_data.empty:
                    continue
                else:
                    combined_data = pd.concat([combined_data, modified_data], ignore_index=True)
            return combined_data

    
    def _perform_outlier_detection(self, data, odm, file_name, suffix='predictions', outlier_threshold=None, ):
        """
        Helper function to perform outlier detection on the given data and save the results.

        Parameters:
            data (pd.DataFrame): DataFrame containing the data to be analyzed.
            odm: IsolationForest Outlier Detection model.
            file_name (str): Name of the file (without extension).
            suffix (str): Suffix to be appended to the file name (default is 'train_sample').
        """
        data['TARGET_ATTRIBUTE_confidence_score'] = odm.score_samples(list(data[self.test_embeddings]))
        data['TARGET_ATTRIBUTE_confidence_score'] = data['TARGET_ATTRIBUTE_confidence_score'] * -1
        data = data.drop(columns=[self.test_embeddings])
        data['TARGET_ATTRIBUTE_model_value'] = None
        data['TARGET_ATTRIBUTE_source_attributes'] = None 
        if self.output_directory is not None:
            self.save_file(data, file_name, suffix)
        if outlier_threshold is not None:
            data = data[data['TARGET_ATTRIBUTE_confidence_score']>outlier_threshold] 
        return data
        
    def save_file(self, df, file_name, suffix='predictions'):
        """
        Saves the given DataFrame as a CSV file in the specified output directory.

        Parameters:
            df (pd.DataFrame): The DataFrame to be saved.
            file_name (str): Name of the file (without extension).
            suffix (str): Suffix to be appended to the file name (default is 'train_sample').
        """
        folder_path = self.output_directory
        csv_filename = f"{file_name}_{suffix}.csv"
        df.to_csv(f"{folder_path}/{csv_filename}", index=False)