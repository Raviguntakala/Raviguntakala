# Sqlalchemy functions
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *


#langchain functions
from langchain import hub
from langchain.agents import AgentExecutor, tool, load_tools
from crewai import Agent, Task, Crew, Process
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from vertexai.language_models import TextGenerationModel

#other necessary functions
import os
import json
import pandas as pd
from modules.dq_engine import DqEngine



class ProductInfoExtractor:
    def __init__(self,engine):
        self.engine = engine
        self.target_attribute = None
        self.source_attribute = None
 
    def load_data(self, schema_name, table_name, uids, target_attribute,source_attribute,exclude_scored=False):
        """
        Retrieve rows where the specified column is NOT NULL.

        Parameters:
            schema_name (str): Name of the database schema.
            table_name (str): The name of the table.
            uids (list): List of column names used as unique identifiers.
            target_attribute (str): The column to be retrieved.

        Returns:
            pd.DataFrame: DataFrame containing training data.
        """
        self.target_attribute = target_attribute
        self.source_attribute = source_attribute

        if exclude_scored==False:
            exclusion = ';'
        else:
            exclusion = f"""AND NOT EXISTS (
                SELECT {uids[0]}
                FROM `{exclude_scored}` as RESULTS
                WHERE RESULTS.TARGET_ATTRIBUTE="{self.target_attribute}"
                AND test_table.{uids[0]}=RESULTS.{uids[0]});"""

        if self.engine:
            try:
                query = f"""SELECT {','.join(uids)},{','.join(self.target_attribute)}, {self.source_attribute} AS TARGET_ATTRIBUTE_source_attributes
                            FROM {schema_name}.{table_name} 
                            WHERE {self.source_attribute} is not null and {self.source_attribute}!=''
                            {exclusion}"""

                data = pd.read_sql_query(query, self.engine)
                return data
            except Exception as e:
                raise RuntimeError(f"Failed with error: {str(e)}")
        else:
            raise RuntimeError('Connection Issue: Engine is not available.')


    def execute_tasks(self,agents, task_descriptions, product_title):
        """
            Creates and executes tasks with a crew of agents using zip.

            Args:
                agents (list): List of Agent objects to assign tasks to.
                task_descriptions (list): List of task descriptions,
                    each containing product_title placeholders for formatting.
                product_title (str): The specific product title for the tasks.

            Returns:
                The result of executing the tasks by the crew.
        """
        print(f"""
        ----------------------------------------------
        product title: {product_title}
        ----------------------------------------------""")
        tasks = [Task(description=desc.format(product_title=product_title), agent=agent) for desc, agent in zip(task_descriptions, agents)]
        tech_crew = Crew(agents=agents, tasks=tasks, process=Process.sequential)
        return tech_crew.kickoff()
    
    
    def extract_attributes(self,data_frame,agents, task_descriptions):
        """
        Extracts attributes from a DataFrame by executing tasks with assigned agents.

        Args:
            data_frame (pd.DataFrame): DataFrame containing product titles.
            agents (list): List of Agent objects to execute the tasks.
            task_descriptions (list): List of task descriptions for each product title.

        Returns:
            The DataFrame with an added 'Attribute_extraction' column containing the extracted attributes.
        """
        
        data_frame['Attribute_extraction']  = data_frame[self.source_attribute].apply(lambda x: self.execute_tasks(agents, task_descriptions,x))

        self._add_attribute_columns(data_frame)

    def _add_attribute_columns(self,data_frame):

        data_frame['net_content_quantity_'] = data_frame['Attribute_extraction'].apply(lambda x: json.loads(x)['net_content_quantity']).astype('float64')
        data_frame['net_content_uom_']      = data_frame['Attribute_extraction'].apply(lambda x: json.loads(x)['net_content_uom'])
        data_frame['count_per_pack_']       = data_frame['Attribute_extraction'].apply(lambda x: json.loads(x)['count_per_pack']).astype('float64')
        data_frame['multi_quantity_']       = data_frame['Attribute_extraction'].apply(lambda x: json.loads(x)['multi_quantity']).astype('float64')
        data_frame['total_quantity_']       = data_frame['Attribute_extraction'].apply(lambda x: json.loads(x)['total_quantity']).astype('float64')

        dqe = DqEngine(self.engine)
        print(self.target_attribute)
        for attr in self.target_attribute:
            source_attr_col = attr
            extracted_attr_col = attr + "_"
            if source_attr_col.lower() !='net_content_uom':
                data_frame[source_attr_col]=data_frame[source_attr_col].astype('float64')

            result_df = dqe.clean_and_compare_columns(data_frame,source_attr_col,extracted_attr_col).reset_index(drop=True)

            if result_df.empty:
                pass
            else:
                
                result_df = result_df.rename(columns={source_attr_col: 'TARGET_ATTRIBUTE_catalog_value',
                                                          extracted_attr_col:'TARGET_ATTRIBUTE_model_value'})
                result_df['TARGET_ATTRIBUTE'] = source_attr_col
                result_df['TARGET_ATTRIBUTE_confidence_score'] =0.95
                print(source_attr_col)
                # Writes results to a BigQuery table.
                dqe.write_dqresults(result_df, 
                                    uids=['asin','category','title'], 
                                    target_attribute=source_attr_col, 
                                    function_name='attr_extr', 
                                    error_id=30003, 
                                    error_code=f"Accuracy : Extracted {source_attr_col} attribute value doesn't match with  catalog value",
                                    output_schema='output'
                                )