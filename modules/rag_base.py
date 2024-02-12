from langchain.agents import initialize_agent, Tool, load_tools, AgentType
from langchain.tools import BaseTool
from langchain_community.utilities import GoogleSearchAPIWrapper
from modules.dq_engine import DqEngine
import os
import time
import pandas as pd
import mapply
import logging

mapply.init(
    n_workers=-1,
    chunk_size=1000,
    max_chunks_per_worker=0,
    progressbar=True
)

class RagBase:
    """
    WebScrapper class for web scraping operations.

    Parameters:
        engine: SQLAlchemy engine for database connection.
    """

    def __init__(self, engine):
        self.engine = engine
        self.output_directory = None
        self.target_attribute = None
        self.source_attribute = None

    def load_data(self, schema_name, table_name, uids, target_attribute, source_attribute,exclude_scored=False):
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
                query = f"""SELECT {','.join(uids)},{self.target_attribute},{source_attribute}
                        FROM {schema_name}.{table_name} AS test_table
                        WHERE {source_attribute} is not null and {source_attribute}!=''
                        {exclusion}"""

                data = pd.read_sql_query(query, self.engine)
                return data
            except Exception as e:
                raise RuntimeError(f"Failed with error: {str(e)}")
        else:
            raise RuntimeError('Connection Issue: Engine is not available.')

    def retrieve_data(self, data_frame, llm, access_keys, prompt, output_directory=None):
        """
        Fetch parent company information using Google Search API.

        Parameters:
            data_frame (pd.DataFrame): DataFrame containing data.
            llm: Langchain Language Model.
            access_keys (dict): Dictionary containing Google API keys.
            output_directory (str): Directory where output files will be saved.
        """
        self.output_directory = output_directory
        os.environ["GOOGLE_CSE_ID"] = access_keys["GOOGLE_CSE_ID"]
        os.environ["GOOGLE_API_KEY"] = access_keys["GOOGLE_API_KEY"]

        search = GoogleSearchAPIWrapper(k=1)
        tool = [Tool(name="Google Search",
                     description="Search Google for recent results.",
                     func=search.run,
                     )]
        
        agent = initialize_agent(tool, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

        data_frame['TARGET_ATTRIBUTE_model_value'] = data_frame[self.source_attribute].apply(
            lambda x: self.fetch_result(agent, eval('f"'+prompt+'"'))
        )
        dqe = DqEngine(self.engine)
        data_frame = dqe.clean_and_compare_columns(data_frame,self.target_attribute,'TARGET_ATTRIBUTE_model_value').reset_index(drop=True)

        data_frame = data_frame.rename(columns={ self.target_attribute : 'TARGET_ATTRIBUTE_catalog_value',        
                                                   self.source_attribute:'TARGET_ATTRIBUTE_source_attributes'})
        data_frame['TARGET_ATTRIBUTE_confidence_score'] =0.95

        if self.output_directory is not None:
            self.save_file(data_frame, "RAG_results")
        else :
            return data_frame

    def fetch_result(self, agent, prompt):
        """
        Fetch result using the Langchain agent.

        Parameters:
            agent: Langchain agent.
            query (str): Query to be executed.

        Returns:
            result: Result of the query execution.
        """
        try:
            result = agent.run(prompt)
            return result
        except Exception as e:
            logging.error(f"Error fetching result: {str(e)}")
            return None

    def save_file(self, df, file_name):
        """
        Saves the given DataFrame as a CSV file in the specified output directory.

        Parameters:
            df (pd.DataFrame): The DataFrame to be saved.
            file_name (str): Name of the file (without extension).
        """
        folder_path = self.output_directory
        csv_filename = f"{file_name}.csv"
        df.to_csv(f"{folder_path}/{csv_filename}", index=False)

