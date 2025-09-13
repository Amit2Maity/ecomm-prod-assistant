import os
import sys
from typing import List
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from utils.config_loader import load_config
from utils.model_loader import ModelLoader
from dotenv import load_dotenv
from pathlib import Path

class Retriever:
    def __init__(self):
        """Initialize the Retriever with necessary configurations.
        """
        self.model_loader = ModelLoader()
        self.config = load_config()
        self._load_env_variables()
        self.vstore = None
        self.retriever = None
        
    def _load_env_variables(self):
        """Load environment variables for AstraDB connection.
        """
        load_dotenv()
        
        required_vars= ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")
        
        self.google_api_key = os.getenv("GROQ_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")
        
    def load_retriever(self):
        """Load the AstraDB retriever.
        """
        
        if not self.vstore:
            collection_name = self.config["astra_db"]["collection_name"]
        
        self.vstore = AstraDBVectorStore(
            embedding=self.model_loader.load_embeddings(),
            collection_name=collection_name,
            api_endpoint=self.db_api_endpoint,
            token=self.db_application_token,
            namespace=self.db_keyspace,
        )
        
        if not self.retriever:
            top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3
            retriever = self.vstore.as_retriever(search_kwargs={"k": top_k})
            print(f"Retriever loaded with top_k={top_k} successfully.")
            return retriever
    
    def call_retriever(self,query):
        """Call the retriever with a query.

        Args:
            query (str): The query string.

        Returns:
            List[Document]: List of retrieved documents.
        """
        retriever = self.load_retriever()
        
        results = retriever.invoke(query)
        return results
    
if __name__ == "__main__":
    retriever_obj = Retriever()
    user_query = "Can you suggest some budget smartphones?"
    results = retriever_obj.call_retriever(user_query)
    
    for idx, doc in enumerate(results,1):
        print(f"Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")
        