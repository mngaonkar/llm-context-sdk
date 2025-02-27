from unittest import loader
from loguru import logger
from src.database.database import Database
import src.database.constants
from src.database.loader import DocumentLoader
from src.database.configuration import Configuration, VectorStoreStatus
import os

class VectorStore():
    """Class for storing vectors."""
    vector_db_initialized = False

    def __init__(self, loader: DocumentLoader, config: Configuration) -> None:
        self.loader = loader
        self.config = config
        self.database = Database() 
        
    def init_vectorstore(self, db_name):
        # webpage = constants.DOC_URL
       
        # for i in range(1, 6):
        #     url = webpage + str(i)
        #     docs = self.loader.load_web_document(url)
        #     self.chat.db.vectorstore.add_documents(docs)
        
        # webpage = "https://gutenberg.org/cache/epub/1661/pg1661.txt"
        # docs = self.loader.load_web_document(webpage)
        # self.chat.db.vectorstore.add_documents(docs)
        database_name = os.path.join(constants.DOCS_LOCATION, db_name + ".db")

        self.config.load_config()
        logger.debug(self.config.get_config())
        logger.debug(f"vector store status = {self.config.get_vector_store_config(database_name).get_vector_store_status()}")

        config_status_good = self.config.get_vector_store_config(database_name).get_vector_store_status() == VectorStoreStatus.READY.value
        db_file_present = self.database.check_db_presence(database_name)

        if not config_status_good:
            logger.info("vector store status not ready as per config file")

        if not db_file_present:
            logger.info(f"vector store database file {database_name} not present")

        if config_status_good and db_file_present:
            self.vector_db_initialized = True
        else:
            self.vector_db_initialized = False

        if not self.vector_db_initialized:
            logger.info("vector store not initialized, initializing...")
            self.database.create_database(database_name)
            db_file_path = os.path.join(constants.DOCS_LOCATION, db_name)
            docs = self.loader.load_documents_from_directory(db_file_path)
            if len(docs) == 0:
                logger.error("No documents found.")
                return
            logger.info(f"Adding {len(docs)} documents to vector store")
            self.database.store_documents(docs)
            logger.info("Done.")
            self.config.get_vector_store_config(database_name).set_vector_store_status(VectorStoreStatus.READY).save_config()  
        else:
            self.database.load_database(database_name)
            logger.info("Vector store already initialized.")