from src.configuration.configdb import ConfigDB
import json
import os
import logging
from tinydb import TinyDB, Query

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DatasetConfig():
    CONFIG_TYPE_KEY = "config_type"
    CONFIG_TYPE_VALUE = "dataset"
    DATASET_NAME_KEY = "dataset_name"

    """Class to manage LLM provider configuration."""
    def __init__(self, db_path, create_if_not_exists=True):
        self.db_path = db_path
        if not os.path.exists(db_path) and not create_if_not_exists:
            logger.warning(f"Configuration database not found at {db_path}, creating new one.")
            raise FileNotFoundError(f"Configuration database not found at {db_path}.")
        else:
            self.db = TinyDB(self.db_path)

    def __del__(self):
        """Close the database connection."""
        self.db.close()

    def set_dataset_config(self, dataset_name, config):
        """Initialize the configuration."""
        q = Query()
        updated = self.db.update(config, (q[DatasetConfig.DATASET_NAME_KEY] == dataset_name) \
                       & (q[DatasetConfig.CONFIG_TYPE_KEY] == DatasetConfig.CONFIG_TYPE_VALUE))
        if len(updated) == 0:
            logger.info("Dataset config not found, inserting new config.")
            inserted = self.db.insert(config)
            if inserted == 0:
                logger.error("Failed to insert dataset config.")
                return False
            return True
        else:
            logger.info("Dataset config updated.")
            return True

    def get_dataset_config(self, dataset_name):
        """Get a configuration from the database."""
        q = Query()
        result = self.db.search((q[DatasetConfig.DATASET_NAME_KEY] == dataset_name) \
                                & (q[DatasetConfig.CONFIG_TYPE_KEY] == DatasetConfig.CONFIG_TYPE_VALUE))
        logger.info(f"Retrieved dataset for {dataset_name}: {result}")
        return result if len(result) > 0 else None
    
    def delete_dataset_config(self, dataset_name):
        """Delete a configuration from the database."""
        q = Query()
        self.db.remove(q[DatasetConfig.PROVIDER_NAME_KEY] == dataset_name \
                       & q[DatasetConfig.CONFIG_TYPE_KEY] == DatasetConfig.CONFIG_TYPE_VALUE)
        logger.info(f"Deleted dataset config for {dataset_name}.")

    def delete_all_dataset_config(self):
        """Delete all configurations from the database."""
        self.db.remove(q[DatasetConfig.CONFIG_TYPE_KEY] == DatasetConfig.CONFIG_TYPE_VALUE)
        logger.info("Deleted all dataset configs.")