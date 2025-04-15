from src.configuration.configdb import ConfigDB
import json
import os
import logging
from tinydb import TinyDB, Query

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PipelineConfig():
    CONFIG_TYPE_KEY = "config_type"
    CONFIG_TYPE_VALUE = "pipeline"
    PIPELINE_NAME_KEY = "pipeline_name"

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

    def set_pipeline_config(self, pipeline_name, config):
        """Initialize the configuration."""
        q = Query()
        updated = self.db.update(config, (q[PipelineConfig.PIPELINE_NAME_KEY] == pipeline_name) \
                       & (q[PipelineConfig.CONFIG_TYPE_KEY] == PipelineConfig.CONFIG_TYPE_VALUE))
        if len(updated) == 0:
            logger.info("Pipeline config not found, inserting new config.")
            inserted = self.db.insert(config)
            if inserted == 0:
                logger.error("Failed to insert pipeline config.")
                return False
            return True
        else:
            logger.info("Pipeline config updated.")
            return True

    def get_pipeline_config(self, pipeline_name):
        """Get a configuration from the database."""
        q = Query()
        result = self.db.search((q[PipelineConfig.PIPELINE_NAME_KEY] == pipeline_name) \
                                & (q[PipelineConfig.CONFIG_TYPE_KEY] == PipelineConfig.CONFIG_TYPE_VALUE))
        logger.info(f"Retrieved pipeline config for {pipeline_name}: {result}")
        return result if len(result) > 0 else None
    
    def delete_pipeline_config(self, pipeline_name):
        """Delete a configuration from the database."""
        q = Query()
        self.db.remove(q[PipelineConfig.PIPELINE_NAME_KEY] == pipeline_name \
                       & q[PipelineConfig.CONFIG_TYPE_KEY] == PipelineConfig.CONFIG_TYPE_VALUE)
        logger.info(f"Deleted pipeline config for {pipeline_name}.")

    def delete_all_pipeline_config(self):
        """Delete all configurations from the database."""
        self.db.remove(q[PipelineConfig.CONFIG_TYPE_KEY] == PipelineConfig.CONFIG_TYPE_VALUE)
        logger.info("Deleted all pipeline configs.")