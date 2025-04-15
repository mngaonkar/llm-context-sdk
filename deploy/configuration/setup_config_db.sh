from src.configuration.configdb import ConfigDB

CONFIG_DB_PATH = "./deploy/configuration"
CONFIG_DB_NAME = "config.db"
CONFIG_FILES = ["dataset_config.json", "pipeline_config.json", "ollama_config.json"]


ConfigDB().setup(CONFIG_DB_PATH, CONFIG_DB_NAME, CONFIG_FILES)

