from tinydb import TinyDB, Query
import os, json

class ConfigDB():
    """A class to manage configuration settings using TinyDB."""
    def __init__(self):
        self.db = None

    def _check_db(self):
        assert self.db is not None, "config db is not initialized"

    def insert_config(self, config):
        """Insert a new configuration into the database."""
        self._check_db()
        self.db.insert(config)

    def get_config(self, key_name, key_value):
        """Get a configuration from the database."""
        self._check_db()
        q = Query()
        result = self.db.search(q[key_name] == key_value)
        return result if result else None
    
    def update_config(self, key_name, key_value, attribute_name, attribute_value):
        """Update a configuration in the database."""
        self._check_db()
        q = Query()
        self.db.update({attribute_name: attribute_value}, q[key_name] == key_value)

    def delete_config(self, key_name, key_value):
        """Delete a configuration from the database."""
        self._check_db()
        q = Query()
        self.db.remove(q[key_name] == key_value)

    def setup(self, config_files_path: str, config_db_path: str, db_name:str, config_files: list):
        """Setup DB from config files"""
        # Check if the config DB exists, if yes delete it
        if os.path.exists(os.path.join(config_db_path, db_name)):
            os.remove(os.path.join(config_db_path, db_name))

        # Initialize fresh DB
        self.db = TinyDB(os.path.join(config_files_path, db_name))

        for config in config_files:
            file = os.path.join(config_files_path, config)
            with open(file, "r") as fp:
                self.insert_config(json.loads(fp.read()))