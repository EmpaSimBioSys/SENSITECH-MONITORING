import configparser
from typing import Any, Dict

class ConfigLoader:
    """
    A class to load and access configuration settings from a config.ini file.

    Attributes:
        config (configparser.ConfigParser): The ConfigParser instance containing the configuration data.
        config_dict (Dict[str, Dict[str, Any]]): A dictionary representation of the configuration sections and items.
    """

    def __init__(self) -> None:
        """
        Initializes the ConfigLoader with the path to the config file and loads the data.

        Args:
            file_path (str): The path to the config.ini file.
        """
        self.file_path = 'config.ini'
        self.config = configparser.ConfigParser()
        self.config.read(self.file_path)
        self.config_dict = self._load_config()

    def _load_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads the configuration data into a dictionary.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing the configuration sections and items.
        """
        config_dict = {}
        for section in self.config.sections():
            config_dict[section] = dict(self.config.items(section))
        return config_dict

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Retrieves all key-value pairs from a specified section.

        Args:
            section (str): The section name in the config file.

        Returns:
            Dict[str, Any]: A dictionary of key-value pairs in the specified section.
        """
        return self.config_dict.get(section, {})

    def get_value(self, section: str, key: str) -> Any:
        """
        Retrieves the value associated with a specific key in a given section.

        Args:
            section (str): The section name in the config file.
            key (str): The key for the desired value.

        Returns:
            Any: The value associated with the key in the specified section.
        """
        return self.config_dict.get(section, {}).get(key)

    def get_all_sections(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves all sections and their key-value pairs from the config file.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing all sections and their key-value pairs.
        """
        return self.config_dict

# Usage example
if __name__ == "__main__":
    config_loader = ConfigLoader()
    
    # Example usage
    paths_section = config_loader.get_section('paths')
    dataset_features = config_loader.get_value('dataset', 'coop_features')
    all_config_data = config_loader.get_all_sections()

    print("Paths section:", paths_section)
    print("Coop features:", dataset_features)
    print("All config data:", all_config_data)