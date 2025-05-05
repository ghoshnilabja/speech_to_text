import os
import sys
import yaml
import subprocess

# Path of the config file
CONFIG_PATH = "/home/vista-ai-07/Desktop/work/sentiment-live_feedsense/config"
config_name = "config.yaml"

# Function to load yaml configuration file
def load_config(config_name,CONFIG_PATH = CONFIG_PATH):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

def install_libraries(requirements_file):
    with open(requirements_file, 'r') as file:
        libraries = [line.strip() for line in file.readlines() if line.strip()]

    for library in libraries:
        subprocess.call(['pip', 'install', library])

if __name__ == "__main__":
    requirements_file = "requirements.txt"
    install_libraries(requirements_file)