import logging
import yaml
import colorama
from neo4j import GraphDatabase
import os



def is_running_in_docker():
    if os.path.exists('/.dockerenv'):
        return True
    try:
        with open('/proc/1/cgroup', 'rt') as f:
            for line in f:
                if 'docker' in line:
                    return True
    except Exception:
        pass
    return False

if is_running_in_docker():
    INPUT_DIR = '/mnt/input'
    OUTPUT_DIR = '/mnt/output'
    print("Running inside Docker container.")
else:
    INPUT_DIR = './..'
    OUTPUT_DIR = './..'
    print("Running on local machine.")



INFO_LOGGER_FILE = "./i_log.txt"
ERROR_LOGGER_FILE = "./e_log.txt"

def clear_log_files():
    with open(INFO_LOGGER_FILE, "w") as f:
        f.write("")
    with open(ERROR_LOGGER_FILE, "w") as f:
        f.write("")



def new_logger_i():
    print("setting up logger")
    def logger(msg):
        print(colorama.Fore.WHITE + str(msg) + colorama.Style.RESET_ALL)
        with open(INFO_LOGGER_FILE, "a") as f:
            f.write(f"{str(msg)}\n")
        
    return logger


def new_logger_e():
    print("setting up logger")
    return lambda msg: (
        print(colorama.Fore.RED + str(msg) + colorama.Style.RESET_ALL),
         open(ERROR_LOGGER_FILE, "a").write(f"{str(msg)}\n") and None,
    )



def create_logger():
    global logger
    logger = logging.getLogger(__name__)
    clear_log_files()
    logger.info = new_logger_i()
    logger.error = new_logger_e()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
  
    return logger




def read_config(file_path=f"{INPUT_DIR}/config.yml",):
    try:
        with open(file_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        return config
                
    except FileNotFoundError:
        logger.info(f"Config file '{file_path}' not found.")



def get_neo4j_credentials():
    config = read_config()
    neo4j_credentials = config.get("neo4j_credentials", {})
    NEO4J_URI = neo4j_credentials.get("NEO4J_URI", "")
    NEO4J_USERNAME = neo4j_credentials.get("NEO4J_USERNAME", "")
    NEO4J_PASSWORD = neo4j_credentials.get("NEO4J_PASSWORD", "")
    NEO4J_DB = neo4j_credentials.get("NEO4J_DB", "")
    logger.info(f"Neo4j Connect to {NEO4J_URI} using {NEO4J_USERNAME}")
    return NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DB


def connect_to_neo4j():
    logger.info("Connecting to Neo4j")
    config = get_neo4j_credentials()
    global NEO4J_DB
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DB = config
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    logger.info("Connected to Neo4j")
    
    return driver


def document(y_pred, y_true,):
    # shapes - features - typpes of features (with numbers)
    # accuracy + precision + recall
    # f1 score
    pass
