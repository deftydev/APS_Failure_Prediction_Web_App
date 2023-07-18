import os
import logging
from datetime import datetime

LOG_FILE= f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_path= os.path.join(os.getcwd(),"logs",LOG_FILE)

os.makedirs(logs_path,exist_ok=True)

Log_File_Path= os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=Log_File_Path,
    format="[%(asctime)s] %(lineno)d -%(name)s -%(levelname)s-%(message)s",
    level=logging.INFO

)