from hate.logger import logging
from hate.exception import CustomException 
#logging.info("welcome to our home")
import sys
try:
    s= 3 / "0"
    
except Exception as e:
    raise CustomException(e,sys)