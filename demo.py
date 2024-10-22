from hate.logger import logging
from hate.exception import CustomException 
#logging.info("welcome to our home")
from hate.configuration.gcloud_syncer import GCloudSync

obj = GCloudSync()
obj.sync_folder_from_gcloud("hate-speech20244", "dataset.zip", "download/dataset.zip")