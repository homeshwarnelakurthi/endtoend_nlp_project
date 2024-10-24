import os
import io
import sys
import keras
import pickle
from PIL import Image
from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from keras.utils import pad_sequences
from hate.configuration.gcloud_syncer import GCloudSync
from hate.components.data_transformation import DataTransformation
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts


class PredictionPipeline:
    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts", "PredictModel")
        self.gcloud = GCloudSync()
        self.data_transformation = DataTransformation(
            data_transformation_config=DataTransformationConfig,
            data_ingestion_artifacts=DataIngestionArtifacts
        )

    def get_model(self) -> str:
        """
        Method Name :   get_model
        Description :   This method checks for the model locally before downloading from Google Cloud Storage.
        Output      :   model_path
        """
        logging.info("Entered the get_model method of PredictionPipeline class")
        try:
            # Check if the model exists locally
            best_model_path = os.path.join(self.model_path, self.model_name)
            if os.path.exists(best_model_path):
                logging.info(f"Model found locally at {best_model_path}. No need to download.")
                return best_model_path

            # If model not found locally, download from GCloud
            os.makedirs(self.model_path, exist_ok=True)
            self.gcloud.sync_folder_from_gcloud(self.bucket_name, self.model_name, self.model_path)
            logging.info(f"Model downloaded from GCloud to {best_model_path}")
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self, best_model_path, text):
        """Load image, returns CUDA tensor"""
        logging.info("Running the predict function")
        try:
            # Load the model
            load_model = keras.models.load_model(best_model_path)

            # Load the tokenizer
            with open('tokenizer.pickle', 'rb') as handle:
                load_tokenizer = pickle.load(handle)

            # Clean and prepare the text
            text = self.data_transformation.concat_data_cleaning(text)
            text = [text]
            seq = load_tokenizer.texts_to_sequences(text)
            padded = pad_sequences(seq, maxlen=300)

            # Make predictions
            pred = load_model.predict(padded)

            if pred > 0.5:
                return "hate and abusive"
            else:
                return "no hate"

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, text):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            # Get the model path
            best_model_path = self.get_model()

            # Perform prediction
            predicted_text = self.predict(best_model_path, text)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_text

        except Exception as e:
            raise CustomException(e, sys) from e
