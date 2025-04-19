from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def main(self):
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)
        trainer = ModelTrainer()
        r2_score = trainer.initiate_model_trainer(train_arr, test_arr)
        return r2_score

if __name__ == "__main__":
    pipeline = TrainPipeline()
    metrics = pipeline.main()
    print(f"Training Metrics: {metrics}")