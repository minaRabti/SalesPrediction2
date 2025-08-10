from data_Preprocessor import SalesPreprocessor
from feature_engineering import FeatureEngineer
from dotenv import load_dotenv
from model_trainer import ModelTrainer
#from model_trainer import DeepLearningTrainer

import os

# Load environment variables (if any)
load_dotenv()


# Step 1: Load and preprocess data
dataPath = os.getenv('DataPath')
dollarDatapath = os.getenv('dollarpath')


preprocessor = SalesPreprocessor(dataPath, dollarDatapath)

daily_data = preprocessor.load_data().preprocess()
print("📊 Sample of daily preprocessed data:")
print(daily_data.to_excel('test.xlsx'))


engineer = FeatureEngineer(daily_data, date_column='SALES_DOCUMENT_DATE')
dataset = engineer.transform(lag_column='GROSS_PRICE', lag=30)


# Optional: Save transformed dataset
dataset.to_excel('Result.xlsx')


# Step 3: Initialize Model Trainer
trainer = ModelTrainer(dataset, target_column='GROSS_PRICE', date_column='SALES_DOCUMENT_DATE', lag=30)
#deep_trainer = DeepLearningTrainer(dataset, target_column='GROSS_PRICE', date_column='SALES_DOCUMENT_DATE', lag=30)


# Step 4: Prepare data, train, evaluate, and visualize
trainer.prepare_data2()
#deep_trainer.multivariate_data(dataset, target='GROSS_PRICE', start_index=0, end_index=None,
                               #history_size=30, target_size=1, single_step=False)

#deep_trainer.prepare_data_Deep(noise_scale=0.001, augmentation_factor=2)


trainer.train_model()
#deep_trainer.train_DeepModel(model_type='rnn', epochs=100, batch_size=8)


trainer.evaluate()
#deep_trainer.evaluate_DeepModel()

trainer.save_results2()
#deep_trainer.save_results_Rnns()

trainer.plot_predictions()
#trainer.plot_feature_importance(top_n=20)

# Step 5: Save monthly results to separate file



