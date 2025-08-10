import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout
from keras.regularizers import l2
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
#from tsaug import TimeWarp

class ModelTrainer:
    
    def __init__(self, df, target_column='GROSS_PRICE', date_column='SALES_DOCUMENT_DATE', lag=30):
        self.df = df.copy()
        self.target_column = target_column
        self.date_column = date_column
        self.lag = lag
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.xgb_model = None
        self.rf_model = None
        self.knn_model = None
        self.y_pred_xgb = None
        self.y_pred_rf = None
        self.y_pred_knn = None
        
        
        
          
        
    def prepare_data2(self):
        
        self.df.set_index(self.date_column, inplace=True)
        self.df.sort_index(inplace=True)
        
        
        
        dataset = self.df.to_numpy()
        target_col_index = self.df.columns.get_loc(self.target_column)

        X, y = self._create_lagged_dataset(dataset, target_col_index)
        
        #y = np.log1p(y)

        split_idx = int(len(X) * 0.8)
        
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]

        self.X_train = self.scaler_x.fit_transform(self.X_train)
        self.X_test = self.scaler_x.transform(self.X_test)

   
        self.y_train = self.scaler_y.fit_transform(self.y_train.reshape(-1, 1)).ravel()
        self.y_test = self.scaler_y.transform(self.y_test.reshape(-1, 1)).ravel()
        
        sales_dates = self.df.index.values
        
        df_test_dates = sales_dates[-len(self.y_test):]
        self.test_dates = df_test_dates

        # Sample weighting: weight=3 for y=0
        self.sample_weight_train = np.array([1.5 if val == 0 else 1 for val in self.y_train])

      

    def _create_lagged_dataset(self, dataset, target_col_index):
        X, y = [], []
        for i in range(self.lag, len(dataset)):
            X.append(dataset[i - self.lag:i].flatten())
            y.append(dataset[i, target_col_index])
        return np.array(X), np.array(y)

 

    def train_model(self):
        # XGBoost
        print('Starting Train model with XGB')
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        self.xgb_model.fit(self.X_train, self.y_train)
        
        # Random Forest
        print('Starting Train model with RF')
        self.rf_model = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            min_samples_split=2,
            min_samples_leaf=2,
            criterion='squared_error'
        )
        self.rf_model.fit(self.X_train, self.y_train)

        # KNN Regressor
        print('Starting Train model with KNN')
        self.knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
        self.knn_model.fit(self.X_train, self.y_train)

        # Predictions
        self.y_pred_xgb = self.xgb_model.predict(self.X_test)
        self.y_pred_rf = self.rf_model.predict(self.X_test)
        self.y_pred_knn = self.knn_model.predict(self.X_test)

    
    def evaluate(self):
        
        def print_metrics(y_true, y_pred, model_name):
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            denominator = (np.abs(y_true) + np.abs(y_pred) + 1) / 2
            #denominator = np.where(denominator == 0, 1e-8, denominator)
            diff = np.abs(y_true - y_pred) / denominator
            smape = np.mean(diff) * 100
            accuracy = (1 - smape/200)*100
  
            print(f"\n📊 Evaluation for {model_name}:")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  MAE: {mae:.3f}")
            print(f"  R2 Score: {r2:.3f}")
            print(f"  smape: {smape:.3f}")
            print(f"  Acc: {accuracy:.3f}")

        print_metrics(self.y_test, self.y_pred_xgb, "XGBoost")
        print_metrics(self.y_test, self.y_pred_rf, "Random Forest")
        print_metrics(self.y_test, self.y_pred_knn, "Knn Regressor")

       
    def save_results(self, save_dir='D:/Mina/SalePredictionProject_withOutLog-Best/results'):

        os.makedirs(save_dir, exist_ok=True)

        # 🔄 Inverse scale predictions
        pred_xgb_inv = self.scaler_y.inverse_transform(self.y_pred_xgb.reshape(-1, 1)).flatten()
        pred_rf_inv = self.scaler_y.inverse_transform(self.y_pred_rf.reshape(-1, 1)).flatten()
        pred_knn_inv = self.scaler_y.inverse_transform(self.y_pred_knn.reshape(-1, 1)).flatten()
        actual_inv = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).flatten()

        # 📊 Create daily results DataFrame
        df_results = pd.DataFrame({
            'Date': self.test_dates,
            'Actual': actual_inv,
            'Predicted - XGBoost': pred_xgb_inv,
            'Predicted - Random Forest': pred_rf_inv,
            'Predicted - Knn': pred_knn_inv,
        })

        df_results['Date'] = pd.to_datetime(df_results['Date'])
        df_results.set_index('Date', inplace=True)

        # 📦 Monthly aggregation (excluding future forecast)
        monthly_df = df_results.resample('M').sum(min_count=1)

        # 🔮 Forecast next 30 days
        preds_xgb_next, preds_rf_next, preds_knn_next = self.predict_next_days(days=30)

        # 🔢 Total forecast for the next 30 days
        next_month_pred_xgb = np.nansum(preds_xgb_next)
        next_month_pred_rf = np.nansum(preds_rf_next)
        next_month_pred_knn = np.nansum(preds_knn_next)

        # 📅 Date for the next month (last day of the month containing the date 30 days from last_date)
        last_date = df_results.index.max()
        # اضافه کردن 30 روز
        future_date = last_date + pd.Timedelta(days=30)
        # تنظیم به آخرین روز همان ماه
        next_month = future_date + pd.offsets.MonthEnd(0)

        # ➕ Append next month forecast
        monthly_df.loc[next_month] = {
            'Actual': None,
            'Predicted - XGBoost': next_month_pred_xgb,
            'Predicted - Random Forest': next_month_pred_rf,
            'Predicted - Knn': next_month_pred_knn
        }

        # 📝 Save results
        df_results.to_excel(os.path.join(save_dir, 'results_daily.xlsx'))
        monthly_df.to_excel(os.path.join(save_dir, 'results_monthly.xlsx'))

        # ✅ Optional: Save next 30 days forecast in separate file
        df_next_30 = pd.DataFrame({
            'Date': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30),
            'Predicted - XGBoost': preds_xgb_next,
            'Predicted - Random Forest': preds_rf_next,
            'Predicted - KNN': preds_knn_next
        })
        df_next_30.set_index('Date', inplace=True)
        df_next_30.to_excel(os.path.join(save_dir, 'results_next_30_days.xlsx'))

        print(f"\n📁 Daily results saved to: {save_dir}/results_daily.xlsx")
        print(f"📁 Monthly results saved to: {save_dir}/results_monthly.xlsx (including next month forecast)")
        print(f"📁 Next 30-day forecast saved to: {save_dir}/results_next_30_days.xlsx")
        print(f"Next month forecast date:{next_month}")
        
    def predict_next_days(self, days=30):
        
        last_sequence = self.df[-self.lag:].to_numpy().flatten()
        input_seq = last_sequence.reshape(1, -1)
        input_seq_scaled = self.scaler_x.transform(input_seq)

        future_preds_xgb = []
        future_preds_rf = []
        future_preds_knn = []

        for _ in range(days):
            # XGBoost prediction
            pred_xgb = self.xgb_model.predict(input_seq_scaled)[0]
            future_preds_xgb.append(pred_xgb)

            # RF prediction
            pred_rf = self.rf_model.predict(input_seq_scaled)[0]
            future_preds_rf.append(pred_rf)
            
            pred_knn = self.knn_model.predict(input_seq_scaled)[0]
            future_preds_knn.append(pred_knn)

            # Prepare next input: shift + append new prediction
            new_row = np.zeros(self.df.shape[1])
            target_idx = self.df.columns.get_loc(self.target_column)
            new_row[target_idx] = pred_xgb  

            last_sequence = np.concatenate([last_sequence[self.df.shape[1]:], new_row])
            input_seq_scaled = self.scaler_x.transform(last_sequence.reshape(1, -1))

        
        preds_xgb_inv = self.scaler_y.inverse_transform(np.array(future_preds_xgb).reshape(-1, 1)).flatten()
        preds_rf_inv = self.scaler_y.inverse_transform(np.array(future_preds_rf).reshape(-1, 1)).flatten()
        preds_knn_inv = self.scaler_y.inverse_transform(np.array(future_preds_knn).reshape(-1, 1)).flatten()

        return preds_xgb_inv, preds_rf_inv, preds_knn_inv    
    
    def save_results2(self, save_dir='D:/Mina/SalePredictionProject_withOutLog-Best/results'):
        
            os.makedirs(save_dir, exist_ok=True)

            # 🔄 Inverse scale predictions
            pred_xgb_inv = self.scaler_y.inverse_transform(self.y_pred_xgb.reshape(-1, 1)).flatten()
            pred_rf_inv = self.scaler_y.inverse_transform(self.y_pred_rf.reshape(-1, 1)).flatten()
            pred_knn_inv = self.scaler_y.inverse_transform(self.y_pred_knn.reshape(-1, 1)).flatten()
            actual_inv = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).flatten()

            # 📊 Create daily results DataFrame
            df_results = pd.DataFrame({
                'Date': self.test_dates,
                'Actual': actual_inv,
                'Predicted - XGBoost': pred_xgb_inv,
                'Predicted - Random Forest': pred_rf_inv,
                'Predicted - Knn': pred_knn_inv,
            })

            df_results['Date'] = pd.to_datetime(df_results['Date'])
            df_results.set_index('Date', inplace=True)

            # 📦 Monthly aggregation (excluding future forecast)
            monthly_df = df_results.resample('M').sum(min_count=1)

            # 📅 Last date of current test data
            last_date = df_results.index.max()

            # 🔮 Forecast next 30 and 7 days
            preds_xgb_30, preds_rf_30, preds_knn_30 = self.predict_next_days(days=30)
            preds_xgb_7, preds_rf_7, preds_knn_7 = self.predict_next_days(days=7)

            # ➕ Add next month forecast
            future_date_30 = last_date + pd.Timedelta(days=30)
            next_month = future_date_30 + pd.offsets.MonthEnd(0)

            monthly_df.loc[next_month] = {
                'Actual': None,
                'Predicted - XGBoost': np.nansum(preds_xgb_30),
                'Predicted - Random Forest': np.nansum(preds_rf_30),
                'Predicted - Knn': np.nansum(preds_knn_30)
            }

            # ➕ (Optional) Add 1-week forecast to monthly_df as well (for reference)
            future_date_7 = last_date + pd.Timedelta(days=7)
            week_label_date = future_date_7 + pd.offsets.Week(weekday=6)  # Next Sunday
            monthly_df.loc[week_label_date] = {
                'Actual': None,
                'Predicted - XGBoost': np.nansum(preds_xgb_7),
                'Predicted - Random Forest': np.nansum(preds_rf_7),
                'Predicted - KNN': np.nansum(preds_knn_7)
            }

            # 📝 Save results
            df_results.to_excel(os.path.join(save_dir, 'results_daily.xlsx'))
            monthly_df.to_excel(os.path.join(save_dir, 'results_monthly.xlsx'))

            # 📁 Save next 30 days forecast
            df_next_30 = pd.DataFrame({
                'Date': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30),
                'Predicted - XGBoost': preds_xgb_30,
                'Predicted - Random Forest': preds_rf_30,
                'Predicted - KNN': preds_knn_30
            }).set_index('Date')
            df_next_30.to_excel(os.path.join(save_dir, 'results_next_30_days.xlsx'))

            # 📁 Save next 7 days forecast
            df_next_7 = pd.DataFrame({
                'Date': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7),
                'Predicted - XGBoost': preds_xgb_7,
                'Predicted - Random Forest': preds_rf_7,
                'Predicted - KNN': preds_knn_7
                
            }).set_index('Date')
            df_next_7.to_excel(os.path.join(save_dir, 'results_next_7_days.xlsx'))

            # ✅ Console log
            print(f"\n📁 Daily results saved to: {save_dir}/results_daily.xlsx")
            print(f"📁 Monthly results saved to: {save_dir}/results_monthly.xlsx (including next month + next week forecasts)")
            print(f"📁 Next 30-day forecast saved to: {save_dir}/results_next_30_days.xlsx")
            print(f"📁 Next 7-day forecast saved to: {save_dir}/results_next_7_days.xlsx")
            print(f"📅 Next month forecast date: {next_month}")
            print(f"📅 1-week forecast date added to monthly_df as: {week_label_date}")
    
    def plot_feature_importance(self, top_n=20):
        
        if self.xgb_model is None:
            print("⚠ XGBoost model has not been trained yet.")
            return

        importances = self.xgb_model.feature_importances_
        feature_names = [f'Lag_{i}' for i in range(1, len(importances)+1)]

        # Sort
        sorted_idx = np.argsort(importances)[::-1][:top_n]
        top_features = np.array(feature_names)[sorted_idx]
        top_importances = importances[sorted_idx]

        plt.figure(figsize=(12, 6))
        plt.barh(top_features[::-1], top_importances[::-1], color='teal')
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features (XGBoost)')
        plt.grid(axis='x')
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self):
        
       

        pred_xgb_inv = self.scaler_y.inverse_transform(self.y_pred_xgb.reshape(-1, 1)).flatten()
        pred_rf_inv = self.scaler_y.inverse_transform(self.y_pred_rf.reshape(-1, 1)).flatten()
        pred_knn_inv = self.scaler_y.inverse_transform(self.y_pred_knn.reshape(-1, 1)).flatten()
        actual_inv = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).flatten()
      
        future_xgb, future_rf, future_knn = self.predict_next_days(days=30)
        future_range = np.arange(len(actual_inv), len(actual_inv) + 30)

        plt.figure(figsize=(14, 6))
        plt.plot(actual_inv, label='Actual', color='black')
        plt.plot(pred_xgb_inv, label='XGBoost Prediction', linestyle='--', color='blue')
        plt.plot(pred_rf_inv, label='Random Forest Prediction', linestyle=':', color='green')
        plt.plot(pred_knn_inv, label='KNN Prediction', linestyle=':', color='red')

       
        plt.plot(future_range, future_xgb, label='Future XGBoost Forecast', linestyle='--', color='deepskyblue')
        plt.plot(future_range, future_rf, label='Future RF Forecast', linestyle=':', color='green')
        plt.plot(future_range, future_knn, label='Future Knn Forecast', linestyle=':', color='red')

        plt.title('📈 Actual vs Predicted (Test + Next 30 Days)')
        plt.xlabel('Time Step')
        plt.ylabel(self.target_column)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()