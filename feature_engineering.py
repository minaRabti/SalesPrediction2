import pandas as pd
import numpy as np
import jdatetime


class FeatureEngineer:
    
    def __init__(self, df: pd.DataFrame, date_column: str = 'SALES_DOCUMENT_DATE'):
        self.df = df.copy()
        self.date_column = date_column
        

        self.df[self.date_column] = pd.to_datetime(self.df[self.date_column], dayfirst=True)

    def _get_persian_season(self, j_month):
        
        if j_month in [1, 2, 3]:
            return '1'
        elif j_month in [4, 5, 6]:
            return '2'
        elif j_month in [7, 8, 9]:
            return '3'
        else:
            return '4'

    def add_jalali_month_and_season(self):
        jalali_months = []
        jalali_seasons = []

        for d in self.df[self.date_column]:
            j_date = jdatetime.date.fromgregorian(date=d)
            j_month = j_date.month
            jalali_months.append(j_month)
            jalali_seasons.append(self._get_persian_season(j_month))

        self.df['Jalali_Month'] = jalali_months
        self.df['Jalali_Season'] = jalali_seasons
        return self

    def create_rolling_average(self, column='GROSS_PRICE'):
        
        self.df['MA_7'] = self.df[column].rolling(window=8).mean().fillna(method='bfill')
        self.df['MA_30'] = self.df[column].rolling(window=16).mean().fillna(method='bfill')
        return self

    def create_lag_features(self, column='GROSS_PRICE', lag=15):
        for i in range(1, lag+1):
            self.df[f'{column}lag{i}'] = self.df[column].shift(i)
        self.df.dropna(inplace=True)
        return self 

    def create_day_of_week(self):
        
        self.df['weekday_num'] = self.df[self.date_column].dt.weekday
        self.df['weekday_name'] = self.df[self.date_column].dt.day_name()
        self.df['is_weekend'] = self.df['weekday_name'].isin(['Thursday', 'Friday']).astype(int)
        del self.df['weekday_name']

        
    def create_feature_isZeroSale(self):
        
        
        m = self.df['GROSS_PRICE'].mean(axis=0)
        st = self.df['GROSS_PRICE'].std(axis=0)
        th_low = self.df[self.df['GROSS_PRICE'] > 0]['GROSS_PRICE'].quantile(0.01)
        
        self.df['was_zero_sales'] = 2       

        mask1 = (self.df['is_weekend'] == 1) & (self.df['GROSS_PRICE'] > 0)
        self.df.loc[mask1, 'was_zero_sales'] = 0
        
        mask2 = (self.df['is_weekend'] == 0) & (self.df['GROSS_PRICE'] == 0)
        self.df.loc[mask2, 'was_zero_sales'] = 1
        
        self.df.loc[mask2, 'GROSS_PRICE'] = np.abs(round(th_low))
        #self.df = self.df[self.df['GROSS_PRICE'] <= th_high]


        return self
    
    def add_high_y_flag(self, method="std", quantile_value=0.95, std_multiplier=3, flag_col_name="y_high_flag"):
   
        if method=="quantile":
            threshold = self.df['GROSS_PRICE'].quantile(quantile_value)
            print(f"✅ Threshold انتخاب شده (quantile {quantile_value*100:.0f}%): {threshold:.2f}")
        elif method=="std":
            mean = self.df['GROSS_PRICE'].mean()
            std = self.df['GROSS_PRICE'].std()
            threshold = mean + std_multiplier*std
            print(f"✅ Threshold انتخاب شده (mean + {std_multiplier}*std): {threshold:.2f}")
        else:
            raise ValueError("method باید 'quantile' یا 'std' باشد.")
        
        self.df["Sale_th"] = (self.df['GROSS_PRICE'] > threshold).astype(int)

        return self


    def transform(self, lag_column='GROSS_PRICE', lag=30):

        self.add_jalali_month_and_season()
        self.create_rolling_average()
        self.create_day_of_week()
        #self.create_feature_isZeroSale()
        #self.add_high_y_flag()
        #self.add_DollarCarrency()
        # self.create_lag_features(column=lag_column, lag=lag)
        return self.df