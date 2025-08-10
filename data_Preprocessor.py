from typing import List
import pandas as pd
import numpy as np


class _PrivateCleaner:
    """
    Inner class to perform private operations such as removing irrelevant attributes or filling in missing values.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
       
        

    def _remove_high_missing_and_irrelevant(self):

        high_missing_cols = ['Type_Ref_ID','AccountGroupName','CustomerType','CustomerName']
        irrelevant_cols = ['CurrencyID', 'CompanyID', 'SystemCode', 'Mat_SystemCode', 'ID', 'MATL_GROUP___T',
                           'HijriMonthName','SeasonName','CompanyPlantID','MaterialID','AlternateStageID',
                           'Creator','MATERIAL','BillTypeCodeGroup']


        self._df.drop(columns = high_missing_cols + irrelevant_cols, inplace=True, errors='ignore')

    def _filter_non_food_products(self):

        self._df = self._df[self._df['MATL_GROUP'] != 'GC-270901']
        self._df['MATL_GROUP'] = self._df['MATL_GROUP'].fillna('FP-120102')

    def clean(self):

        self._remove_high_missing_and_irrelevant()
        self._filter_non_food_products()
        return self._df


class SalesPreprocessor:
    """
    Main class for preprocessing sales data
   """

    def __init__(self, file_path: str, dollarpath_path: str):
        self._file_path = file_path
        self.dollardata_path = dollarpath_path
        self._df = None
        
        self.daily_df = None

    def load_data(self):
        self._df = pd.read_excel(self._file_path)
        
        
        return self
          
   

    def _add_return_flag(self):
        return_customers = self._df[self._df['Type_Sale'] != 1]['CustomerID'].unique()
        self._df['HasReturn'] = self._df['CustomerID'].isin(return_customers).astype(int)

    def _encode_features(self):
        
        one_hot_mat = pd.get_dummies(self._df['MATL_GROUP'], prefix='MATL_GROUP').astype(int)
        
        one_hot_acc = pd.get_dummies(self._df['AccountType'], prefix='AccountType').astype(int)
        self._df = pd.concat([self._df, one_hot_acc + one_hot_mat], axis=1)

    def _filter_sales_docs(self):

        self._df = self._df[self._df['Type_Sale'] == 1].copy()
        self._df['SALES_DOCUMENT_DATE'] = pd.to_datetime(self._df['SALES_DOCUMENT_DATE'])
        self._df.set_index('SALES_DOCUMENT_DATE', inplace=True)

    def _aggregate_daily(self):
        
        mat_group_cols = [col for col in self._df.columns if col.startswith('MATL_GROUP')]
        
        account_type_cols = [col for col in self._df.columns if col.startswith('AccountType')]
        
        li = ['AccountType','MATL_GROUP']
        

        agg_dict = {
            'GROSS_PRICE': 'sum',
            'DECREASE': 'sum',
            'INCREASE': 'sum',
            'BillingQty': 'sum',
            
            'HasReturn': lambda x: x.mode()[0] if not x.mode().empty else None
        }

        for col in mat_group_cols + account_type_cols:
             agg_dict[col] = 'sum'
        for col in mat_group_cols + account_type_cols:
             agg_dict[col] = 'sum'
             
        #for col in  account_type_cols:
             #agg_dict[col] = 'sum'
        #for col in account_type_cols:
             #agg_dict[col] = 'sum'     
             
 
        daily = self._df.resample('D').agg(agg_dict).reset_index()
        daily['HasReturn'] = daily['HasReturn'].fillna(2)
        
        #print('the null is:', daily['dollarPrice'].isnull().sum())      
        daily = daily[daily['GROSS_PRICE'] >= 0]
        
        daily = daily.drop(li, axis=1)

        self.daily_df = daily
        
        
    def merge_on_date(self, sales_date_col='SALES_DOCUMENT_DATE', dollar_date_col='Datemiladi'):
        
        self.daily_df[sales_date_col] = pd.to_datetime(self.daily_df[sales_date_col])
        dollarData = pd.read_excel(self.dollardata_path)
        
        dollarData[dollar_date_col] = pd.to_datetime(dollarData[dollar_date_col])

       
        merged_df = pd.merge(
            self.daily_df,
            dollarData,
            left_on=sales_date_col,
            right_on=dollar_date_col,
            how='left'
        )

       
        merged_df.drop(columns=[dollar_date_col], inplace=True)

        return merged_df
    

    def preprocess(self):
        
        self._add_return_flag()

        cleaner = _PrivateCleaner(self._df)
        self._df = cleaner.clean()

        self._encode_features()
        self._filter_sales_docs()
        self._aggregate_daily()
        self.daily_df = self.merge_on_date()
        return self.daily_df