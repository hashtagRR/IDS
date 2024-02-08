#Two classifier configuration

import pandas as pd
import sys
import os

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder

from imblearn.under_sampling import EditedNearestNeighbours


from colorama import init, Fore, Style

import warnings
warnings.filterwarnings('ignore')

import time

class Data_Loader:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file

    def load_dataset(self):
        columns = ['id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 
                   'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 
                   'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 
                   'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 
                   'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
                   'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 
                   'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat', 'label'
                   ]

        raw_train_data = pd.read_csv(self.train_file, usecols=columns, index_col=None)
        raw_test_data = pd.read_csv(self.test_file, usecols=columns, index_col=None)

        return raw_train_data, raw_test_data

from imblearn.under_sampling import EditedNearestNeighbours

class Data_Preprocessor:
    def __init__(self, raw_train_data, raw_test_data):
        self.raw_train_data = raw_train_data
        self.raw_test_data = raw_test_data
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    def encode_non_numerics(self):
        non_numerical_columns = self.raw_train_data.select_dtypes(exclude=['float64', 'int64']).columns.tolist()
        numerical_columns = self.raw_train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

        encoded_train_features = self.encoder.fit_transform(self.raw_train_data[non_numerical_columns])
        encoded_test_features = self.encoder.transform(self.raw_test_data[non_numerical_columns])

        x_train_encoded = pd.concat([self.raw_train_data[numerical_columns],
                                     pd.DataFrame(encoded_train_features, columns=self.encoder.get_feature_names_out(non_numerical_columns)),], axis=1)

        # Concatenate numerical columns, encoded categorical columns, and 'attack_cat' for test data
        x_test_encoded = pd.concat([self.raw_test_data[numerical_columns],
                                    pd.DataFrame(encoded_test_features, columns=self.encoder.get_feature_names_out(non_numerical_columns)), ], axis=1)

        return x_train_encoded, x_test_encoded
    
    def normalize_dataset(self, x_train_encoded, x_test_encoded):
        scaler = MinMaxScaler()

        numerical_columns_train = x_train_encoded.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numerical_columns_test = x_test_encoded.select_dtypes(include=['float64', 'int64']).columns.tolist()

        x_train_normalized = pd.DataFrame(scaler.fit_transform(x_train_encoded[numerical_columns_train]), columns=numerical_columns_train, index=x_train_encoded.index)

        x_test_normalized = pd.DataFrame(scaler.transform(x_test_encoded[numerical_columns_train]), columns=numerical_columns_train, index=x_test_encoded.index)

        # Drop the original numerical columns from the dataset
        x_train_n = x_train_encoded.drop(columns=numerical_columns_train)
        x_test_n = x_test_encoded.drop(columns=numerical_columns_test)

        # Add the encoded numerical columns to the dataset
        x_train_n = pd.concat([x_train_normalized, x_train_n], axis=1)
        x_test_n = pd.concat([x_test_normalized, x_test_n], axis=1)

        return x_train_n, x_test_n

    def split_dataset(self, x_train_n, x_test_n):
        columns_to_drop = ['id', 'label', 'attack_cat']

        x_train = x_train_n.drop(columns=columns_to_drop)
        y_train = x_train_n['label']
        y_train_ac = x_train_n['attack_cat']

        x_test = x_test_n.drop(columns=columns_to_drop)
        y_test = x_test_n['label']
        y_test_ac = x_test_n['attack_cat']

        return x_train, y_train, y_train_ac, x_test, y_test, y_test_ac

    def balance_dataset(self, x_train, y_train, y_train_ac, x_test, y_test, y_test_ac):

        y_train_ac = y_train_ac.astype(int)
        y_test_ac = y_test_ac.astype(int)

        algo = EditedNearestNeighbours()

        # Resample training data
        x_train_balanced, y_train_balanced = algo.fit_resample(x_train, y_train)
        x_train_ac_balanced, y_train_ac_balanced = algo.fit_resample(x_train, y_train_ac)

        # Resample testing data using the same resampler instance
        x_test_balanced, y_test_balanced = algo.fit_resample(x_test, y_test)
        x_test_ac_balanced, y_test_ac_balanced = algo.fit_resample(x_test, y_test_ac)

        return x_train_balanced, y_train_balanced, x_train_ac_balanced, y_train_ac_balanced, x_test_balanced, y_test_balanced, x_test_ac_balanced, y_test_ac_balanced


    def preprocess_data(self):
        # Encode non-numeric values using Ordinal Encoder
        x_train_encoded, x_test_encoded = self.encode_non_numerics()
        print("Encoding Completed \n")

        # Normalize the dataset using MinMaxScaler
        x_train_n, x_test_n = self.normalize_dataset(x_train_encoded, x_test_encoded)
        print("Normalizing Completed \n")

        # Split dataset
        x_train, y_train, y_train_ac, x_test, y_test, y_test_ac = self.split_dataset(x_train_n, x_test_n)
        print("Split Completed \n")

        # Balance datasets using ENN
        x_train_balanced, y_train_balanced, x_train_ac_balanced, y_train_ac_balanced, x_test_balanced, y_test_balanced, x_test_ac_balanced, y_test_ac_balanced = self.balance_dataset(
            x_train, y_train, y_train_ac, x_test, y_test, y_test_ac)
        print("Balancing Completed \n")

        return x_train_balanced, y_train_balanced, x_test_balanced, y_test_balanced, x_train_ac_balanced, y_train_ac_balanced, x_test_ac_balanced, y_test_ac_balanced




class Evaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def evaluate(self):
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred, average='weighted')
        recall = recall_score(self.y_true, self.y_pred, average='weighted')
        f1 = f1_score(self.y_true, self.y_pred, average='weighted')

        results = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }

        return results





class Attack_Classifier:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train_attack_classifier(self):
        print(Fore.RED + "Attack Classifier\n" + Style.RESET_ALL, end='', flush=True,)
        clf_rf = RandomForestClassifier(n_estimators=45)
        clf_rf.fit(self.x_train, self.y_train)
        y_pred_rf = clf_rf.predict(self.x_test)
        
        evaluator = Evaluator(self.y_test, y_pred_rf)
        results = evaluator.evaluate()

        for metric, value in results.items():
            print(f"{metric}: {value}")






class Type_Classifier:
    def __init__(self, x_train_ac, y_train_ac, x_test_ac, y_test_ac):
        self.x_train_ac = x_train_ac
        self.y_train_ac = y_train_ac
        self.x_test_ac = x_test_ac
        self.y_test_ac = y_test_ac

        print("\n" * 2)  # Add one-line space
        print("x_train_ac columns:")
        print("\n".join(self.x_train_ac.columns))
        print("\n" * 2)  # Add one-line space

        print("y_train_ac columns:")
        print(self.y_train_ac.name)  # Print the name of the Series
        print("\n" * 2)  # Add one-line space


        print("x_test_ac columns:")
        print("\n".join(self.x_test_ac.columns))
        print("\n" * 2)  # Add one-line space

        print("y_test_ac columns:")
        print(self.y_test_ac.name)
        print("\n" * 2)  # Add one-line space

    def train_type_classifier(self):
        print(Fore.RED + "Attack Type Classifier\n" + Style.RESET_ALL, end='', flush=True)
        clf_xgb = xgb.XGBClassifier(learning_rate=0.01, n_estimators=100, max_depth=5, subsample=1, colsample_bytree=0.5,
                                     objective='multi:softprob', num_class=9, eval_metric='logloss')
        clf_xgb = clf_xgb.fit(self.x_train_ac, self.y_train_ac)
        y_pred_xgb = clf_xgb.predict(self.x_test_ac)
        
        evaluator = Evaluator(self.y_test_ac, y_pred_xgb)
        results = evaluator.evaluate()

        for metric, value in results.items():
            print(f"{metric}: {value}")









if __name__ == "__main__":
    train_file = "/home/azureuser/cloudfiles/code/users/IDS/UNSW_NB15_training-set.csv"
    test_file = "/home/azureuser/cloudfiles/code/users/IDS/UNSW_NB15_testing-set.csv"

    data_loader = Data_Loader(train_file, test_file)
    raw_train_data, raw_test_data = data_loader.load_dataset()

    data_preprocessor = Data_Preprocessor(raw_train_data, raw_test_data)
    x_train_balanced, y_train_balanced, x_test_balanced, y_test_balanced, x_train_ac_balanced, y_train_ac_balanced, x_test_ac_balanced, y_test_ac_balanced = data_preprocessor.preprocess_data()

    #attack_classifier = Attack_Classifier(x_train_balanced, y_train_balanced, x_test_balanced, y_test_balanced)
    #attack_classifier.train_attack_classifier()

    type_classifier = Type_Classifier(x_train_ac_balanced, y_train_ac_balanced, x_test_ac_balanced, y_test_ac_balanced)
    type_classifier.train_type_classifier()
