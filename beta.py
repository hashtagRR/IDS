#RFE for the 1st classifier
#OH encoder.

import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import xgboost as xgb

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import randint
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from collections import Counter
from contextlib import contextmanager

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, EditedNearestNeighbours


from colorama import init, Fore, Style

import warnings
warnings.filterwarnings('ignore')

import time

class IntrusionDetectionSystem:
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

        self.raw_train_data = pd.read_csv(self.train_file, usecols=columns, index_col=None)
        self.raw_test_data = pd.read_csv(self.test_file, usecols=columns, index_col=None)
      
    def encode_non_numerics(self):
        # Identify non numerical columns
        non_numerical_columns = self.raw_train_data.select_dtypes(exclude=['float64', 'int64']).columns.tolist()
        
        # Drop 'attack_cat' from the list if present
        if 'attack_cat' in non_numerical_columns:
            non_numerical_columns.remove('attack_cat')

        # Identify numerical columns
        numerical_columns = self.raw_train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Initialize OneHotEncoder with specified parameters
        encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        #encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

        # One-hot encode categorical columns for training and testing data
        encoded_train_features = encoder.fit_transform(self.raw_train_data[non_numerical_columns])
        encoded_test_features = encoder.transform(self.raw_test_data[non_numerical_columns])

       # Concatenate numerical columns, encoded categorical columns, and 'attack_cat' for training data
        self.train_data = pd.concat([self.raw_train_data[numerical_columns], 
                                pd.DataFrame(encoded_train_features, columns=encoder.get_feature_names_out(non_numerical_columns)),
                                self.raw_train_data['attack_cat']
                                ], axis=1)

        # Concatenate numerical columns, encoded categorical columns, and 'attack_cat' for test data
        self.test_data = pd.concat([self.raw_test_data[numerical_columns], 
                               pd.DataFrame(encoded_test_features, columns=encoder.get_feature_names_out(non_numerical_columns)),
                               self.raw_test_data['attack_cat']
                               ], axis=1)
       
    def split_dataset(self):
        columns_to_drop = ['id', 'label', 'attack_cat']

        x_train = self.train_data.drop(columns=columns_to_drop)
        y_train = self.train_data['label']

        #y_train_attack_cat = self.train_data['attack_cat']
       
        x_test = self.test_data.drop(columns=columns_to_drop)
        y_test = self.test_data['label']
        #y_test_attack_cat = self.test_data['attack_cat']

        self.x_train, self.x_test = x_train, x_test
        self.y_train, self.y_test = y_train, y_test
        #self.y_train_attack_cat, self.y_test_attack_cat = y_train_attack_cat, y_test_attack_cat


    def normalize_dataset(self):
        scaler = MinMaxScaler()

        # Separate numerical columns for normalization
        numerical_columns_train = self.x_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numerical_columns_test = self.x_test.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Normalize only the numerical columns
        x_train_normalized = pd.DataFrame(scaler.fit_transform(self.x_train[numerical_columns_train]),columns=numerical_columns_train,
                                          index=self.x_train.index)
        
        x_test_normalized = pd.DataFrame(scaler.transform(self.x_test[numerical_columns_train]),columns=numerical_columns_train,
                                          index=self.x_test.index)

        # Drop the original numerical columns from the datasets
        self.x_train = self.x_train.drop(columns=numerical_columns_train)
        self.x_test = self.x_test.drop(columns=numerical_columns_test)
        
        # Concatenate numerical columns and encoded categorical columns for train and test data
        self.x_train = pd.concat([x_train_normalized, self.x_train], axis=1)
        self.x_test = pd.concat([x_test_normalized, self.x_test], axis=1)

    def balance_dataset(self):
        
        print('Original dataset shape %s' % Counter(self.y_train))
        algo = EditedNearestNeighbours() 
        self.x_train, self.y_train = algo.fit_resample(self.x_train, self.y_train)
        print('Resampled dataset shape %s' % Counter(self.y_train))

    def train_voting_classifier(self):
        #type = "Voting > RF, DT, XGB"
        clf_ab = AdaBoostClassifier(n_estimators=50, learning_rate=1)
        clf_rf = RandomForestClassifier(random_state=1)
        clf_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        clf_v = VotingClassifier(estimators=[('xgb', clf_ab), ('rf', clf_rf), ('dt', clf_gb)], voting='soft')
        clf_v = clf_v.fit(self.x_train, self.y_train)
        y_pred_v = clf_v.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_v)
        self.cross_validation(self.x_train, self.y_train, clf_v)

    def train_rf_classifier(self):
        clf_rf = RandomForestClassifier(n_estimators=45)
        clf_rf.fit(self.x_train, self.y_train)
        y_pred_rf = clf_rf.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_rf)
        self.cross_validation(self.x_train, self.y_train, clf_rf)
        
        # Get feature importances
        feature_importances = clf_rf.feature_importances_

        # Create a DataFrame to store feature names and their importance scores
        feature_ranking = pd.DataFrame({'Feature': self.x_train.columns, 'Importance': feature_importances})
        # Sort features by importance in descending order
        feature_ranking = feature_ranking.sort_values(by='Importance', ascending=False)

        # Initialize the classifier
        clf_rf = RandomForestClassifier(n_estimators=45)

        from sklearn.feature_selection import RFE

        # Initialize the classifier
        clf_rf = RandomForestClassifier(n_estimators=45)

        from sklearn.feature_selection import RFE

        # Initialize the classifier
        clf_rf = RandomForestClassifier(n_estimators=45)

        # Set the range of feature counts to test
        feature_counts_to_test = range(1, len(self.x_train.columns) + 1)

        for num_features in feature_counts_to_test:
            # Initialize RFE with the classifier and the current number of features to select
            rfe = RFE(estimator=clf_rf, n_features_to_select=num_features)

            # Fit RFE
            rfe.fit(self.x_train, self.y_train)

            # Get the selected features
            selected_features = self.x_train.columns[rfe.support_]

            # Train the classifier with the selected features
            clf_rf_rfe = RandomForestClassifier(n_estimators=45)
            clf_rf_rfe.fit(self.x_train[selected_features], self.y_train)

            # Make predictions on the test set
            y_pred_rf_rfe = clf_rf_rfe.predict(self.x_test[selected_features])

            # Print results
            print(f"Number of features: {num_features}")
            print(f"Selected features: {selected_features}")
            self.evaluation(self.y_test, y_pred_rf_rfe)
            print("=" * 50)

            num_features =num_features+5





    def evaluation(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        #confusion = confusion_matrix(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1 )
        #print("Confusion Matrix:\n", confusion)
    
    def cross_validation(self, x_train, y_train, clf):
        '''k_folds = KFold(n_splits = 10)
        sk_folds = StratifiedKFold(n_splits = 10)
        scores = cross_val_score(clf, x_train, y_train, cv = sk_folds)
        print("Cross Validation Scores: ", scores)
        print("Average CV Score: ", scores.mean())
        print("Number of CV Scores used in Average: ", len(scores))'''

    def run(self):
        
        self.load_dataset()
        self.encode_non_numerics()
        self.split_dataset()
        self.normalize_dataset()
        self.balance_dataset()
        
        #classifiers = [(self.train_voting_classifier, "Voting")]
        classifiers = [(self.train_rf_classifier, "RF")]

        for classifier, classifier_type in classifiers:
            # Timing measurement before training
            start_time = time.time()
            print(Fore.RED +f"{classifier_type}\n"+ Style.RESET_ALL, end='', flush=True,)
            classifier()
            
            # Timing measurement after training
            end_time = time.time()
            program_time = end_time - start_time
            print(Fore.GREEN +f'Program Time: {program_time} seconds \n'+ Style.RESET_ALL)
        
if __name__ == "__main__":
    train_file = "/home/azureuser/cloudfiles/code/users/IDS/UNSW_NB15_training-set.csv"
    test_file = "/home/azureuser/cloudfiles/code/users/IDS/UNSW_NB15_testing-set.csv"
    ids = IntrusionDetectionSystem(train_file, test_file)
    ids.run()
