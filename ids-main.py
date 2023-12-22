import pandas as pd
import sys
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Kn
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
from imblearn.under_sampling import RandomUnderSampler


from colorama import init, Fore, Style

import warnings
warnings.filterwarnings('ignore')

import time

class IntrusionDetectionSystem:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file

    def load_data(self):
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
         # Identify categorical columns
        categorical_columns = ['proto', 'service', 'state', 'attack_cat']
        numerical_columns = self.raw_train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

        encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')

        # One-hot encode categorical columns
        encoded_train_features = encoder.fit_transform(self.raw_train_data[categorical_columns])
        encoded_test_features = encoder.transform(self.raw_test_data[categorical_columns])

        # Replace the original categorical columns with the encoded features for training data
        self.train_data = pd.concat([self.raw_train_data[numerical_columns], pd.DataFrame(encoded_train_features, columns=encoder.get_feature_names_out(categorical_columns))], axis=1)
        self.test_data = pd.concat([self.raw_test_data[numerical_columns], pd.DataFrame(encoded_test_features, columns=encoder.get_feature_names_out(categorical_columns))], axis=1)
       
        # Replace the original categorical columns with the encoded features for testing data
        #self.train_data = self.train_data.drop(columns=categorical_columns)
        #self.test_data = self.test_data.drop(columns=categorical_columns)

    def split_data(self):
        x_train = self.train_data.drop(columns=["label"])
        y_train = self.train_data["label"]

        x_test = self.test_data.drop(columns=["label"])
        y_test = self.test_data["label"]

        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test

    def normalize_dataset(self):
        scaler = MinMaxScaler()

        # Separate numerical columns for normalization
        numerical_columns_train = self.train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numerical_columns_test = self.test_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Normalize only the numerical columns
        self.x_train[numerical_columns_train] = scaler.fit_transform(self.train_data[numerical_columns_train])
        self.x_test[numerical_columns_test] = scaler.transform(self.test_data[numerical_columns_test])

    def balance_dataset(self):
        
        print('Original dataset shape %s' % Counter(self.y_train))
        oversample = SMOTE()
        self.x_train, self.y_train = oversample.fit_resample(self.x_train, self.y_train)
        #u_sample = RandomUnderSampler(sampling_strategy='majority')
        #u_sample = RandomUnderSampler(sampling_strategy=1)
        #self.x_train, self.y_train = u_sample.fit_resample(self.x_train, self.y_train)
        print('Resampled dataset shape %s' % Counter(self.y_train))

    def train_decision_tree_classifier(self):
        clf_dt = DecisionTreeClassifier()
        clf_dt.fit(self.x_train, self.y_train)
        y_pred_dt = clf_dt.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_dt)
        self.cross_validation(self.x_train, self.y_train, clf_dt)

    def train_random_forest_classifier(self):
        clf_rf = RandomForestClassifier(n_estimators=45)
        clf_rf.fit(self.x_train, self.y_train)
        y_pred_rf = clf_rf.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_rf)
        self.cross_validation(self.x_train, self.y_train, clf_rf)

    def train_naive_bayes_classifier(self):
        clf_nb = GaussianNB()
        clf_nb.fit(self.x_train, self.y_train)
        y_pred_nb = clf_nb.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_nb)
        self.cross_validation(self.x_train, self.y_train, clf_nb)

    def train_logistic_regression_classifier(self):
        clf_lr = LogisticRegression(random_state=50, solver='lbfgs', max_iter=300)
        clf_lr.fit(self.x_train, self.y_train)
        y_pred_lr = clf_lr.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_lr)
        self.cross_validation(self.x_train, self.y_train, clf_lr)

    def train_knn_classifier(self):
        clf_knn = KNeighborsClassifier(n_neighbors=100)
        clf_knn.fit(self.x_train, self.y_train)
        y_pred_knn = clf_knn.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_knn)
        self.cross_validation(self.x_train, self.y_train, clf_knn)

    def train_svm_classifier(self):
        clf_svm = svm.SVC()
        clf_svm.fit(self.x_train, self.y_train)
        y_pred_svm = clf_svm.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_svm)
        self.cross_validation(self.x_train, self.y_train, clf_svm)

    def train_AB_classifier(self):
        clf_ab = AdaBoostClassifier(n_estimators=50, learning_rate=1)
        clf_ab = clf_ab.fit(self.x_train, self.y_train)
        y_pred_ab = clf_ab.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_ab)
        self.cross_validation(self.x_train, self.y_train, clf_ab)

    def train_gb_classifier(self):
        clf_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        clf_gb = clf_gb.fit(self.x_train, self.y_train)
        y_pred_gb = clf_gb.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_gb)
        self.cross_validation(self.x_train, self.y_train, clf_gb)

    def train_xgb_classifier(self):
        clf_xgb = xgb.XGBClassifier(n_jobs=16, eval_metric='mlogloss')
        clf_xgb = clf_xgb.fit(self.x_train, self.y_train)
        y_pred_xgb = clf_xgb.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_xgb)
        self.cross_validation(self.x_train, self.y_train, clf_xgb)

    def train_voting_classifier(self):
        #type = "Voting > RF, DT, XGB"
        clf_xgb = xgb.XGBClassifier(n_jobs=16, eval_metric='mlogloss')
        clf_rf = RandomForestClassifier(random_state=1)
        clf_dt = DecisionTreeClassifier()
        clf_v = VotingClassifier(estimators=[('xgb', clf_xgb), ('rf', clf_rf), ('dt', clf_dt)], voting='soft')
        clf_v = clf_v.fit(self.x_train, self.y_train)
        y_pred_v = clf_v.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_v)
        self.cross_validation(self.x_train, self.y_train, clf_v)

    def train_stacking_classifier(self):
        #type = "Stacking > RF, DT, XGB"
        clf_xgb = xgb.XGBClassifier(n_jobs=16, eval_metric='mlogloss')
        clf_rf = RandomForestClassifier(random_state=1)
        clf_dt = DecisionTreeClassifier()
        
        clf_s = StackingClassifier(estimators=[('xgb', clf_xgb), ('rf', clf_rf), ('dt', clf_dt)], final_estimator=LogisticRegression())
        clf_s = clf_s.fit(self.x_train, self.y_train)
        y_pred_s = clf_s.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_s)
        self.cross_validation(self.x_train, self.y_train, clf_s)

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
        #k_folds = KFold(n_splits = 10)
        sk_folds = StratifiedKFold(n_splits = 10)
        scores = cross_val_score(clf, x_train, y_train, cv = sk_folds)
        print("Cross Validation Scores: ", scores)
        print("Average CV Score: ", scores.mean())
        print("Number of CV Scores used in Average: ", len(scores))

    def run(self):
        
        self.load_data()
        self.encode_non_numerics()
        self.split_data()
        self.normalize_dataset()
        self.balance_dataset()
        
        classifiers = [
            (self.train_decision_tree_classifier, "Decision Tree"),
            (self.train_random_forest_classifier, "Random Forest"),
            (self.train_naive_bayes_classifier, "Naive Bayes"),
            (self.train_logistic_regression_classifier, "Logistic Regression"),
            (self.train_knn_classifier, "KNN"),
            (self.train_svm_classifier, "SVM"),
            (self.train_AB_classifier, "AdaBoost"),
            (self.train_gb_classifier, "Gradient Boosting"),
            (self.train_xgb_classifier, "XGBoost"),
            (self.train_voting_classifier, "Voting"),
            (self.train_stacking_classifier, "Stacking")
             ]
        
        #classifiers = [(self.train_xgb_classifier, "XGB")]

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
    train_file = "/home/azureuser/cloudfiles/code/users/2306143/UNSW_NB15_training-set.csv"
    test_file = "/home/azureuser/cloudfiles/code/users/2306143/UNSW_NB15_testing-set.csv"
    ids = IntrusionDetectionSystem(train_file, test_file)
    ids.run()
