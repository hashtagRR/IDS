import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.ensemble import StackingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from colorama import init, Fore, Style

import warnings
warnings.filterwarnings('ignore')

import time

class IntrusionDetectionSystem:
    def __init__(self, data_file, test_size=0.25, random_state=50):
        self.data_file = data_file
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self):
        columns = ['id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 
                   'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 
                   'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 
                   'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 
                   'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
                   'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 
                   'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat', 'label'
                   ]


        self.raw_data = pd.read_csv(self.data_file, usecols=columns, index_col=None)
    
    def encode_non_numerics(self):
        # Identify categorical columns
        categorical_columns = ['proto', 'service', 'state', 'attack_cat']

        # One-hot encode categorical columns
        encoder = OneHotEncoder(drop='first', sparse=False)
        encoded_features = encoder.fit_transform(self.raw_data[categorical_columns])

        # Replace the original categorical columns with the encoded features
        self.raw_data = pd.concat([self.raw_data, pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))], axis=1)
        self.raw_data = self.raw_data.drop(columns=categorical_columns)

    def split_data(self):
        x = self.raw_data.drop(columns=["label"])
        y = self.raw_data["label"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)

    def normalize_dataset(self):
        scaler = MinMaxScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)

    def train_decision_tree_classifier(self):
        type = "Decision Tree"
        clf_dt = DecisionTreeClassifier()
        clf_dt.fit(self.x_train, self.y_train)
        y_pred_dt = clf_dt.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_dt, type)

    def train_random_forest_classifier(self):
        type = "Random Forest"
        clf_rf = RandomForestClassifier(n_estimators=45)
        clf_rf.fit(self.x_train, self.y_train)
        y_pred_rf = clf_rf.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_rf, type)

    def train_naive_bayes_classifier(self):
        type = "Naive Bayes"
        clf_nb = GaussianNB()
        clf_nb.fit(self.x_train, self.y_train)
        y_pred_nb = clf_nb.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_nb, type)

    def train_logistic_regression_classifier(self):
        type = "Logistic Regression"
        clf_lr = LogisticRegression(random_state=50, solver='lbfgs', max_iter=300)
        clf_lr.fit(self.x_train, self.y_train)
        y_pred_lr = clf_lr.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_lr, type)

    def train_knn_classifier(self):
        type = "KNN"
        clf_knn = KNeighborsClassifier(n_neighbors=100)
        clf_knn.fit(self.x_train, self.y_train)
        y_pred_knn = clf_knn.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_knn, type)

    def train_svm_classifier(self):
        type = "SVM"
        clf_svm = svm.SVC()
        clf_svm.fit(self.x_train, self.y_train)
        y_pred_svm = clf_svm.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_svm, type)

    def train_AB_classifier(self):
        type = "Ada Boost"
        clf_ab = AdaBoostClassifier(n_estimators=50, learning_rate=1)
        clf_ab = clf_ab.fit(self.x_train, self.y_train)
        y_pred_ab = clf_ab.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_ab, type)

    def train_gb_classifier(self):
        type = "Gradient Boost"
        clf_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        clf_gb = clf_gb.fit(self.x_train, self.y_train)
        y_pred_gb = clf_gb.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_gb, type)

    def train_xgb_classifier(self):
        type = "XG Boost"
        clf_xgb = xgb.XGBClassifier(n_jobs=8)
        clf_xgb =clf_xgb.fit(self.x_train, self.y_train)
        y_pred_xgb = clf_xgb.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_xgb, type)

    def train_voting_classifier(self):
        type = "Voting > RF, DT, XGB"
        clf_xgb = xgb.XGBClassifier(n_jobs=4)
        clf_rf = RandomForestClassifier(random_state=1)
        clf_dt = DecisionTreeClassifier()
        clf_v = VotingClassifier(estimators=[('xgb', clf_xgb), ('rf', clf_rf), ('dt', clf_dt)], voting='soft')
        clf_v = clf_v.fit(self.x_train, self.y_train)
        y_pred_v = clf_v.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_v, type)

    def train_stacking_classifier(self):
        type = "Voting > RF, DT, XGB"
        clf_xgb = xgb.XGBClassifier(n_jobs=4)
        clf_rf = RandomForestClassifier(random_state=1)
        clf_dt = DecisionTreeClassifier()
        
        clf_s = StackingClassifier(estimators=[('xgb', clf_xgb), ('rf', clf_rf), ('dt', clf_dt)], final_estimator=LogisticRegression())
        clf_s = clf_s.fit(self.x_train, self.y_train)
        y_pred_s = clf_s.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_s, type)


    def evaluation(self, y_test, y_pred, type):
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
    
    def run(self):
        
        self.load_data()
        self.encode_non_numerics()
        self.split_data()
        #self.normalize_dataset()
        
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
    data_file = "E:\\stuff\\DS\\NUSW-NB15\\UNSW_NB15_training-set.csv"
    ids = IntrusionDetectionSystem(data_file)
    ids.run()
    