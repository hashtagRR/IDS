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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualization
from sklearn.tree import export_graphviz
import graphviz

import warnings
warnings.filterwarnings('ignore')

import time

class IntrusionDetectionSystem:
    def __init__(self, data_file, test_size=0.25, random_state=50):
        self.data_file = data_file
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self):
        columns =[
            "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
            "Fwd Packets Length Total", "Bwd Packets Length Total", "Fwd Packet Length Max",
            "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
            "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
            "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
            "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
            "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
            "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Fwd Header Length",
            "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Packet Length Min",
            "Packet Length Max", "Packet Length Mean", "Packet Length Std",
            "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
            "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "ECE Flag Count", "Down/Up Ratio",
            "Avg Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Subflow Fwd Packets",
            "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init Fwd Win Bytes",
            "Init Bwd Win Bytes", "Fwd Act Data Packets", "Fwd Seg Size Min", "Active Mean",
            "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Y"
         ]
        self.raw_data = pd.read_csv(self.data_file, usecols=columns, index_col=None)

    def split_data(self):
        x = self.raw_data.drop(columns=["Y"])
        y = self.raw_data["Y"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)

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
        clf_lr = xgb.XGBClassifier(n_jobs=4)
        clf_rf = RandomForestClassifier(random_state=1)
        clf_dt = DecisionTreeClassifier()
        clf_v = VotingClassifier(estimators=[('lr', clf_lr), ('rf', clf_rf), ('gnb', clf_dt)], voting='soft')
        clf_v = clf_v.fit(self.x_train, self.y_train)
        y_pred_v = clf_v.predict(self.x_test)
        self.evaluation(self.y_test, y_pred_v, type)

    def evaluation(self, y_test, y_pred, type):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        #confusion = confusion_matrix(y_test, y_pred)

        print(type)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        #print("Confusion Matrix:\n", confusion)
        print(" ")

    def run(self):
        start_time = time.time()
        self.load_data()
        self.split_data()

        self.train_decision_tree_classifier()
        self.train_random_forest_classifier()
        self.train_naive_bayes_classifier()
        self.train_logistic_regression_classifier()
        self.train_knn_classifier()
        self.train_svm_classifier()
        self.train_AB_classifier()
        self.train_gb_classifier()
        self.train_xgb_classifier()
        self.train_voting_classifier()

        end_time = time.time()
        program_time = end_time - start_time

        print(f'Program Time: {program_time} seconds')

if __name__ == "__main__":
    #data_file = "E:\stuff\DS\CICIDS2018.csv"
    data_file = "E:\stuff\DS\CICIDS2018-Balanced.csv"
    ids = IntrusionDetectionSystem(data_file)
    ids.run()
    