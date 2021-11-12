#Only for creating and dumping the model

from sklearn.externals import joblib

from train_classifier import TrainClassifier
if __name__ == "__main__":
    model = TrainClassifier()
    model.load_data()
    model.create_pipe()
    joblib.dump(model, 'models/model.pkl')