from disasterReportapp import app


if __name__ == '__main__':
    from models.train_classifier import TrainClassifier, StartingVerbExtractor, NewMultiOutput
    app.run()
