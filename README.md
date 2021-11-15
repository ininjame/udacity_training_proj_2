# For Data Science project 2 - Creating pipeline for Figure Eight

## Problem statement
The project aims to create a full application to classify disaster responses for relief effort.  
Disasters happen around the world and are always great tragedies.  
As part of the disaster relief effort, agencies around the world encourage, and often receive, an overwhelming amount of messages communicating the current situations of affected sites, by the people who are caught in such disasters.  
It will be very useful to have a model to somehow be able to classify a message into a certain type of disaster responses, to identify the correct needs and required attention, and this is the aim of this solution.  

## Target and work flow
Training data is provided consisting of a list of disaster response messages, and their corresponding category.  
The solution will process this training data, using the processed data to train a predictive model for classifying future messages.  
The solution will also incorporate a way to show some insights related to the provided data, as well as a way for user to input a new message and recieve its corresponding type-ification.  

All in all, the solution consists of 3 main components:
- A pipeline for processing training data and save it to a database
- A pipeline for the predictive model, which will fit the training data, and can predict new messages
- A web interface to show insights into the training data, as well as receive user input and output result visually 

## Understanding input and methodology
The training data provided consists of a list of past disaster response messages, their genres, as well as their types.  
The target for classification is the type of a disaster responses, making the main input the messages themselves, and the problem a typical natural language processing problem.  
Several standard tool in the NTL toolkit proves beneficial, such as tokenization, lemmatization, count vectorizer, tf-idf and so on. These will all be used in the pipeline.  
There are multiple categories, and a message can be labeled as more than 1 category, thus making this a multi-class, multilabel classification problem. For this, scikit-learn's multi-output estimator will be used, with the base being Random Forest Classifier. A small upgrade has been made to the multi-output estimator, however, to reflect an interesting trend in the data, resulting in a custom NewMultiOutput class.

## Installation instruction
1. Clone the git repo to your local computer
2. Open your CLI of choice, make sure it is pointing to the folder "udacity_training_proj_2", or whichever folder the cloned git repo is in.
3. Run the following command in your CLI and wait till an IP is generated: 
```
python disasterReport.py
```
4. Click on the resulting IP address to open the web app in a separate folder.
5. Type a message you want to classify into the text box, click "Classify message" button once you're ready.

## File structure and script explanation
The structure of the project and important files are as follows:
```
|--data
|   |--DisasterResponse.db
|   |--categories.csv
|   |--messages.csv
|
|--disasterReportapp
|   |--templates
|   |--__init__.py
|   |--routes.py
|
|--models
|   |--gen_fig_data.py
|   |--npy files (data for graph)
|   |--process_data.py
|   |--train_classifier.py
|--disasterReport.py

```
A detailed explanation of the folders and files are as follows:

**data**: contains the training data and db for processed data. 
- *categories.csv*: the category labels for the messages
- *messages.csv*: the messages themselves
- *DisasterResponse.db*: SQL database for the processed training data. Created with SQLAlchemy

**disasterReportapp**: contains the main Flask web app files
- *templates*: folder for html files. contains the index.html file for the main page, as well as the dashboard.html file for the classification result dynamic content
- *__init__.py*: initiation file for the app
- *routes.py*: the main web app file. Contains routes for the webpages, as well as code for retrieving graph data, generating and loading the model, and provide the data as variables for the front-end

**models**: contains the main file of the pipeline and model
- *gen_fig_data.py*: code for pre-generating the graph data, as well as returning the data in Plotly graph obj format
- *npy files*: graph data generated
- *process_data.py*: pipeline for processing data and adding it to database. If database does not exist, will create database
- *train_classifier.py*: the model itself. Includes custom class for identifying messages starting with verbs, a customer multi-output classifier, as well as a TrainClassifier class for generating and saving the model.

## Result
Unfortunately, the r2 score for the current model is very low. (<0.1).  
The general prediction is that the result is greatly affected by the skewness in the data, with certain types having very high count, and lines where no count where defined.  
Additionally, tokenization and lemmatization has proven only partially effective in creating inputs that make sense. The writer has identified several cases where shortened forms (e.g. "n't") or punctuation marks were not correctly removed. These may have further lowered the performance of the model.  
Writing a custom transformer to address these issues is the next improvement target for this model, as well as re-balancing the class label composition.  

## Other notes
As a record of experience, the writer encountered the following difficulties during this project
1. Correctly creating a pickle file of the model that can be executed when running the web app
2. General performance when generating insights
3. Internal URL error

For the 1st problem, the writer encountered several "no attributes" error when trying to load a pickle file of the model in the Flask app. The problem was finally identified to be due to the pickle file being generated via code in the model itself ("train_classifier.py"), making the references "__main__" and not pointing to the correct file.  
A solution was found by including the pickle file generator code in the Flask app file instead, and do a check to see if a generated model is already in the main folder.  

For the 2nd problem, the writer encountered slow performance when opening the web app and loading the insights. The reason was identified as the data for the insights being re-generated every time the web app loads it, which takes a lot of time. A solution was found to pre-generate the data, and save them to numpy array files, before loading the files in the web app. This resulted in the insights display now being instantaneous.

For the 3rd problem, the writer encountered "Internal URL error" when going from the insights page to the page displaying classification results in the web app. The reason was identified as incorrect variable references from the result page ("dashboard.html) to the main page ("index.html"). This was fixed, and now the web app should run correctly.
