# Disaster Response Pipeline Project

### Table of Contents

1. [Project Motivation](#motivation)
2. [Running Instructions](#instructions)
3. [App Components](#components)
4. [File Descriptions](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)


### Project Motivation: <a name="motivation"></a>
This Project contains a scripts that analyzes disaster data from Figure Eight to build a model for an API that classifies disaster messages.
In data folder is a data set containing real messages, that were sent during disaster events. With them a machine learning pipeline is trained automatically to categorize these events,  so that emergency worker can send the messages to an appropriate disaster relief agency.

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. 

### Running Instructions: <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Project Components: <a name="components"></a>
There are three components of this app:

1. ETL Pipeline - A data cleaning pipeline script, process_data.py, that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline - A machine learning pipeline script, train_classifier.py, that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App - An app for data visualizations using Plotly.

### File Structure: <a name="files"></a>
'''
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
'''

### License: <a name="licensing"></a>

MIT License

Copyright (c) 2020 Dennis Dachkovski
This project was created during the "Data Scientist" Nanodegree from Udacity. 
Credit to Figure 8 for the data and Udacity for instructions. You can find more informations about the Nanodegree [here](https://www.udacity.com/course/data-scientist-nanodegree--nd025). 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.