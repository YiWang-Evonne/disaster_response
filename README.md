# disaster_response
### summary:
This project is part of Udacity Data Scienist Non-Degree Program.

The main goal is to build a model for an API that classifies disaster messages by analyzing disaster data from Figure Eight. This project also provide a web UI to help classifying text message to different disaster response categories.

Currently, the champion model has an average accuracy of 0.76 across different response categories. 

### structure:

 ```bash
├── app
│   ├── templates
│   ├── run.py: start the web page
├── data: store alll data and bs
├── models
│   ├── train_classifier.py: Champion models. 
│   ├── other_candidate_model.py: another model I tried but didn't out-perform the champion model.
│   ├── bert_transformer.py: bert transformer for text columns. This is too slow to run. 
├── README.md
└── .gitignore
```


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    
3. Go to http://0.0.0.0:3001/
