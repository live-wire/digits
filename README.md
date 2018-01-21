# Digit Recognizer 
Comparison/Analysis of several ML classifiers. Run the python server file to load the web-page which takes user input from HTML Canvas and gives a prediction based on a trained model.


### SETUP INSTRUCTIONS:
Assuming python3 and pip3 are already installed. Setting up the virtual environment:
```
virtualenv -p python3 nistvenv
source nistvenv/bin/activate
pip install -r requirements.txt
```

Then simply run the server file which picks up a trained model from model.json and model.h5:
```
python server.py
```

The UI will be accessible on port 5001.


