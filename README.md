# Restaurant Recommender System (Chatbot)

This project focuses on developing a restaurant recommendation systems that interacts with users utilising dialogue (i.e. simple chatbot). 

## Files 
- `state_transition.py`: The backbone for this system is a state-transition model that manages the dialog flow by transitioning between predefined states based on the predicted user’s dialog acts. Much of the project is based on this architecture.
- `machine_learning.py`:  The system classifies sentences given by the user to predict what the user wants using multiple ((non-)machine learning) models. The model selection is detailed in `baseline_model_evaluation.py`
- `main.py`: Runs the chatbot, and manages extra configurability options such as text-to-speech (tts), speech delay, and spell-check accuracy along with `globals.py`.



