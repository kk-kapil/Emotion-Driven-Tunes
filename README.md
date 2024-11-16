# EMOTION_DRIVEN_TUNES
# Music-Recommendation-using-real-time-Facial-Expressions
An application by which we detects the user's facial expressions in real time and accordingly, plays a suitable song based on the perceived emotion.
The application uses a pre-trained model for emotion or mood recognition which has been trained on Kaggle's 'Fer2013' dataset.
The project's working is divided into three major steps:
1. Facial expression recognition
2. Song selection for the perceived emotion
3. Selected song play

The first step is handled by the file "emotions.py" which uses haarcascade classifier for facial recognition and a pretrained Machine Learning model for detecting facial expression and identifying the corresponding emotion.
The model, however, is not completely accurate and any slight change in facial expression results in a completely different emotion as perceived by the model. To compensate for this, the camera window uses the most common facial expression in the time frame of 5 seconds and uses that result for next steps.

The second step uses the "select_music.py" file.
The file uses a "data_moods.csv" dataset from kaggle which helps selecting a suitable song for the given emotion. The dataset contains songs from classical, country, rap and rock music.

The final step is handled by "play_music.py" file which plays the selected song on youtube web.
