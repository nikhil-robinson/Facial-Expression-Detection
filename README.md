# Facial-Expression-Detection
A realtime facial expression detection algorithm written in python with opencv and tensorflow. Trained using Face expression recognition dataset. This algorithm can diffrentiate betewen Happy, Sad , Angry , Surprised and Neutarl. 

# Dependencies

opencv-library
tensorflow
matplotlib

# DataSet
https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset

# Instalation

git clone https://github.com/nikhil-robinson/Facial-Expression-Detection

pip install -r requirements.txt

python facialExpression.py


# Note

Make sure your webcam is not in use and can be access by the program

# Docker Build

docker build -t facial_expression

docker run facial_expression

