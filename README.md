# Predicting train delays

How many times before heading to the train station have we asked ourselves: but will my train be on time? To answer this question, we developed a model using neural networks that consider the train station and departure time of a train and predict its punctuality.

For instance, there are two scripts. The first one, *retrieve_and_cleaning_data.py*, has the task of retreiving and cleaning the data, while the second one, *train_delay_model.py*, contains the model. 

A few words about the project:

The data is retrieved on the website data.sbb.ch. Here, it is possible to find the data about train delays one day later, i.e., today are reported yesterday's delays. 

For the Swiss Federal Railways, a train is late if it has more than 180 seconds of retard. Therefore, we try to predict how many seconds the train will be late using a regressive model. If the prediction is bigger than 180, then the train will have an official delay. 
