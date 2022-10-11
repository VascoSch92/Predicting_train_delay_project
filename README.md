# Predicting train delays

How many times before heading to the train station have we asked ourselves: but will my train be on time? To answer this question, we developed a model using neural networks that consider the train station and departure time of a train and predict its punctuality.

For istance there are two scripts. The first one, *retrieve_and_cleaning_data.py*, has the task of retreiving the data and to clean it, while the second one, *train_delay_model.py*, contains the model. 

The data was retrieved on the website data.SBB.ch. Here, it is possible to find the data about train delays one day later. This means that if today we go check the data, it will report the delays of yesterday. 

For the SBB (Swiss Railway Sercive) a train has delay if it has more than 180 seconds of retard. We decided for a regressive model. Therefore, we try to predict how many seconds the train will be late. If the prediction is bigger than 180, then the train will have officially delay. 
