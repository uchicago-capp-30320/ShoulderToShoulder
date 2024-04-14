# Shoulder2Shoulder

A web application to foster community engagement and fight the loneliness pandemic.

## Virtual Environments

Both front and back ends use conda virtual environments.

<pre>
```
conda activate frontend
```
</pre>

<pre>
```
conda activate backend
```
</pre>

## Pre-Commit Checklist

- backend: run flake8 to check for errors

<pre>
```
cd backend
flake8
```
</pre>

- frontend: format and lint

<pre>
```
cd frontend
npm run lint
npm run format
```
</pre>

## Components
ShoulderToShoulder consists of three major components: a frontend to display information and 
interact with users, a backend to manage accounts, data, and information, and a machine learning 
component to provide users recommendations for events to attend.

### Frontend

### Backend

### Machine Learning
To generate user recommendations, we employ a deep factorization machine (DeepFM). A DeepFM
consists of three main components, as shown in the picture below. 

![DeepFM architecture](https://d2l.ai/_images/rec-deepfm.svg)

The embedding layer takes in vectors of user by event information and encodes it into a higher
dimensional space. Each unique feature value is represented by a unique integer which corresponds
to an index into an embedding matrix. For example, if each user by item data point contains 
five features and we want to represent features by ten dimensional embeddings, then each of 
the five features would have a 1 x 10 embedding vector and the user would have a 5x10 embedding, 
which is updated during training. This is equivalent to learning a linear layer but it saves 
computation by not having to execute expensive matrix multiplications. The outputs of the 
factorizaion machine are raw scores for each user.

The user-event embeddings are then passed into a facorization machine and an MLP. A factorization 
machine is like a linear regression that accounts for every second order user-event x user-event 
interaction but instead of naively conducting such a regression, the factorization computes 
an equivalents but more computationally efficient model. 

The user-event embeddings are also passed to an MLP, which utilizes dense layers with relu and
a tunable dropout parameter for regularization. The final output of the MLP are also raw scores 
for each user.

Finally, the raw outputs from the factorization machine and the MLP are summed and passed 
through a sigmoid function to calculate probabilities of a user attending an event.

Our overall training strategy is to pretrain our DeepFM once we have a sufficient amount of 
information and then periodically fine tune it by executing a small number of training epochs
on new data.

For more information on factorization machines and DeepFMs, see:

    Rendle, Steffen. "Factorization machines." In 2010 IEEE International conference on data 
        mining, pp. 995-1000. IEEE, 2010.

    Guo, Huifeng, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. "DeepFM: a 
        factorization-machine based neural network for CTR prediction." arXiv preprint 
        arXiv:1703.04247 (2017).
