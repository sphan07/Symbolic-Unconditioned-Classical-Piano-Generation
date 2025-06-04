# cse153_task1
This project was created by Charile Ngo and Samantha Phan on May/30 - June/3 for CSE 153: Machine Learning for Music<br> <br>

Our task was symbolic uncondition music generation. We use techniques such as REMI tokenization, enhancing our LSTM model through additional layers, sampling technqiues such as top-p and top k and music functionality functions in order to promote music structure. <br>

## Our Model for Symbolic Uncondition Music Generation
Dataset<br>
Why Maestro? We used Maestro due to realizing that classical music was better for musical training. It took trial and error, as we wanted to create edm music. This did not turn out well because of our limited computation resources in addition to pubic datasets avaliable.<br>

Our dataset was split into two data subsets, train and test in order to be used for our model.<br>

Task1.ipynb<br>

LTSM Model <br>
Musicality Functions <br>
Sampling with Music Structure <br>

## Analysis
Basline.ipynb: <br>
- We used a basic Second-Order Markov and a basic LSTM to train our model in order to obtain a baseline we can compare too. <br>
- Audio Output:
  Markov:<br>
  LSTM: <br>

  As we can hear it sounds pretty awful and it is a good start to understand what not to strive for. This was used in our analysis against our model to prove that we have a "better" model.<br>

  Task1analysis.ipynb<br>

  
