# Uzapy

Uzapy is Python deep learning framework.

## Features

### Optimezers

- gd (Gradient Descent)
- sgd (Stochastic Gradient Descent)

### Weights initializers

- zeros
- ones,
- standard_distribution
- uniform_distribution
- xavier_normal or glorat_normal
- xavier_uniform or glorat_uniform
- he_normal
- he_uniform
- lecun_normal
- lecun_uniform

he initializers are good with ReLU and it variants, LeCun with SELU and Glorat with linear, tanh, softmax, sigmoid.

### Activations functions

- relu (Rectified Linear Unit)
  - pros
    - does not have a maximun value
    - very fast to compute
  - cons:
    - dying relu

- leaky_relu
  - pros
    - solve dying relu

- elu (Exponential Linear Units)
  - pros
    - outperformed all ReLU variants
    - helps speed up convergence
  - cons:
    - at test time, elu is slower that other ReLU variants

- selu (Scaled Exponitial Linear Unit)
  - pros
    - solve vanishing and exploding gradients problem
  - cons:
    - works only for a neural network composed exclusively of a stack of dense layers
    - LeCun normal initialization is required for each layer
    - input features must be standardized with mean 0 and standard deviation

- sigmoid (Logistic)
  - pros
    - the output is centered at 0.5 with range from 0 to 1
  - cons:
    - vanishing gradient
    - computationally expensive
    - the output is not zero centered
  
- tanh (Hyperbolic Tangent)
  - pros
    - the output is centered at 0 with range from -1 to 1
    - helps speed up convergence
  - cons:
    - vanishing gradient
    - computationally expensive

- softmax

### Losses

- mse (Mean Squared Error)
- mae (Mean Absolute Error)
- Binary crossentropy / log loss
- Categorical crossentropy

### Metrics

- Binary accuracy
- Categorical accuracy

## Notes

### Data normalization

- data = (data - mean) / std

NB: you must use train mean and std to normalize test set and also when you want do prediction.

## TODO

- [ ] Neural Network
- [ ] Perception
- [ ] Linear regression
- [ ] Logestic regression
- [ ] Decision tree
- [ ] SVM (Support Vector Machine)
- [ ] Naive bayes algorithm
- [ ] KNN (K-Nearst Neighbors)
- [ ] Random forest
- [ ] Dimensionality reduction algorithms
- [ ] AdaBoosting
- [ ] Gradient boosting
- [ ] K-Means
- [ ] Genetic algorithms
- [ ] DNN (Dense Neural Network)
- [ ] CNN (Convolutional Neural Network)
- [ ] RNN (Recurrent Neural Network)
- [ ] LSTM (Long short-term memory)
- [ ] Transformers

### Projects

- [ ] chatbot from scratch
- [ ] face detection
- [ ] flou remouver in images
- [ ] SpeechRecognition from scratch
