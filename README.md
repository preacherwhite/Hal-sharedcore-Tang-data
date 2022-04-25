# Shared core model based on Hal's implementation

# How to use
The main files to consider are under modeling/scripts. The training wrapper is train_data_driven_cnn where it will train the model on Tang's data of M1S2. The model is the bethege model defined in modeling/models/cnns. The training loop is in modeling/train_utils. I used a scheduler to guide training with a customized function. I also added a loss function based on correlation in modeling/losses, which is the current one being used. 

# model saving 
The original model saving method is done after training and the latest model will be saved in saved_models. I added another saving method in the simple train loop function, where the model with lowest loss is saved each epoch.
