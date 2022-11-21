# Report - Model Evaluation

I started my first experiments training the model with the base model freezed and training only the last layers. I think this is a good technique considering the images dataset its not so big, and wasn’t necessary to run the model with so many parameters (more than 23 million).

During the first four experiments the learning rate was 0.0001 as suggested, and I started changing a bit the dropout rate and the data augmentation parameters. After all this experiments were the loss and accuracy didn’t show an improvement in performance, I decided to lower the dropout rate to zero and learning rate 0.0008 (exp_007). I also added an early stopping callback as to stop iterating when de val_accuracy didn’t improve after 3 epochs. The val_loss was reduced in each epoch but val_accuracy got stuck in 0.205. So I reduced a bit more the learning rate to 0.0005 and te result in loss and accuracy became a bit better.
For the next experiment I tried a learning rate of 0.0003 and added a drop out rate of 0.1 and performance became also a bit better from previous results.

For exp_11 I deleted from model the L2 regularizer I didn’t realized it was very high for my model (default L2 value), and continued training with imagenet weights. Loss and accuracy results became better with values of 2.7367 and 0.3507 respectively. But also became to overfit in accuracy values (0.7172 in train accuracy vs the value of 0.3507 of validation accuracy).

Taking this weights as input I made another experiment adding  a L2 regularizer of 0.0005 in dense layer, the results became a bit better with values of 2.7519 and 0.3771 for validation loss and accuracy. Although the metrics improved there was overfitting between train and test accuracy.

The result of accuracy in test dataset was of 0.3381.
