# Jigsaw-toxic
Jigsaw Unintended Bias in Toxicity Classification Kaggle competition

Can you help detect toxic comments ― and minimize unintended model bias? That's your challenge in this competition.

The Conversation AI team, a research initiative founded by Jigsaw and Google (both part of Alphabet), builds technology to protect voices in conversation. A main area of focus is machine learning models that can identify toxicity in online conversations, where toxicity is defined as anything rude, disrespectful or otherwise likely to make someone leave a discussion.

Last year, in the Toxic Comment Classification Challenge, you built multi-headed models to recognize toxicity and several subtypes of toxicity. This year's competition is a related challenge: building toxicity models that operate fairly across a diverse range of conversations.

Here’s the background: When the Conversation AI team first built toxicity models, they found that the models incorrectly learned to associate the names of frequently attacked identities with toxicity. Models predicted a high likelihood of toxicity for comments containing those identities (e.g. "gay"), even when those comments were not actually toxic (such as "I am a gay woman"). This happens because training data was pulled from available sources where unfortunately, certain identities are overwhelmingly referred to in offensive ways. Training a model from data with these imbalances risks simply mirroring those biases back to users.

In this competition, you're challenged to build a model that recognizes toxicity and minimizes this type of unintended bias with respect to mentions of identities. You'll be using a dataset labeled for identity mentions and optimizing a metric designed to measure unintended bias. Develop strategies to reduce unintended bias in machine learning models, and you'll help the Conversation AI team, and the entire industry, build models that work well for a wide range of conversations.


# My Solution
There are three my final models: LSTM with attention (Keras), LSTM with pooling (PyTorch) and Bert (Hugging Face PyTorch implementation) with custom head for multiclass classification.

In keras model i used two layers of bi-LSTM with top of attention-with-context layer and simple linear layers with dropout wich outperformed best publik keras kernels with bi-LSTM and pooling. Optimizer - standart Adam + LambdaLR (as it named in pytorch)

PyTorch model architechture was the same as in publik kernel but i used AdamW optimizer (Adam with decay instead of l2 regularization) instead of usual Adam from PyTorch and LambdaLR scheduler. I tryed a bunch of optimizers and schedulers here, one cycle policy too, unfortunatly onecycle policy didnt improved score this time as any of other optimizers and schedulers. So i used AdamW with LambdaLR

Keras and PyTorch models were multioutput models (two heads in other word) - one output for target data, the other for aux otputs (types of toxicity).

The result prediction (keras and pytorch) - weighted average of every epoch output.

Final Solution was blend of Keras and PyTorch LSTM's that brings 0.936**

Bert model has not been used for final submission as i started to train Bert too late - 1 day before deadline =(
But anyway i evaluated Bert with train-test split (1,5m / 0.3m) and it gave me best result (0.938**)
This model included custom loss from public kernel and custom head made by myself on top of classification layer for multiclass prediction.
Due to limmit of time i did not do any preprocessing, used 1,5m rows out of 1,8 for train, only one epoch, without any lr scheduler (as i saw from later competitors solutions one of the used onecycle policy).

PyTorch LSTM and BERT was trained with custom loss that weights identity's information.

Also i spend a lot of time trying linear models but best that i have achived 0.906** with NB-SVM classifier. 
I added notebook with NB-SVM to save it in case.

Competition metric was custom ROC-AUC metric that combines several submetrics to balance overall performance with various aspects of unintended bias.
