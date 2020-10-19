# Pytorch-Image-Classifier-LeNet-
Difference between this repo and the tensorflow one:

1. Train / Validation data split is done by enter percentage in tf but pytorch is done by seperating train and validation file, which may affect the validation accuracy
2. Tensorflow use exponential decay in learning rate but Pytorch lr is fixed
3. The Lenet Model may have some differences

Requirements
1. pytorch (https://pytorch.org/get-started/locally/ , if you do not have gpu , dont click cuda )
2. opencv  (pip install opencv-python)

How to Run:

1. Dataset Preparation:
-Download the following dataset and create a folder name "data"
-Put the "cifar10png" file inside the "data" folder
https://drive.google.com/file/d/1d666e1DLtlAKk9D_P-aVkRB9P-jSV9AT/view?usp=

Folder Arch:

        Lconfig
        Ldata
        Ldataset
        Lmodel
        Lsaved
        Ltrainer
        test.py
        train.py

2. Run train.py

(Optional) 
3. Edit configuration in "./config/config.py" to edit the number of batch size and epoch

        num_of_epoch = 30
        batch_size = 32
        
4. Result Log (lr = 0.005 , epoch = 24, bs= 32 , Model = lenet5)

                Epoch 24/24
                --------------------
                train
                batch num: 0 cost time: 0.025916099548339844
                batch num: 100 cost time: 0.45509910583496094
                batch num: 200 cost time: 0.42002367973327637
                batch num: 300 cost time: 0.4263756275177002
                batch num: 400 cost time: 0.42834019660949707
                batch num: 500 cost time: 0.42302560806274414
                batch num: 600 cost time: 0.4291098117828369
                batch num: 700 cost time: 0.43551158905029297
                batch num: 800 cost time: 0.41848182678222656
                batch num: 900 cost time: 0.4343116283416748
                batch num: 1000 cost time: 0.42071032524108887
                train Loss: 0.5777 Acc: 0.7911
                val
                batch num: 0 cost time: 0.021633148193359375
                batch num: 100 cost time: 0.4298257827758789
                batch num: 200 cost time: 0.41722559928894043
                batch num: 300 cost time: 0.43596792221069336
                batch num: 400 cost time: 0.42412757873535156
                val Loss: 1.5678 Acc: 0.5862
                epoch complete in 0m 7s
                Training complete in 2m 54s
                Best val Acc: 0.6059

                Process finished with exit code 0


Details of the code:(TBW)
1. Check model.py to see the details of the LeNet5

                class LeNet(nn.Module):
                    def __init__(self , num_class):
                        super(LeNet,self).__init__()
                        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0, bias=True)
                        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
                        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                        self.fc1 = nn.Linear(16 * 5 * 5, 120)
                        self.fc2 = nn.Linear(120, 84)
                        self.fc3 = nn.Linear(84, 10)
                        self.fc4 = nn.Linear(10,num_class)

                    def forward(self,x):
                        x = functional.relu(self.conv1(x))
                        x = self.max_pool_1(x)
                        x = functional.relu(self.conv2(x))
                        x = self.max_pool_2(x)
                        x = x.view(-1, self.num_flat_features(x))
                        x = functional.relu(self.fc1(x))
                        x = functional.relu(self.fc2(x))
                        x = functional.relu(self.fc3(x))
                        x = self.fc4(x)
                        return x

                    def num_flat_features(self, x):
                        size = x.size()[1:]
                        num_features = 1
                        for s in size:
                            num_features *= s
                        return num_features

