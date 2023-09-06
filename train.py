import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import pandas as pd
from skimage import io
import os
import matplotlib.pyplot as plt
from types import SimpleNamespace

# Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, mode, transform=None):
        self.mode = mode
        if self.mode == 'test':
            self.img_name = os.listdir('./'+self.mode)
        else:
            self.sport = pd.read_csv(self.mode+'.csv')
            self.img_name = list(self.sport["names"])
            self.label = list(self.sport["label"])
        self.transform = transform

    def __len__(self):
        return len(self.img_name)
    
    def __getitem__(self, index):
        image_path = self.mode + '/' + self.img_name[index]
        self.img = io.imread(image_path)
        if self.mode != 'test':
            self.target = self.label[index]
        
        if self.transform:
            self.img = self.transform(self.img)
        
        if self.mode == 'test':
            return self.img_name[index], self.img
        else:
            return self.img, self.target

# Activation functions
act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}

# Network
class InceptionBlock(nn.Module):
    def __init__(self, c_in, c_red : dict, c_out : dict, act_fn):
        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
            nn.BatchNorm2d(c_out["1x1"]),
            act_fn()
        )

        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm2d(c_red["3x3"]),
            act_fn(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            act_fn()
        )

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm2d(c_red["5x5"]),
            act_fn(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            act_fn()
        )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            act_fn()
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out

class Network(nn.Module):

    def __init__(self, num_classes=10, act_fn_name="relu", **kwargs):
        super().__init__()
        self.hparams = SimpleNamespace(num_classes=num_classes,
                                       act_fn_name=act_fn_name,
                                       act_fn=act_fn_by_name[act_fn_name])
        self._create_network()
        self._init_params()

    def _create_network(self):
        self.input_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.hparams.act_fn()
        )
        # Stacking inception blocks
        self.inception_blocks = nn.Sequential(
            InceptionBlock(64, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8}, act_fn=self.hparams.act_fn),
            InceptionBlock(64, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12}, act_fn=self.hparams.act_fn),
            nn.MaxPool2d(3, stride=2, padding=1),  # 32x32 => 16x16
            InceptionBlock(96, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12}, act_fn=self.hparams.act_fn),
            InceptionBlock(96, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16}, act_fn=self.hparams.act_fn),
            InceptionBlock(96, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16}, act_fn=self.hparams.act_fn),
            InceptionBlock(96, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24}, act_fn=self.hparams.act_fn),
            nn.MaxPool2d(3, stride=2, padding=1),  # 16x16 => 8x8
            InceptionBlock(128, c_red={"3x3": 48, "5x5": 16}, c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16}, act_fn=self.hparams.act_fn),
            InceptionBlock(128, c_red={"3x3": 48, "5x5": 16}, c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16}, act_fn=self.hparams.act_fn)
        )
        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.hparams.num_classes)
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.inception_blocks(x)
        x = self.output_net(x)
        return x

num_classes = 10
num_epochs = 40
batch_size = 16
learning_rate = 0.003
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Network().to(device)

number_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters : {}'.format(number_of_params))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 0.005, momentum = 0.9)

transform = transforms.Compose([
    transforms.ToTensor()
])

train_loader = DataLoader(MyDataset('train', transform), batch_size, shuffle = True)
valid_loader = DataLoader(MyDataset('val', transform), batch_size, shuffle = True)
test_loader = DataLoader(MyDataset('test', transform), 1, shuffle = False)

y_loss = {}
y_loss['train'] = []
y_loss['valid'] = []
y_acc = {}
y_acc['train'] = []
y_acc['valid'] = []
x_epoch = []
fig = plt.figure()
loss_fig = fig.add_subplot(121, title="Loss")
acc_fig = fig.add_subplot(122, title="Accuracy")

def draw_curve(epoch):
    x_epoch.append(epoch)
    loss_fig.plot(x_epoch, y_loss['train'], 'b-', label='train')
    loss_fig.plot(x_epoch, y_loss['valid'], 'r-', label='valid')
    acc_fig.plot(x_epoch, y_acc['train'], 'b-', label='train')
    acc_fig.plot(x_epoch, y_acc['valid'], 'r-', label='valid')
    if epoch == 0:
        loss_fig.legend()
        acc_fig.legend()
    fig.savefig(os.path.join('./Graphs', 'graphs.jpg'))

# Train
# PATH = 'HW1_311551119.pt'
top_correct = 0
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_total = 0
    train_correct = 0
    for (image, label) in train_loader:
        image = image.to(device)
        label = label.to(device)
        now_batch, c, h, w = image.shape
        output = model(image)
        loss = criterion(output, label)
        _, predicted = torch.max(output.data, 1)
        train_total += label.size(0)
        train_correct += (predicted == label).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * now_batch
    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total
    
    # print('Epoch {}/{} , Loss: {:.4f} , Accuracy: {:.4f}%'.format(epoch+1, num_epochs, loss.item(), 100 * train_correct / train_total))
    
    #-----Validation-----
    with torch.no_grad():
        model.eval()
        valid_correct = 0
        valid_total = 0
        valid_loss = 0
        for (image, label) in valid_loader:
            image = image.to(device)
            label = label.to(device)
            now_batch, c, h, w = image.shape
            output = model(image)
            loss = criterion(output, label)
            _, predicted = torch.max(output.data, 1)
            
            valid_total += label.size(0)
            valid_correct += (predicted == label).sum().item()
            del image,label,output

            valid_loss += loss.item() * now_batch
        epoch_valid_loss = valid_loss / valid_total
        epoch_valid_acc = valid_correct / valid_total
        print('Epoch {}/{} , Validation Loss: {:.4f} , Accuracy: {:.4f}%'.format(epoch+1, num_epochs, loss.item(), 100 * valid_correct / valid_total)) 
    if valid_correct > top_correct:
        torch.save(model.state_dict(), 'HW1_311551119.pt')
        top_correct = valid_correct

    # Draw loss / accuracy curve graphs
    y_loss['train'].append(epoch_train_loss)
    y_loss['valid'].append(epoch_valid_loss)
    y_acc['train'].append(epoch_train_acc)
    y_acc['valid'].append(epoch_valid_acc)
    draw_curve(epoch)
    
