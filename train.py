from model import EasyModel
from generate_fake_img import generate_fake_image
import function
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

## parameter
epochs = 30
lr = 0.001
momentum = 0.9
weight_decay = 0.0001
print_freq = 10
valid_freq = 1
save_freq = 10
save_dir = r'.\output_model'
batch_size = 16
class_num = 3
#labels_map = {"0": "circle", "1": "line", "2": "rectangle"}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(os.path.join(save_dir)):
        os.makedirs(os.path.join(save_dir))
## parameter

def data_prepare(data_dir):

    ## data prepare
    train_dir = os.path.join(data_dir,'train')
    valid_dir = os.path.join(data_dir,'valid')


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

    train_tfms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    train_dataset = datasets.ImageFolder(train_dir, train_tfms)
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,num_workers=4, pin_memory=True)

    valid_tfms = transforms.Compose([ transforms.ToTensor(),normalize,])
    valid_dataset = datasets.ImageFolder(valid_dir, valid_tfms)
    valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=1, shuffle=True,num_workers=4, pin_memory=True)
    return train_loader,valid_loader

def model_prepare():
    ## model prepare
    model = EasyModel(class_num = class_num)
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr,
                 momentum = momentum,weight_decay = weight_decay)
    return model,criterion,optimizer

def main(train_loader,valid_loader,model,criterion,optimizer):
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=15,
                                               gamma=0.1)
    ## train and valid    
    best_acc = 0
    is_best = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        ## train
        for i, (images, label) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.cuda()
            label = label.to(device, dtype=torch.long)
            output = model(images)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if i%print_freq == 0:
                print("[{}-th epoch/{}-th iteration]LR = {:.5f}, avg_loss = {:.5f}"
                       .format(epoch+1,i,optimizer.param_groups[0]['lr'],total_loss/(i+1)))
        lr_scheduler.step()
        
        ## valid
        total_acc = 0
        if (epoch+1) % valid_freq == 0:
            model.eval()
            
            with torch.no_grad():
                for i, (images, label) in enumerate(valid_loader):
                    images = images.cuda()                
                    output = model(images)
                    #print(label,output)
                    preds = torch.topk(output, k=1).indices.squeeze(0).tolist()
                    if label == preds[0]:
                        total_acc = total_acc+1
            print("====VALID====[{}-th epoch] avg_acc = {:.5f}".format(epoch+1,total_acc/(i+1))) 
            if total_acc/i+1 > best_acc:
                best_acc = total_acc/i+1
                is_best = 1
                print("[{}-th epoch] is the new best model, save it".format(epoch+1))
               
        ## save model
        if is_best:
            function.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': model.name,
                    'state_dict': model.state_dict(),
                    'avg_acc': total_acc/(i+1),
                    'optimizer' : optimizer.state_dict(),
                }, is_best,filename = os.path.join(save_dir,'ckpt{:03d}.pth.tar'.format(epoch+1)))
            is_best = 0


if __name__ == '__main__':
    data_dir = r'./data'
    generate_fake_image(data_dir, image_size=128, train_num=100, valid_num=30)
    train_loader,valid_loader = data_prepare(data_dir)
    model,criterion,optimizer = model_prepare()
    main(train_loader,valid_loader,model,criterion,optimizer)

