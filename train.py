import torch
import torchvision
import argparse
import os
from tqdm import tqdm

from dataset import CocoDataset

# Define the command line arguments
parser = argparse.ArgumentParser(description='Train a model on the COCO-2017 dataset')
parser.add_argument('--data-dir', default='coco-2017', help='path to the COCO-2017 dataset')
parser.add_argument('--batch-size', default=32, type=int, help='batch size for training')
parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for the optimizer')
parser.add_argument('--device', default='cuda:0', help='device to use for training (e.g. cpu, cuda:0)')

# Parse the command line arguments
args = parser.parse_args()

# Define the device to use for training
device = torch.device(args.device)

# Define the transform for the images
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the COCO-2017 dataset
train_dataset = CocoDataset(root_dir=args.data_dir,
                            set_name='train',
                            transform=transform)

val_dataset = CocoDataset(root_dir=args.data_dir,
                          set_name='val',
                          transform=transform)

# Create the data loaders for the datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Define the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, num_classes=91)

# Move the model to the device
model.to(device)

# Define the optimizer and the learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

# Define the training loop
for epoch in tqdm(range(args.epochs)):
    # Set the model to train mode
    model.train()

    # Initialize the running loss and accuracy
    running_loss = 0.0
    running_accuracy = 0.0

    for i, (images, targets) in enumerate(train_loader):
        # Move the images and targets to the device
        images = images[0].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets[0]]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, targets)
        loss = sum(loss for loss in outputs.values())

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update the running loss and accuracy
        running_loss += loss.item()
        running_accuracy += 1.0

    # Update the learning rate scheduler
    lr_scheduler.step()

    # Print the epoch number, training loss, and training accuracy
    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, args.epochs))