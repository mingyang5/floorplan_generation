import torch
from torch_geometric.data import DataLoader

from evaluation.metrics import mIOU
from model.models_unet import GraphFloorplanUNet

from utils import load_pickle
from data.data import msdDataset
from tqdm import tqdm

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_node_features = 4
classes = 11

test_dataset = msdDataset('./dataset_processed/test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

input_nc = 3
output_nc = 11

model = GraphFloorplanUNet(num_node_features, input_nc, output_nc, features=[64, 128, 256, 512])
model.load_state_dict(torch.load('results/11020029/model_checkpoint_epoch_99.pt'))

model = model.to(device)

# Evaluate the model
model.eval()
total_iou = 0
total_samples = 0

with torch.no_grad():
    # Wrap the test_loader with tqdm for a progress bar
    test_progress_bar = tqdm(test_loader, desc='Testing')
    for images, graph_data, gt_images in test_progress_bar:
        images = images.to(device)
        graph_data = graph_data.to(device)
        gt_images = gt_images.to(device)
        
        # Forward pass
        outputs = model(images, graph_data)
        
        # Adjust class values
        gt_images[gt_images == 13] = 10
        
        # Compute IoU
        pred_tensor = torch.argmax(outputs, dim=1)
        iou = mIOU(pred_tensor, gt_images, classes)
        
        # Accumulate total IoU and samples
        total_iou += iou
        total_samples += 1
        
        # Update the progress bar with the latest IoU
        test_progress_bar.set_postfix({'Current mIoU': iou})

# Calculate and print mean IoU
mean_iou = total_iou / total_samples
print(f'Mean IoU: {mean_iou}')