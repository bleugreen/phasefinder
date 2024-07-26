import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from model.model import PhasefinderModel
from dataset import BeatDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter 
from val import test_model_f_measure
from torch.optim.lr_scheduler import LambdaLR
import argparse
import json

parser = argparse.ArgumentParser(description='Train PhasefinderModel')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--phase_width', type=int, default=5, help='Phase width')
parser.add_argument('--model_root', type=str, default='kl9-pw5', help='Model root name')
parser.add_argument('--num_channels', type=int, default=36, help='Number of channels')
parser.add_argument('--num_classes', type=int, default=360, help='Number of classes')
parser.add_argument('--num_tcn_layers', type=int, default=16, help='Number of TCN layers')
parser.add_argument('--dilation', type=int, default=8, help='Dilation')
parser.add_argument('--start_epoch', type=int, default=0, help='Start Epoch')
parser.add_argument('--use_attention', type=bool, default=True, help='Use attention mechanism')
parser.add_argument('--load_weights', type=str, default=None, help='Path to model weights file')
parser.add_argument('--data_path', type=str, default='stft_db_b_phase.hdf5', help='Path to dataset')
parser.add_argument('--max_epochs', type=int, default=20, help='Max epochs to train')
args = parser.parse_args()

LR = args.lr
PHASE_WIDTH = args.phase_width
START_EPOCH = args.start_epoch
model_root = args.model_root

def warmup_lambda(epoch):
    ep = max(epoch, START_EPOCH)
    if ep < 5:
        return (ep + 1) / 5
    return 1.0

data_path = args.data_path

model = PhasefinderModel(
    num_bands=81, 
    num_channels=args.num_channels, 
    num_classes=args.num_classes, 
    num_tcn_layers=args.num_tcn_layers, 
    dilation=args.dilation, 
    use_attention=args.use_attention
)
if(args.load_weights):
    model.load_state_dict(torch.load(args.load_weights))

model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=LR)
warmup_scheduler = LambdaLR(optimizer, warmup_lambda)
criterion = nn.KLDivLoss(reduction="batchmean")
writer = SummaryWriter(f'runs/{model_root}')
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

def train_one_epoch(epoch):
    train_dataset = BeatDataset(data_path, 'train', mode='beat', items=['stft', 'phase'], phase_width=PHASE_WIDTH)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    model.train()
    running_loss = 0.0
    for i, (stft, beat_phase) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        
        stft = stft.cuda()
        beat_phase = beat_phase.cuda()
        
        downbeat_phase_pred = model(stft)
        loss = criterion(downbeat_phase_pred.unsqueeze(0), beat_phase)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        if (i + 1) % 1000 == 0:
            writer.add_scalar('Loss/Train', running_loss / 1000, epoch * len(train_loader) + i)
            running_loss = 0.0
    print(f"Epoch: {epoch}, Train Loss: {running_loss}")
    return running_loss / len(train_loader)

def validate(epoch):
    val_dataset = BeatDataset(data_path, 'val', mode='beat', items=['stft', 'phase'], phase_width=PHASE_WIDTH)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (stft, beat_phase) in enumerate(tqdm(val_loader)):
            stft = stft.cuda()
            beat_phase = beat_phase.cuda()
            beat_phase_pred = model(stft)
            loss = criterion(beat_phase_pred.unsqueeze(0), beat_phase)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Epoch: {epoch}, Val Loss: {val_loss}")
    writer.add_scalar('Loss/Validate', val_loss, epoch)
    return val_loss

def test(epoch):
    overall_f_measure, overall_cmlt, overall_amlt = test_model_f_measure(model, data_path)
    writer.add_scalar('F-Measure/Val', overall_f_measure, epoch)
    writer.add_scalar('Accuracy/CMLt', overall_cmlt, epoch)
    writer.add_scalar('Accuracy/AMLt', overall_amlt, epoch)
    return overall_f_measure, overall_cmlt, overall_amlt

best_val_loss = float('inf')
best_f_measure = 0
best_model_path = ''

epochs_no_improve = 0
max_epochs = args.max_epochs

# Initialize results dictionary
results = {
    "epochs": [],
    "train_loss": [],
    "val_loss": [],
    "f_measure": [],
    "cmlt": [],
    "amlt": []
}

if __name__ == '__main__':
    for epoch in range(START_EPOCH, max_epochs):
        train_loss = train_one_epoch(epoch)
        val_loss = validate(epoch)
        f_measure, cmlt, amlt = test(epoch)
        
        # Log results
        results["epochs"].append(epoch)
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["f_measure"].append(f_measure)
        results["cmlt"].append(cmlt)
        results["amlt"].append(amlt)
        
        # Save results to JSON file after each epoch
        with open(f'{model_root}_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        warmup_scheduler.step()
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_path = f'{model_root}_epoch_{epoch}_loss_{val_loss:.4f}_f_{f_measure:.3f}_cmlt_{cmlt:.3f}_amlt_{amlt:.3f}.pt'
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == 50:
                print("Early stopping due to no improvement in validation loss.")
                break
    writer.close()