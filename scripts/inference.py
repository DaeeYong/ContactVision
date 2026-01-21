'''
Author: DY Kim
Date: 25.09.03
.npy는 pelvis 기준, 상대좌표로 변환된 lower joints가 들어가야 함.(OpenPose 좌표)
'''

import torch
from src.model import FootContactTransformer
import numpy as np
import argparse
from colorama import Fore, Style, init

init(autoreset=True)

MODEL_PTH = './checkpoints/best_model.pth'
DEVICE    = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

def infer_full_sequence(model, data, seq_len, device):
    model.eval()
    T = data.shape[0]
    if data.ndim == 3:
        data = data.reshape(T, -1)
    preds = []
    with torch.no_grad():
        for start in range(0, T, seq_len):
            end = min(start + seq_len, T)
            chunk = data[start:end]
            if end - start < seq_len:
                pad = np.zeros((seq_len - (end - start), data.shape[1]), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)
            inp = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
            logits = model(inp)
            pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().squeeze(0)  # shape: (4,)

            preds.append(pred)
    return np.concatenate(preds, axis=0)[:T]  # (T, 4)

def load_model(model_path=MODEL_PTH, device=DEVICE):
    # Load Model
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    hyper = ckpt['hyperparams']
    input_dim = hyper.get('input_dim', 39)
    
    model = FootContactTransformer(
        input_dim=input_dim,
        embed_dim=hyper['embed_dim'],
        n_heads=hyper['n_heads'],
        ff_dim=hyper['ff_dim'],
        num_layers=hyper['num_layers'],
        dropout=hyper['dropout']
    ).to(device)
    
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    return model
    
def main():
    # ArgParse
    parser = argparse.ArgumentParser(description='Runs inference on an input .npy file and saves the result.')
    parser.add_argument('--input_path', type=str, required=True, help='.npy path')
    parser.add_argument('--output_path', type=str, required=True, help='save .npy path')
    parser.add_argument('--model_path', type=str, default=MODEL_PTH)
    
    args = parser.parse_args()
    intput_npy = args.input_path
    output_npy = args.output_path
    
    print(Fore.GREEN + "Load Model")
    # Load Model
    model = load_model(MODEL_PTH, DEVICE)
    
    print(Fore.GREEN + "inference...") 
    # load npy
    data = np.load(intput_npy)
    pred = infer_full_sequence(model, data, seq_len=128, device=DEVICE)
    np.save(output_npy, pred.astype(np.float32))
    print(Fore.GREEN + f'saved done: {output_npy}')

if __name__ == "__main__":
    main()