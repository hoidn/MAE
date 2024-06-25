import torch
from tqdm import tqdm

def train_epoch(model_dict, optimizer, dataloader):
    model_dict['model'].train()
    total_loss = 0
    
    for _, diff_img in tqdm(dataloader, desc="Training"):
        diff_img = diff_img.to(model_dict['device'])
        
        # Forward pass
        predicted_img, mask = model_dict['model'](diff_img)
        
        # Calculate losses
        batch_loss = 0
        for loss_fn, weight in zip(model_dict['loss_fns'], model_dict['loss_weights']):
            model_dict['predicted_img'] = predicted_img
            model_dict['diff_img'] = diff_img
            model_dict['mask'] = mask
            loss = loss_fn(model_dict)
            batch_loss += weight * loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
    
    return total_loss / len(dataloader)

def validate(model_dict, dataloader):
    model_dict['model'].eval()
    total_loss = 0
    
    with torch.no_grad():
        for _, diff_img in tqdm(dataloader, desc="Validation"):
            diff_img = diff_img.to(model_dict['device'])
            
            # Forward pass
            predicted_img, mask = model_dict['model'](diff_img)
            
            # Calculate losses
            batch_loss = 0
            for loss_fn, weight in zip(model_dict['loss_fns'], model_dict['loss_weights']):
                model_dict['predicted_img'] = predicted_img
                model_dict['diff_img'] = diff_img
                model_dict['mask'] = mask
                loss = loss_fn(model_dict)
                batch_loss += weight * loss
            
            total_loss += batch_loss.item()
    
    return total_loss / len(dataloader)
