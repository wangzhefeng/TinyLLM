# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_translation.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-13
# * Version     : 1.0.071304
# * Description : description
# * Link        : https://www.k-a.in/llm7.html
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from layers.masking import (
    create_padding_mask, 
    create_look_ahead_mask,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


class TranslationDataset(Dataset):

    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=100):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        # Convert source sentence to indices
        src_indices = [self.src_vocab.get(word, self.src_vocab['']) for word in self.src_sentences[idx].split()]
        src_indices = [self.src_vocab['']] + src_indices + [self.src_vocab['']]
        
        # Convert target sentence to indices
        tgt_indices = [self.tgt_vocab.get(word, self.tgt_vocab['']) for word in self.tgt_sentences[idx].split()]
        tgt_indices = [self.tgt_vocab['']] + tgt_indices + [self.tgt_vocab['']]
        
        # Pad sequences
        src_indices = src_indices[:self.max_len]
        tgt_indices = tgt_indices[:self.max_len]
        
        src_indices = src_indices + [self.src_vocab['']] * (self.max_len - len(src_indices))
        tgt_indices = tgt_indices + [self.tgt_vocab['']] * (self.max_len - len(tgt_indices))
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices[:-1], dtype=torch.long), # Input to decoder
            'tgt_y': torch.tensor(tgt_indices[1:], dtype=torch.long) # Expected output
        }


def create_toy_dataset():
    # Simple English to French translation pairs
    eng_sentences = [
        'hello how are you',
        'i am fine thank you',
        'what is your name',
        'my name is john',
        'where do you live',
        'i live in new york',
        'i love programming',
        'this is a test',
        'please translate this',
        'thank you very much'
    ]
    
    fr_sentences = [
        'bonjour comment vas tu',
        'je vais bien merci',
        'quel est ton nom',
        'je m appelle john',
        'où habites tu',
        'j habite à new york',
        'j aime programmer',
        'c est un test',
        's il te plaît traduis cela',
        'merci beaucoup'
    ]
    
    # Create vocabularies
    src_vocab = {'': 0, '': 1, '': 2, '': 3}
    tgt_vocab = {'': 0, '': 1, '': 2, '': 3}
    
    # Add words to vocabularies
    i = 4
    for sent in eng_sentences:
        for word in sent.split():
            if word not in src_vocab:
                src_vocab[word] = i
                i += 1
    
    i = 4
    for sent in fr_sentences:
        for word in sent.split():
            if word not in tgt_vocab:
                tgt_vocab[word] = i
                i += 1
    
    # Create reverse vocabularies for decoding
    src_idx2word = {idx: word for word, idx in src_vocab.items()}
    tgt_idx2word = {idx: word for word, idx in tgt_vocab.items()}
    
    return eng_sentences, fr_sentences, src_vocab, tgt_vocab, src_idx2word, tgt_idx2word


def train_transformer(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for batch in train_loader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        tgt_y = batch['tgt_y'].to(device)
        
        # Create masks
        src_mask = create_padding_mask(src)
        tgt_mask = create_padding_mask(tgt) & create_look_ahead_mask(tgt.size(1)).to(device)
        
        # Forward pass
        output = model(src, tgt, src_mask, tgt_mask)
        
        # Reshape output and target for loss calculation
        output = output.view(-1, output.size(-1))
        tgt_y = tgt_y.view(-1)
        
        # Calculate loss
        loss = criterion(output, tgt_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)


def evaluate_transformer(model, val_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            tgt_y = batch['tgt_y'].to(device)
            
            # Create masks
            src_mask = create_padding_mask(src)
            tgt_mask = create_padding_mask(tgt) & create_look_ahead_mask(tgt.size(1)).to(device)
            
            # Forward pass
            output = model(src, tgt, src_mask, tgt_mask)
            
            # Reshape output and target for loss calculation
            output = output.view(-1, output.size(-1))
            tgt_y = tgt_y.view(-1)
            
            # Calculate loss
            loss = criterion(output, tgt_y)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(val_loader)


def translate(model, sentence, src_vocab, tgt_idx2word, device, max_len=100):
    model.eval()
    
    # Tokenize and convert to indices
    tokens = sentence.split()
    src_indices = [src_vocab.get(word, src_vocab['']) for word in tokens]
    src_indices = [src_vocab['']] + src_indices + [src_vocab['']]
    
    # Pad source sequence
    src_indices = src_indices + [src_vocab['']] * (max_len - len(src_indices))
    src_indices = src_indices[:max_len]
    
    # Convert to tensor
    src_tensor = torch.tensor([src_indices], dtype=torch.long).to(device)
    
    # Create mask
    src_mask = create_padding_mask(src_tensor)
    
    # Get encoder output
    enc_output = model.encode(src_tensor, src_mask)
    
    # Initialize decoder input with  token
    dec_input = torch.tensor([[src_vocab['']]], dtype=torch.long).to(device)
    
    # Generate translation
    output_indices = []
    
    for _ in range(max_len):
        # Create mask for decoder input
        tgt_mask = create_look_ahead_mask(dec_input.size(1)).to(device)
        
        # Get decoder output
        dec_output = model.decode(dec_input, enc_output, src_mask, tgt_mask)
        
        # Get predicted token
        pred = model.linear(dec_output[:, -1])
        pred_idx = pred.argmax(dim=-1).item()
        
        # Add predicted token to output
        output_indices.append(pred_idx)
        
        # Check if end of sequence
        if pred_idx == src_vocab['']:
            break
        
        # Update decoder input
        dec_input = torch.cat([dec_input, torch.tensor([[pred_idx]], dtype=torch.long).to(device)], dim=1)
    
    # Convert indices to words
    output_words = [tgt_idx2word.get(idx, '') for idx in output_indices]
    
    # Remove special tokens
    output_words = [word for word in output_words if word not in ['', '', '']]
    
    return ' '.join(output_words)




# 测试代码 main 函数
def main():
    import matplotlib.pyplot as plt
    from layers.transformer_block import Transformer

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    eng_sentences, fr_sentences, src_vocab, tgt_vocab, src_idx2word, tgt_idx2word = create_toy_dataset()
    
    # Create train and validation datasets
    train_size = int(0.8 * len(eng_sentences))
    train_dataset = TranslationDataset(
        eng_sentences[:train_size], 
        fr_sentences[:train_size], 
        src_vocab, 
        tgt_vocab
    )
    val_dataset = TranslationDataset(
        eng_sentences[train_size:], 
        fr_sentences[train_size:], 
        src_vocab, 
        tgt_vocab
    )
    
    # Create data loaders
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    
    # Use smaller model for toy dataset
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=64,
        n_heads=2,
        d_ff=128,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab[''])
    
    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        train_loss = train_transformer(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_transformer(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer_model.pth')
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('transformer_loss.png')
    plt.show()
    
    # Test translation
    test_sentences = [
        'hello how are you',
        'i love programming',
        'thank you very much'
    ]
    
    model.load_state_dict(torch.load('best_transformer_model.pth'))
    
    print("\nTest Translations:")
    for sentence in test_sentences:
        translation = translate(model, sentence, src_vocab, tgt_idx2word, device)
        print(f"English: {sentence}")
        print(f"French: {translation}")
        print()

if __name__ == "__main__":
    main()
