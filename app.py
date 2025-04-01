#!/usr/bin/env python
# coding: utf-8

# In[30]:


import sentencepiece as spm
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


# In[3]:


import pandas as pd

# Load dataset from the Kaggle input directory, specifying the separator as '\t' for a TSV file
data = pd.read_csv('tatoeba-dev.ara-eng.tsv', sep='\t', encoding='utf-8', header=None, names=['Arabic', 'English'])

# Inspect the first few rows
print(data.head())


# In[4]:


print(data.columns)


# In[5]:


# Drop rows with missing values
data_cleaned = data.dropna()

# Inspect the cleaned dataset
print(data_cleaned.head())
print("Dataset Shape:", data_cleaned.shape)


# In[6]:


data = data.drop_duplicates()


# In[7]:


import re

def remove_diacritics(text):
    arabic_diacritics = re.compile(r'[\u064B-\u0652]')  # Match diacritics
    text = re.sub(arabic_diacritics, '', text)  # Remove diacritics
    text = text.replace("ى", "ي").replace("ة", "ه")  # Normalize letters
    return text.strip()

data['Arabic'] = data['Arabic'].apply(remove_diacritics)


# In[8]:


sample_text = "كُتِبَ في الكِتابِ شيءٌ مُهِمٌّ."
clean_text = remove_diacritics(sample_text)
print("Before:", sample_text)
print("After :", clean_text)


# In[9]:


data['English'] = data['English'].str.lower()


# In[10]:


data.to_csv("cleaned_dataset.tsv", sep='\t', index=False)


# In[11]:


def write_sentences_to_file(sentences, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence.strip() + '\n')

# Save Arabic and English sentences to files
write_sentences_to_file(data_cleaned['Arabic'], 'arabic_sentences.txt')
write_sentences_to_file(data_cleaned['English'], 'english_sentences.txt')


# In[12]:


import sentencepiece as spm

# Train Arabic SentencePiece model
spm.SentencePieceTrainer.train(
    input='arabic_sentences.txt',
    model_prefix='spm_arabic',
    vocab_size=15475,
    model_type='unigram',  # You can use 'bpe' instead if preferred
    character_coverage=1.0,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3  # Consistent special token IDs
)

# Train English SentencePiece model
spm.SentencePieceTrainer.train(
    input='english_sentences.txt',
    model_prefix='spm_english',
    vocab_size=6897,
    model_type='unigram',
    character_coverage=1.0,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3
)

# Load the SentencePiece models
arabic_sp = spm.SentencePieceProcessor(model_file='spm_arabic.model')
english_sp = spm.SentencePieceProcessor(model_file='spm_english.model')

# Test tokenization
arabic_example = "عمرك رايح المكسيك؟"
english_example = "Have you ever been to Mexico?"
print("Arabic Tokens:", arabic_sp.encode(arabic_example, out_type=str))
print("Arabic IDs:", arabic_sp.encode(arabic_example, out_type=int))
print("English Tokens:", english_sp.encode(english_example, out_type=str))
print("English IDs:", english_sp.encode(english_example, out_type=int))


# In[13]:


def preprocess_sequence(sp_processor, sentence, max_len, bos_id, eos_id, pad_id):
    """Preprocess a sentence with SentencePiece."""
    tokens = sp_processor.encode(sentence, out_type=int)
    tokens = [bos_id] + tokens + [eos_id]  # Add <BOS> and <EOS>
    return pad_sequence(tokens, max_len, pad_id)

def pad_sequence(tokens, max_len, pad_id):
    """Pad or truncate sequence to max_len."""
    return tokens[:max_len] + [pad_id] * max(0, max_len - len(tokens)) if len(tokens) < max_len else tokens[:max_len]


# In[14]:


max_len = 100
arabic_pad_id = 0  # From SentencePiece training
english_sos_id = 2  # BOS
english_eos_id = 3  # EOS
english_pad_id = 0  # PAD

# Tokenize and preprocess sequences
arabic_sequences = [pad_sequence(arabic_sp.encode(sentence, out_type=int), max_len, arabic_pad_id) 
                   for sentence in data_cleaned['Arabic']]
english_sequences = [preprocess_sequence(english_sp, sentence, max_len, english_sos_id, english_eos_id, english_pad_id) 
                    for sentence in data_cleaned['English']]

# Verify a sample
print("Sample Arabic Sequence:", arabic_sequences[0])
print("Sample English Sequence:", english_sequences[0])
print("Decoded English Sample:", english_sp.decode(english_sequences[0]))


# In[15]:


from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# Arabic BPE tokenizer setup
arabic_tokenizer = Tokenizer(models.BPE())
arabic_pre_tokenizer = pre_tokenizers.Whitespace()
arabic_tokenizer.pre_tokenizer = arabic_pre_tokenizer
arabic_trainer = trainers.BpeTrainer(vocab_size=20000, special_tokens=["<PAD>", "<UNK>", "<SOS>", "<EOS>"])

# Train the Arabic tokenizer
arabic_tokenizer.train(files=["arabic_sentences.txt"], trainer=arabic_trainer)
arabic_tokenizer.save("arabic_bpe_tokenizer.json")

# English BPE tokenizer setup
english_tokenizer = Tokenizer(models.BPE())
english_pre_tokenizer = pre_tokenizers.Whitespace()
english_tokenizer.pre_tokenizer = english_pre_tokenizer
english_trainer = trainers.BpeTrainer(vocab_size=10000, special_tokens=["<PAD>", "<UNK>", "<SOS>", "<EOS>"])

# Train the English tokenizer
english_tokenizer.train(files=["english_sentences.txt"], trainer=english_trainer)
english_tokenizer.save("english_bpe_tokenizer.json")


# In[16]:


# Replace Cell 17
import torch
arabic_tensors = torch.tensor(arabic_sequences, dtype=torch.long)
english_tensors = torch.tensor(english_sequences, dtype=torch.long)
print("Arabic Tensor Shape:", arabic_tensors.shape)
print("English Tensor Shape:", english_tensors.shape)


# In[17]:


from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    def __init__(self, src_tensors, tgt_tensors):
        self.src_tensors = src_tensors
        self.tgt_tensors = tgt_tensors
 
    def __len__(self):
        return len(self.src_tensors)
 
    def __getitem__(self, idx):
        return self.src_tensors[idx], self.tgt_tensors[idx]

# Create dataset
train_dataset = TranslationDataset(arabic_tensors, english_tensors)

# Create DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# In[18]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


# # Transformer

# In[19]:


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output


# In[20]:


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# In[21]:


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# In[22]:


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# In[23]:


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


# In[24]:


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(src.device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(tgt.device)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=tgt.device), diagonal=1)).bool()
        # print(f"src_mask device: {src_mask.device}, tgt_mask device: {tgt_mask.device}, nopeak_mask device: {nopeak_mask.device}")
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


# # Training 

# In[25]:


# Hyperparameters
src_vocab_size = 15475  # Matches spm_arabic vocab
tgt_vocab_size = 6897  # Matches spm_english vocab
d_model = 512          # Embedding dimension
num_heads = 8          # Number of attention heads
num_layers = 6         # Number of encoder/decoder layers
d_ff = 2048            # Feedforward dimension
max_seq_length = 100    # Matches your current setup
dropout = 0.3          # Dropout rate

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    max_seq_length=max_seq_length,
    dropout=dropout
).to(device)

model = nn.DataParallel(model)
model = model.to(device)


# In[26]:


# Updated Cell 27
import torch.optim as optim
import torch.nn as nn

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=english_pad_id)  # Use english_pad_id (0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad()
        tgt_input = tgt[:, :-1].to(device)
        tgt_output = tgt[:, 1:].to(device)
        output = model(src, tgt_input)
        loss = criterion(output.reshape(-1, tgt_vocab_size), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        predictions = output.argmax(dim=-1)
        mask = (tgt_output != english_pad_id)
        correct = (predictions == tgt_output) & mask
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    return avg_loss, accuracy



# In[ ]:


model_path = "transformer_full_updated.pth"
model = torch.load(model_path, map_location=device, weights_only=False)  # Load full model
model.eval()
model = model.to(device)


# # Inference

# In[28]:


def translate_sentence(model, arabic_sentence, arabic_sp, english_sp, max_len, device):
    """
    Translate an Arabic sentence to English using the trained Transformer model.
    
    Args:
        model: Trained Transformer model
        arabic_sentence: String containing the Arabic input sentence
        arabic_sp: SentencePiece processor for Arabic
        english_sp: SentencePiece processor for English
        max_len: Maximum sequence length (22 in your case)
        device: torch.device (cuda or cpu)
    Returns:
        translated_sentence: String containing the English translation
    """
    # Preprocess the input Arabic sentence
    arabic_sentence = remove_diacritics(arabic_sentence)
    
    # Tokenize and encode the Arabic sentence
    src = pad_sequence(arabic_sp.encode(arabic_sentence, out_type=int), max_len, 0)
    src = torch.tensor([src], dtype=torch.long).to(device)
    
    # Initialize target sequence with <BOS> token
    tgt = torch.tensor([[2]], dtype=torch.long).to(device)  # BOS = 2
    
    # Generate translation token by token
    model.eval()
    with torch.no_grad():
        for _ in range(max_len - 1):
            output = model(src, tgt)
            next_token = output[:, -1, :].argmax(dim=-1).item()
            tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
            if next_token == 3:  # EOS = 3
                break
    
    # Convert token IDs back to text
    translated_ids = tgt[0].cpu().tolist()
    if translated_ids[0] == 2:  # Remove BOS
        translated_ids = translated_ids[1:]
    if translated_ids[-1] == 3:  # Remove EOS
        translated_ids = translated_ids[:-1]
    
    translated_sentence = english_sp.decode(translated_ids)
    return translated_sentence


# In[ ]:


import gradio as gr
import torch

def translate_with_gradio(arabic_input):
    try:
        model.eval()
        with torch.no_grad():
            translated = translate_sentence(model, arabic_input, arabic_sp, english_sp, max_len=100, device=device)
        return translated
    except Exception as e:
        return f"Error: {str(e)}"

interface = gr.Interface(
    fn=translate_with_gradio,
    inputs=gr.Textbox(label="Arabic Input", placeholder="Enter an Arabic sentence..."),
    outputs=gr.Textbox(label="English Translation"),
    title="Arabic to English Translator",
    description="Enter an Arabic sentence to get its English translation using a Transformer model trained on Tatoeba data.",
    examples=[
        ["أبي ماهر في قيادة السيارة."],
        ["عمرك رايح المكسيك؟"],
        ["اليوم هو يوم مشمس وجميل."]
    ],
    theme="default"
)

interface.launch(share=True, inline=False, inbrowser=True)


# In[ ]:




