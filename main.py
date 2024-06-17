import torch
import torch.nn as nn
import torch.optim as optim
import nltk
import math
import copy
from Transformer import Transformer
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import io
import random
import time

# Returns lines of data
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().splitlines()

# Returns a padded list of length equal to max_seq_length
def pad_data(data, max_seq_length):
    if len(data) > max_seq_length:
        data = data[:max_seq_length]
    elif len(data) < max_seq_length:
        data += [''] * (max_seq_length - len(data))
    return data

# Trains a transformer model using CrossEntropyLoss criterio and Adam optimizer using LearningQ training datasets
def training(d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, batch_size, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    source_data = load_data(r"C:\Users\c03nr\OneDrive\Desktop\school\LTP\LearningQ\data\experiments\khan\src-train.txt")
    target_data = load_data(r"C:\Users\c03nr\OneDrive\Desktop\school\LTP\LearningQ\data\experiments\khan\tgt-train.txt")

    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(map(tokenizer, iter(source_data)), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    torch.save(vocab, r"vocab.pt")


    src_vocab_size = vocab.__len__()
    tgt_vocab_size = vocab.__len__()

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    transformer.train()

    return
    for epoch in range(epochs):
        # Training loop using batch training
        optimizer.zero_grad()

        sample_indices = random.sample(range(len(source_data)), batch_size)

        # Empty lists for training data
        src_batch = []
        tgt_batch = []

        for index in sample_indices:
            src_batch.append(vocab(pad_data(tokenizer(source_data[index]), max_seq_length)))
            tgt_batch.append(vocab(pad_data(tokenizer(target_data[index]), max_seq_length)))

        src_data = torch.tensor(src_batch, dtype=torch.long)
        tgt_data = torch.tensor(tgt_batch, dtype=torch.long)

        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    torch.save(transformer.state_dict(), r"C:\Users\c03nr\OneDrive\Desktop\school\LTP\transfomer.pt")
    return

# Tests a saved model using CrossEntropyLoss criterion and Adam optimizer using LearningQ testing datasets. Validation loss is calculated using 50 samples.
def testing(d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    source_data = load_data(r"C:\Users\c03nr\OneDrive\Desktop\school\LTP\LearningQ\data\experiments\khan\src-test.txt")
    target_data = load_data(r"C:\Users\c03nr\OneDrive\Desktop\school\LTP\LearningQ\data\experiments\khan\tgt-test.txt")

    tokenizer = get_tokenizer("basic_english")
    vocab = torch.load('vocab.pt')
    vocab.set_default_index(vocab['<unk>'])

    src_vocab_size = vocab.__len__()
    tgt_vocab_size = vocab.__len__()

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    transformer.load_state_dict(torch.load(r"C:\Users\c03nr\OneDrive\Desktop\school\LTP\transfomer.pt"))
    transformer.eval()

    sample_test = random.sample(range(len(source_data)), batch_size)
    test = []
    test_target = []

    for index in sample_test:
        test.append(vocab(pad_data(tokenizer(source_data[index]), max_seq_length)))
        test_target.append(vocab(pad_data(tokenizer(target_data[index]), max_seq_length)))

    test_data = torch.tensor(test, dtype=torch.long)
    test_target_data = torch.tensor(test_target, dtype=torch.long)
    out = transformer(test_data, test_target_data[:,:-1])
    BLEUscore = []

    for index, item in enumerate(sample_test):
        print("\nTest input:")
        print(source_data[item])
        print("\nTest output:")
        text_out = []
        for i in range(len(out[index])):
            num = torch.argmax(out[index], dim=1)
            text_out.append(vocab.lookup_token(num[i].item()))
        text_out = text_out[:len(tokenizer(target_data[item]))]
        print(' '.join(text_out))
        print("\nExpected output:")
        print(target_data[item])
        BLEUscore.append(nltk.translate.bleu_score.sentence_bleu([target_data[item]], text_out))
        print(BLEUscore[index])

    sample_test = random.sample(range(len(source_data)), batch_size)
    test = []
    test_target = []

    for index in sample_test:
        test.append(vocab(pad_data(tokenizer(source_data[index]), max_seq_length)))
        test_target.append(vocab(pad_data(tokenizer(target_data[index]), max_seq_length)))

    test_data = torch.tensor(test, dtype=torch.long)
    test_target_data = torch.tensor(test_target, dtype=torch.long)

    with torch.no_grad():
        val_output = transformer(test_data, test_target_data[:,:-1])
        val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), test_target_data[:, 1:].contiguous().view(-1))
        print(f"\nValidation Loss: {val_loss.item()}")

    print("BLEU score: " + str(sum(BLEUscore)/len(BLEUscore)))
    total_params = sum(p.numel() for p in transformer.parameters())
    training_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Total Parameters: " + str(total_params))
    print("Total Trainable Parameters: " + str(training_params))
    return

def main():
    time1 = time.perf_counter()
    d_model = 512           # Model's dimension
    num_heads = 8           # Number of attention heads
    num_layers = 6          # Number of encoder and decoder layers
    d_ff = 2048             # Dimonsionality of feed-forward input
    max_seq_length = 150    # Maximum sequence length
    dropout = 0.1
    batch_size = 10
    epochs = 100000

    training(d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, batch_size, epochs)
    testing(d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, 50)

    time2 = time.perf_counter()
    print("time elapsed: " + str(time2-time1))

if __name__ == '__main__':
    main()
