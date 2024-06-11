import time

import torch
import torch.nn as nn

from data_load import (get_batch_indices, load_vocab, load_train_data,
                                           maxlen)
from model import Transformer

# Config
batch_size = 2
lr = 0.0001
d_model = 512
d_ff = 2048
n_layers = 6
heads = 8
dropout_rate = 0.2
n_epochs = 500 
PAD_ID = 0
print_interval = 50

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cn2idx, idx2cn = load_vocab('cn')
    en2idx, idx2en = load_vocab('en')
    assert 12==len(idx2en)
    assert 11==len(idx2cn)


    Y, X = load_train_data()
    assert (4,10)==X.shape
    assert (4,10)==Y.shape

    

    model = Transformer(len(en2idx), len(cn2idx), PAD_ID, d_model, d_ff,
                        n_layers, heads, dropout_rate, maxlen)
    model.to(device)
    

    optimizer = torch.optim.Adam(model.parameters(), lr)

    citerion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    tic = time.time()
    cnter = 0
    
    for epoch in range(n_epochs):
        for index, _ in get_batch_indices(len(X), batch_size):
            x_batch = torch.LongTensor(X[index]).to(device)
            y_batch = torch.LongTensor(Y[index]).to(device)
            y_input = y_batch[:, :-1]
            y_label = y_batch[:, 1:]

            if epoch==0:
                print(y_input.shape,y_label.shape)
                print(y_input,y_label)

            y_hat = model(x_batch, y_input)
            # print(y_label.size())
            # print(y_hat.size())

            y_label_mask = y_label != PAD_ID
            preds = torch.argmax(y_hat, -1)
            correct = preds == y_label
            acc = torch.sum(y_label_mask * correct) / torch.sum(y_label_mask)

            n, seq_len = y_label.shape
            y_hat = torch.reshape(y_hat, (n * seq_len,-1)) 
            y_label = torch.reshape(y_label, (n * seq_len, ))
            loss = citerion(y_hat, y_label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if cnter % print_interval == 0:
                toc = time.time()
                interval = toc - tic
                minutes = int(interval // 60)
                seconds = int(interval % 60)
                print(f'{cnter:08d} {minutes:02d}:{seconds:02d}'
                      f' loss: {loss.item()} acc: {acc.item()}')
            cnter += 1

    model_path = 'runs/model.pth'
    torch.save(model.state_dict(), model_path)

    print(f'Model saved to {model_path}')
    


if __name__ == '__main__':
    main()
