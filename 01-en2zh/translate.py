import torch

from data_load import (idx_to_sentence, load_vocab,load_test_data,
                                            maxlen)
from model import Transformer


d_model = 512
d_ff = 2048
n_layers = 6
heads = 8
dropout_rate = 0.2


PAD_ID = 0


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cn2idx, idx2cn = load_vocab('cn')
    en2idx, idx2en = load_vocab('en')

    model = Transformer(len(en2idx), len(cn2idx), 0, d_model, d_ff, n_layers,
                        heads, dropout_rate, maxlen)
    model.to(device)
    model.eval()

    model_path = 'runs/model.pth'
    model.load_state_dict(torch.load(model_path))

 
    X,_ = load_test_data()
    #X,_ = load_train_data()
    
    for x in X:
        x_batch = torch.LongTensor([x]).to(device)
        print(idx_to_sentence(x, idx2en, True))
        y_input = torch.ones(1, maxlen,
                            dtype=torch.long).to(device) * PAD_ID
        y_input[0] = en2idx['<S>']
        with torch.no_grad():
            for i in range(1, y_input.shape[1]):
                y_hat = model(x_batch, y_input)
                y_input[0, i] = torch.argmax(y_hat[0, i - 1])
                if y_input[0, i]==3:
                    break
        output_sentence = idx_to_sentence(y_input[0], idx2cn, True)
        print(output_sentence)


if __name__ == '__main__':
    main()
