import former
from former import util
from former.util import d, here

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from torchtext import data, datasets, vocab

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, gzip, time  # Added time for inference tracking

# Used for converting between nats and bits
LOG2E = math.log2(math.e)
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)
NUM_CLS = 2

def go(arg):
    """
    Creates and trains a basic transformer for the IMDB sentiment classification task.
    """
    tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging

    # load the IMDB data
    if arg.final:
        train, test = datasets.IMDB.splits(TEXT, LABEL)

        TEXT.build_vocab(train, max_size=arg.vocab_size - 2)
        LABEL.build_vocab(train)

        train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=arg.batch_size, device=util.d())
    else:
        tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
        train, test = tdata.split(split_ratio=0.8)

        TEXT.build_vocab(train, max_size=arg.vocab_size - 2) # - 2 to make space for <unk> and <pad>
        LABEL.build_vocab(train)

        train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=arg.batch_size, device=util.d())

    print(f'- nr. of training examples {len(train_iter)}')
    print(f'- nr. of {"test" if arg.final else "validation"} examples {len(test_iter)}')

    if arg.max_length < 0:
        mx = max([input.text[0].size(1) for input in train_iter])
        mx = mx * 2
        print(f'- maximum sequence length: {mx}')
    else:
        mx = arg.max_length

    # create the model
    model = former.CTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=mx, num_tokens=arg.vocab_size, num_classes=NUM_CLS, max_pool=arg.max_pool)
    if torch.cuda.is_available():
        model.cuda()
        
    # Add parameter counting
    param_count = util.non_zero_count(model)
    print(f"Model has {param_count} non-zero parameters")
    tbw.add_scalar('model/parameters', param_count, 0)
    
    # Reset GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))

    # training loop
    seen = 0
    for e in range(arg.num_epochs):

        print(f'\n epoch {e}')
        model.train(True)

        for batch in tqdm.tqdm(train_iter):

            opt.zero_grad()

            input = batch.text[0]
            label = batch.label - 1

            if input.size(1) > mx:
                input = input[:, :mx]
            out = model(input)
            loss = F.nll_loss(out, label)

            loss.backward()

            # clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if arg.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

            opt.step()
            sch.step()

            seen += input.size(0)
            tbw.add_scalar('classification/train-loss', float(loss.item()), seen)

        with torch.no_grad():

            model.train(False)
            tot, cor = 0.0, 0.0
            
            # Track inference metrics
            inference_times = []

            for batch in test_iter:

                input = batch.text[0]
                label = batch.label - 1

                if input.size(1) > mx:
                    input = input[:, :mx]
                    
                # Measure inference time
                start_time = time.time()
                out = model(input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                inference_times.append(time.time() - start_time)
                
                pred = out.argmax(dim=1)

                tot += float(input.size(0))
                cor += float((label == pred).sum().item())

            acc = cor / tot
            avg_inference_time = sum(inference_times) / len(inference_times)
            print(f'-- {"test" if arg.final else "validation"} accuracy: {acc:.3}')
            print(f'-- avg inference time per batch: {avg_inference_time*1000:.2f} ms')
            
            # Log memory usage
            if torch.cuda.is_available():
                max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                print(f'-- max GPU memory: {max_memory:.2f} MB')
                tbw.add_scalar('system/memory_mb', max_memory, e)
            
            tbw.add_scalar('classification/test-accuracy', acc, e)
            tbw.add_scalar('system/inference_time_ms', avg_inference_time * 1000, e)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=80, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=50_000, type=int)

    parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=512, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=10_000, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)
