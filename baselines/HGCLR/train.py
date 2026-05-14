import json
from pathlib import Path

from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from model.optim import ScheduledOptim, Adam
from tqdm import tqdm
import argparse
import os
from eval import evaluate
from model.contrast import ContrastModel

import utils


class BertDataset(Dataset):
    def __init__(self, max_token=512, device='cpu', pad_idx=0, data_path=None):
        self.device = device
        super(BertDataset, self).__init__()
        data_root = Path(data_path)
        tok_pt = data_root / 'tok.pt'
        y_pt = data_root / 'Y.pt'
        if tok_pt.exists() and y_pt.exists():
            self.data = torch.load(tok_pt)
            self.labels = torch.load(y_pt)
            self._pt_backed = True
        else:
            from fairseq.data import data_utils
            self.data = data_utils.load_indexed_dataset(
                data_path + '/tok', None, 'mmap'
            )
            self.labels = data_utils.load_indexed_dataset(
                data_path + '/Y', None, 'mmap'
            )
            self._pt_backed = False
        self.max_token = max_token
        self.pad_idx = pad_idx

    def __getitem__(self, item):
        if self._pt_backed:
            data = torch.tensor(self.data[item][:self.max_token - 2], dtype=torch.long, device=self.device)
            labels = torch.tensor(self.labels[item], dtype=torch.long, device=self.device)
        else:
            data = self.data[item][:self.max_token - 2].to(self.device)
            labels = self.labels[item].to(self.device)
        return {'data': data, 'label': labels, 'idx': item, }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        label = torch.stack([b['label'] for b in batch], dim=0)
        data = torch.full([len(batch), self.max_token], self.pad_idx, device=label.device, dtype=batch[0]['data'].dtype)
        idx = [b['idx'] for b in batch]
        for i, b in enumerate(batch):
            data[i][:len(b['data'])] = b['data']
        return data, label, idx


class Saver:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate.')
parser.add_argument('--data', type=str, default='WebOfScience', help='Dataset directory name under ./data.')
parser.add_argument('--batch', type=int, default=12, help='Batch size.')
parser.add_argument('--early-stop', type=int, default=6, help='Epoch before early stop.')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--name', type=str, required=True, help='A name for different runs.')
parser.add_argument('--plm', type=str, default='bert-base-uncased', help='Pretrained BERT model name.')
parser.add_argument('--max-token', type=int, default=None, help='Maximum token length. Uses dataset metadata if omitted.')
parser.add_argument('--update', type=int, default=1, help='Gradient accumulate steps')
parser.add_argument('--warmup', default=2000, type=int, help='Warmup steps.')
parser.add_argument('--contrast', default=1, type=int, help='Whether use contrastive model.')
parser.add_argument('--graph', default=1, type=int, help='Whether use graph encoder.')
parser.add_argument('--layer', default=1, type=int, help='Layer of Graphormer.')
parser.add_argument('--multi', default=True, action='store_false', help='Whether the task is multi-label classification.')
parser.add_argument('--lamb', default=1, type=float, help='lambda')
parser.add_argument('--thre', default=0.02, type=float, help='Threshold for keeping tokens. Denote as gamma in the paper.')
parser.add_argument('--tau', default=1, type=float, help='Temperature for contrastive model.')
parser.add_argument('--seed', default=5, type=int, help='Random seed.')
parser.add_argument('--wandb', default=False, action='store_true', help='Use wandb for logging.')


def get_root(path_dict, n):
    ret = []
    while path_dict[n] != n:
        ret.append(n)
        n = path_dict[n]
    ret.append(n)
    return ret


def resolve_max_token(data_path, requested, default=512):
    if requested is not None:
        return int(requested)
    meta_path = Path(data_path) / 'dataset_meta.json'
    if meta_path.exists():
        with meta_path.open('r', encoding='utf-8') as handle:
            meta = json.load(handle)
        if meta.get('max_token_recommended') is not None:
            return int(meta['max_token_recommended'])
    return int(default)


if __name__ == '__main__':
    args = parser.parse_args()
    device = args.device
    print(args)
    if args.wandb:
        import wandb
        wandb.init(config=args, project='htc')
    utils.seed_torch(args.seed)
    args.name = args.data + '-' + args.name
    tokenizer = AutoTokenizer.from_pretrained(args.plm)
    data_path = os.path.join('data', args.data)
    max_token = resolve_max_token(data_path, args.max_token)
    label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)

    dataset = BertDataset(max_token=max_token, device=device, pad_idx=tokenizer.pad_token_id, data_path=data_path)
    model = ContrastModel.from_pretrained(args.plm, num_labels=num_class,
                                          contrast_loss=args.contrast, graph=args.graph,
                                          layer=args.layer, data_path=data_path, multi_label=args.multi,
                                          lamb=args.lamb, threshold=args.thre, tau=args.tau,
                                          plm_name=args.plm)
    if args.wandb:
        wandb.watch(model)
    split = torch.load(os.path.join(data_path, 'split.pt'))
    train = Subset(dataset, split['train'])
    dev = Subset(dataset, split['val'])
    if args.warmup > 0:
        optimizer = ScheduledOptim(Adam(model.parameters(),
                                        lr=args.lr), args.lr,
                                   n_warmup_steps=args.warmup)
    else:
        optimizer = Adam(model.parameters(),
                         lr=args.lr)

    train = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=dataset.collate_fn)
    dev = DataLoader(dev, batch_size=args.batch, shuffle=False, collate_fn=dataset.collate_fn)
    model.to(device)
    save = Saver(model, optimizer, None, args)
    best_score_macro = 0
    best_score_micro = 0
    early_stop_count = 0
    checkpoint_dir = os.path.join('checkpoints', args.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = open(os.path.join(checkpoint_dir, 'log.txt'), 'w')

    for epoch in range(1000):
        if early_stop_count >= args.early_stop:
            print("Early stop!")
            break
        model.train()
        i = 0
        loss = 0

        # Train
        pbar = tqdm(train)
        for data, label, idx in pbar:
            padding_mask = data != tokenizer.pad_token_id
            output = model(data, padding_mask, labels=label, return_dict=True, )
            loss /= args.update
            output['loss'].backward()
            loss += output['loss'].item()
            i += 1
            if i % args.update == 0:
                optimizer.step()
                optimizer.zero_grad()
                if args.wandb:
                    wandb.log({'train_loss': loss})
                pbar.set_description('loss:{:.4f}'.format(loss))
                i = 0
                loss = 0
                # torch.cuda.empty_cache()
        pbar.close()

        model.eval()
        pbar = tqdm(dev)
        with torch.no_grad():
            truth = []
            pred = []
            for data, label, idx in pbar:
                padding_mask = data != tokenizer.pad_token_id
                output = model(data, padding_mask, labels=label, return_dict=True, )
                for l in label:
                    t = []
                    for i in range(l.size(0)):
                        if l[i].item() == 1:
                            t.append(i)
                    truth.append(t)
                for l in output['logits']:
                    pred.append(torch.sigmoid(l).tolist())

        pbar.close()
        scores = evaluate(pred, truth, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        print('macro', macro_f1, 'micro', micro_f1)
        print('macro', macro_f1, 'micro', micro_f1, file=log_file)
        if args.wandb:
            wandb.log({'val_macro': macro_f1, 'val_micro': micro_f1, 'best_macro': best_score_macro,
                       'best_micro': best_score_micro})
        early_stop_count += 1
        if macro_f1 > best_score_macro:
            best_score_macro = macro_f1
            save(macro_f1, best_score_macro, os.path.join('checkpoints', args.name, 'checkpoint_best_macro.pt'))
            early_stop_count = 0

        if micro_f1 > best_score_micro:
            best_score_micro = micro_f1
            save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_best_micro.pt'))
            early_stop_count = 0
        # save(macro_f1, best_score, os.path.join('checkpoints', args.name, 'checkpoint_{:d}.pt'.format(epoch)))
        # save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_last.pt'))
    log_file.close()
