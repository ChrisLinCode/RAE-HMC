from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse
import json
import os
from pathlib import Path
from train import BertDataset
from eval import evaluate
from model.contrast import ContrastModel

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch', type=int, default=32, help='Batch size.')
parser.add_argument('--name', type=str, required=True, help='Name of checkpoint. Commonly as DATASET-NAME.')
parser.add_argument('--extra', default='_macro', choices=['_macro', '_micro'], help='An extra string in the name of checkpoint.')
parser.add_argument('--plm', type=str, default=None, help='Override pretrained BERT model name.')
parser.add_argument('--max-token', type=int, default=None, help='Maximum token length. Uses dataset metadata if omitted.')
args = parser.parse_args()


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
    plm_override = args.plm
    checkpoint = torch.load(
        os.path.join('checkpoints', args.name, 'checkpoint_best{}.pt'.format(args.extra)),
        map_location='cpu',
        weights_only=False,
    )
    batch_size = args.batch
    device = args.device
    extra = args.extra
    args = checkpoint['args'] if checkpoint['args'] is not None else args
    if plm_override is not None:
        args.plm = plm_override
    data_path = os.path.join('data', args.data)

    if not hasattr(args, 'graph'):
        args.graph = False
    if not hasattr(args, 'plm'):
        args.plm = 'bert-base-uncased'
    print(args)
    plm_name = args.plm if args.plm is not None else 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(plm_name)
    max_token = resolve_max_token(data_path, args.max_token)

    label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)

    dataset = BertDataset(max_token=max_token, device=device, pad_idx=tokenizer.pad_token_id, data_path=data_path)
    model = ContrastModel.from_pretrained(plm_name, num_labels=num_class,
                                          contrast_loss=args.contrast, graph=args.graph,
                                          layer=args.layer, data_path=data_path, multi_label=args.multi,
                                          lamb=args.lamb, threshold=args.thre, plm_name=plm_name)
    split = torch.load(os.path.join(data_path, 'split.pt'))
    test = Subset(dataset, split['test'])
    test = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    model.load_state_dict(checkpoint['param'])

    model.to(device)

    truth = []
    pred = []
    index = []
    slot_truth = []
    slot_pred = []

    model.eval()
    pbar = tqdm(test)
    with torch.no_grad():
        for data, label, idx in pbar:
            padding_mask = data != tokenizer.pad_token_id
            output = model(data, padding_mask, return_dict=True, )
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
