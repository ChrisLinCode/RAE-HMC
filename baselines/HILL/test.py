import argparse
import os

from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Subset

from train import BertDataset, resolve_max_token
from eval import evaluate
from model.contrast import ContrastModel, StructureContrast, GraphContrast
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('-b', '--batch', type=int, default=32, help='Batch size.')
parser.add_argument('-n', '--name', type=str, required=True, help='Checkpoint directory name.')
parser.add_argument('-e', '--extra', default='macro', choices=['macro', 'micro'],
                    help='Checkpoint suffix to evaluate.')
parser.add_argument('--plm', type=str, default=None, help='Override pretrained model name or path.')
parser.add_argument('--max_token', type=int, default=None,
                    help='Maximum token length. Uses dataset metadata when omitted.')
args = parser.parse_args()


if __name__ == '__main__':
    cli_args = args
    checkpoint = torch.load(
        os.path.join('ckpt', cli_args.name, 'best_{}.pt'.format(cli_args.extra)),
        map_location='cpu',
        weights_only=False,
    )

    run_args = checkpoint['args'] if checkpoint['args'] is not None else cli_args
    data_dir = getattr(run_args, 'data_dir', 'data')
    dataset_name = getattr(run_args, 'dataset')
    cfg_dir = getattr(run_args, 'cfg_dir', 'config')
    model_name = getattr(run_args, 'model_name', 'hill')
    plm_name = cli_args.plm or getattr(run_args, 'plm', 'bert-base-uncased')
    data_path = os.path.join(data_dir, dataset_name)

    config = utils.Configure(config_json_file=os.path.join(cfg_dir, model_name + '.json'))
    config.update(run_args.__dict__)
    config.device_setting.device = cli_args.device
    config.plm = plm_name

    tokenizer = AutoTokenizer.from_pretrained(plm_name)
    max_token = resolve_max_token(data_path, cli_args.max_token)

    label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)

    dataset = BertDataset(max_token=max_token, device=cli_args.device,
                          pad_idx=tokenizer.pad_token_id, data_path=data_path)
    split = torch.load(os.path.join(data_path, 'split.pt'))
    test = Subset(dataset, split['test'])
    test = DataLoader(test, batch_size=cli_args.batch, shuffle=False, collate_fn=dataset.collate_fn)

    models = {
        'hill': StructureContrast,
        'hgclr': ContrastModel,
        'gclr': GraphContrast,
    }
    model = models[config.model_name].from_pretrained(plm_name, num_labels=num_class, local_config=config)
    model.load_state_dict(checkpoint['param'])
    model.to(cli_args.device)

    truth = []
    pred = []

    model.eval()
    with torch.no_grad():
        for data, label, idx in test:
            padding_mask = data != tokenizer.pad_token_id
            output = model(data, padding_mask, return_dict=True)
            for label_row in label:
                truth.append([i for i in range(label_row.size(0)) if label_row[i].item() == 1])
            for logits in output['logits']:
                pred.append(torch.sigmoid(logits).tolist())

    scores = evaluate(pred, truth, label_dict)
    macro_f1 = scores['macro_f1']
    micro_f1 = scores['micro_f1']
    print('Test performance with best_val_{} ↓\nmicro-f1: {:.4f}\nmacro-f1: {:.4f}'.format(
        cli_args.extra, micro_f1, macro_f1
    ))
