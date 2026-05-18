import argparse
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import from detector_codes.detectors_utils
from detector_codes.detectors_utils import (
    DEVICE,
    detector_classes,
    weight_mapping,
)

CACHE_DIR = None

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

SEED = 123
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True


def calculate_auc_metrics(id_conf, ood_conf):
    all_conf = np.concatenate([id_conf, ood_conf])
    labels = np.concatenate([np.ones(len(id_conf)), np.zeros(len(ood_conf))])
    fpr, tpr, _ = metrics.roc_curve(labels, all_conf)
    auroc = metrics.auc(fpr, tpr)
    tpr_threshold = 0.95
    valid_indices = tpr >= tpr_threshold
    fpr_at_95 = fpr[np.argmax(valid_indices)] if np.any(valid_indices) else fpr[-1]
    return auroc, fpr_at_95


def calculate_average_precision(id_predictions, ood_predictions):
    all_predictions = np.concatenate([id_predictions, ood_predictions])
    labels = np.concatenate(
        [np.ones(len(id_predictions)), np.zeros(len(ood_predictions))]
    )
    return metrics.average_precision_score(labels, all_predictions)


def calculate_accuracy(id_conf, ood_conf, use_optimal=False):
    """Calculates class-specific accuracies.
    Returns (real_accuracy, fake_accuracy)"""
    if use_optimal:
        all_conf = np.concatenate([id_conf, ood_conf])
        labels = np.concatenate([np.ones(len(id_conf)), np.zeros(len(ood_conf))])

        fpr, tpr, thresholds = metrics.roc_curve(labels, all_conf)

        # We maximize the arithmetic mean of TPR (real acc) and TNR (fake acc)
        # to find the optimal balanced threshold
        balanced_accs = (tpr + (1 - fpr)) / 2
        best_idx = np.argmax(balanced_accs)

        return tpr[best_idx], 1.0 - fpr[best_idx]
    else:
        # Use fixed 0.5 threshold
        r_acc = (id_conf >= 0.5).mean()
        f_acc = (ood_conf < 0.5).mean()
        return r_acc, f_acc


def print_table_header():
    print('\n' + '=' * 95)
    print(
        f'{"Dataset":<25} | {"Similarity":<10} | {"Accuracy":<10} | {"AUC":<10} | {"AP":<10} | {"FPR95":<10}'
    )
    print('-' * 95)


def print_legend(use_optimal_threshold=False):
    print('\nLegend:')
    print(
        '- Similarity: The average detector score indicating the predicted probability of the image being Real (ID).'
    )
    if use_optimal_threshold:
        print(
            '- Accuracy: The class-specific accuracy (Real accuracy for the Real row, Fake accuracy for Generator rows)'
        )
        print('  using an optimal threshold calculated pairwise.')
    else:
        print('- Accuracy: The class-specific accuracy using a 0.5 threshold.')
        print(
            '  (For Real: score >= 0.5 is correct; For Generated: score < 0.5 is correct)'
        )
    print('- AUC: Area Under the Receiver Operating Characteristic Curve (ROC AUC).')
    print('- AP: Average Precision, summarizing the precision-recall curve.')
    print('- FPR95: False Positive Rate when the True Positive Rate (TPR) is at 95%.')


def print_evaluation_results(similarities, datasets, use_optimal_threshold=False):
    id_confi = similarities[0]
    id_name = datasets[0]

    # Pre-calculate metrics to get average Real accuracy
    ood_results = []
    id_acc_scores = []

    for ood_confi, dataset_name in zip(similarities[1:], datasets[1:]):
        auroc, fpr_95 = calculate_auc_metrics(id_confi, ood_confi)
        aver_p = calculate_average_precision(id_confi, ood_confi)
        r_acc, f_acc = calculate_accuracy(
            id_confi, ood_confi, use_optimal=use_optimal_threshold
        )
        sim = ood_confi.mean()

        ood_results.append(
            {
                'name': dataset_name,
                'sim': sim,
                'acc': f_acc,
                'auc': auroc,
                'ap': aver_p,
                'fpr': fpr_95,
            }
        )
        id_acc_scores.append(r_acc)

    avg_id_acc = np.mean(id_acc_scores) if id_acc_scores else 0.0

    print_table_header()

    # Real Section
    id_sim = id_confi.mean()
    print(
        f'{id_name:<25} | {id_sim:<10.4f} | {avg_id_acc:<10.4f} | {"-":<10} | {"-":<10} | {"-":<10}'
    )
    print(
        f'{"Average Real":<25} | {id_sim:<10.4f} | {avg_id_acc:<10.4f} | {"-":<10} | {"-":<10} | {"-":<10}'
    )
    print('-' * 95)

    # Generated Section
    auc_scores, ap_scores, fpr_scores, sim_scores, acc_scores = [], [], [], [], []

    for res in ood_results:
        print(
            f'{res["name"]:<25} | {res["sim"]:<10.4f} | {res["acc"]:<10.4f} | {res["auc"]:<10.4f} | {res["ap"]:<10.4f} | {res["fpr"]:<10.4f}'
        )
        sim_scores.append(res['sim'])
        acc_scores.append(res['acc'])
        auc_scores.append(res['auc'])
        ap_scores.append(res['ap'])
        fpr_scores.append(res['fpr'])

    avg_sim = np.mean(sim_scores)
    avg_acc = np.mean(acc_scores)
    avg_auc = np.mean(auc_scores)
    avg_ap = np.mean(ap_scores)
    avg_fpr = np.mean(fpr_scores)

    print('-' * 95)
    print(
        f'{"Average Generated":<25} | {avg_sim:<10.4f} | {avg_acc:<10.4f} | {avg_auc:<10.4f} | {avg_ap:<10.4f} | {avg_fpr:<10.4f}'
    )
    print('=' * 95)

    # Summary Table
    total_acc = (avg_id_acc + avg_acc) / 2
    print('\nSummary:')
    print('=' * 95)
    print(
        f'{"Accuracy":<12} | {"Accuracy (Real)":<18} | {"Accuracy (Gen)":<18} | {"AUC":<10} | {"AP":<10} | {"FPR95":<10}'
    )
    print('-' * 95)
    print(
        f'{total_acc:<12.4f} | {avg_id_acc:<18.4f} | {avg_acc:<18.4f} | {avg_auc:<10.4f} | {avg_ap:<10.4f} | {avg_fpr:<10.4f}'
    )
    print('=' * 95)


class HFImageDataset(Dataset):
    def __init__(self, hf_data, transform=None):
        self.hf_data = hf_data
        self.transform = transform

    def __len__(self):
        return len(self.hf_data)

    def __getitem__(self, idx):
        item = self.hf_data[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=[
            'AIDE',
            'C2P-CLIP',
            'C2P-CLIP-Original',
            'C2P-DINOv2',
            'CLIPDetection',
            'CNNDetection',
            'DeForge-AI',
            'DFFreq',
            'Effort',
            'FreqNet',
            'GramNet',
            'LaDeDa',
            'LGrad',
            'NPR',
            'RIGID',
            'Resnet50',
            'SAFE',
        ],
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='AIGC-Detection-Benchmark',
        choices=['AIGC-Detection-Benchmark', 'MS-COCOAI', '140k-Real-and-Fake-Faces'],
        help='HuggingFace dataset to evaluate on',
    )
    parser.add_argument(
        '--limit', type=int, default=1000, help='Limit samples per subset for speed'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16, help='Batch size for evaluation'
    )
    parser.add_argument(
        '--show_legend',
        type=lambda x: str(x).lower() == 'true',
        default=False,
        help='Whether to show the legend (default: False)',
    )
    args = parser.parse_args()

    dataset_configs = {
        'AIGC-Detection-Benchmark': {
            'path': 'TheKernel01/AIGC-Detection-Benchmark',
            'mapping': {
                1: 'ADM',
                2: 'BigGAN',
                3: 'CycleGAN',
                4: 'DALLE2',
                5: 'GauGAN',
                6: 'GLIDE',
                7: 'Midjourney',
                8: 'ProGAN',
                9: 'SD14',
                10: 'SD15',
                11: 'SDXL',
                12: 'StarGAN',
                13: 'StyleGAN',
                14: 'StyleGAN2',
                15: 'VQDM',
                16: 'WhichFaceIsReal',
                17: 'Wukong',
            },
        },
        'MS-COCOAI': {
            'path': 'TheKernel01/MS-COCOAI',
            'mapping': {1: 'SD21', 2: 'SDXL', 3: 'SD3', 4: 'DALLE3', 5: 'Midjourney 6'},
        },
        '140k-Real-and-Fake-Faces': {
            'path': 'TheKernel01/140k-Real-and-Fake-Faces',
            'mapping': {1: 'StyleGAN'},
        },
    }

    print(f'Initializing {args.model} detector...')
    detector = detector_classes[args.model](weight_mapping[args.model])

    print(f'Loading dataset {args.dataset}...')
    config = dataset_configs[args.dataset]
    test_data = load_dataset(
        config['path'],
        split='test',
        token=HF_TOKEN,
        cache_dir=CACHE_DIR,
    )
    all_generators = np.array(test_data['generator'])
    generator_mapping = config['mapping']

    # Prepare subsets
    real_indices = np.nonzero(all_generators == 0)[0]
    real_dataset = HFImageDataset(
        test_data.select(real_indices), transform=detector.transform
    )
    evaluation_datasets = [('Real (ID)', real_dataset)]

    for gen_id, gen_name in generator_mapping.items():
        fake_indices = np.nonzero(all_generators == gen_id)[0]
        fake_dataset = HFImageDataset(
            test_data.select(fake_indices), transform=detector.transform
        )
        evaluation_datasets.append((f'{gen_name} (OOD)', fake_dataset))

    # Run detection
    sim_datasets = []
    test_datasets = [name for name, _ in evaluation_datasets]

    for dataset_name, dataset_obj in evaluation_datasets:
        loader = DataLoader(
            dataset_obj,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
        scores = []
        total = 0

        # Calculate expected number of batches based on samples limit
        total_batches = (
            min(len(dataset_obj), args.limit) + args.batch_size - 1
        ) // args.batch_size
        pbar = tqdm(
            loader, total=total_batches, desc=f'Evaluating {dataset_name}', leave=False
        )

        for i, (imgs, _) in enumerate(pbar):
            imgs = imgs.to(DEVICE)
            # Detector returns p(fake), so we take 1 - p(fake) to get p(real)
            p_fake = detector.detect(imgs)
            score = 1.0 - p_fake
            scores.append(score.cpu())
            total += len(imgs)
            if total >= args.limit:
                break

        scores = torch.cat(scores)[: args.limit]
        print(
            f'{dataset_name:<25}, Count: {len(scores)}, Similarity: {scores.mean():.4f}'
        )
        sim_datasets.append(scores.numpy())

    print('\n' + '=' * 95)
    print(f'Results for {args.model} on {args.dataset}:')
    print('=' * 95)
    print_evaluation_results(
        sim_datasets,
        test_datasets,
        use_optimal_threshold=detector.use_optimal_threshold,
    )
    if args.show_legend:
        print_legend(use_optimal_threshold=detector.use_optimal_threshold)


if __name__ == '__main__':
    main()
