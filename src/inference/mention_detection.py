#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import jsonlines
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm


def get_device(verbose=False):
    """Setup and return the appropriate device for inference."""
    device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device)
    if verbose:
        print('using device:', str(device))
    return device


def load_model_and_pipeline(model_path, device, batch_size=64):
    """Load model, tokenizer and create inference pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    classifier = pipeline(
        task='ner', 
        model=model, 
        tokenizer=tokenizer,
        aggregation_strategy='simple',
        device=device,
        batch_size=batch_size
    )
    
    return classifier


def read_input_data(input_file, test_mode=False, batch_size=64):
    """Read input data from CSV or TSV file."""
    # Determine separator based on file extension
    sep = None
    if input_file.endswith('.feather'):
        df = pd.read_feather(input_file)
    elif input_file.endswith('.tsv') or input_file.endswith('.tab'):
        sep = '\t'
        df = pd.read_csv(input_file, sep=sep)
    elif input_file.endswith('.csv'):
        sep = ','
        df = pd.read_csv(input_file, sep=sep)
    else:
        raise ValueError('input file must be a feather (.feather), tab-separated (.tsv, .tab) or comma-separated (.csv) file')
    
    
    # Sample data if in test mode
    if test_mode:
        n_ = batch_size * 10
        if n_ < len(df):
            df = df.sample(n=n_, random_state=42)
    
    return df


def predict_labels(classifier, texts, verbose=False):
    """Run inference on texts and return predictions."""
    if verbose:
        print(f'Predicting labels for {len(texts)} inputs')
    
    # Use Dataset with KeyDataset for better progress tracking with pipeline
    dataset = Dataset.from_dict({"text": texts})
    
    # Run inference with automatic progress bar using KeyDataset
    predictions = []
    for out in tqdm(classifier(KeyDataset(dataset, "text")), total=len(dataset), disable=not verbose):
        predictions.append(out)
    
    return predictions


def add_spans_to_dataframe(df, predictions):
    """Add predicted spans to the dataframe."""
    df['spans'] = [
        [
            [span['start'], span['end'], span['entity_group']]
            for span in spans
        ]
        for spans in predictions 
    ]
    return df


def write_jsonl_output(df, output_file, id_col, text_col, metadata_cols, verbose=False):
    """Write predictions as JSONL file."""
    if verbose:
        print(f'Writing text and predicted labels in JSONL format to {output_file}')
    
    with jsonlines.open(output_file, 'w') as file:
        for _, d in df.iterrows():
            out = {
                'id': d[id_col],
                'text': d[text_col],
                'labels': d['spans'],
                'metadata': {c: d[c] for c in metadata_cols},
            }
            file.write(out)


def write_spanlevel_output(df, output_file, id_col, text_col, metadata_cols, verbose=False):
    """Convert to span-level format and write as TSV file."""
    
    # Get relevant columns
    df = df[[id_col, text_col] + metadata_cols + ['spans']].copy()
    
    # Get span index (within text unit)
    df.loc[:, 'span_nr'] = df.spans.apply(lambda x: list(range(len(x))))
    
    # Explode nested list of spans to span level (like tidyr::unnest_longer in R)
    df = df.explode(['spans', 'span_nr'])
    
    # Drop inputs with no predicted spans
    df = df[~df.spans.isna()]
    
    df['span_nr'] = df.span_nr + 1
    
    # Get the span label and text (a.k.a 'mention')
    df['label'] = df.apply(lambda r: r.spans[2], axis=1)
    df['span'] = df.apply(lambda r: r[text_col][r.spans[0]:r.spans[1]], axis=1)
    
    # Bring the columns in the right order
    df = df[metadata_cols + [id_col, text_col, 'span_nr', 'span', 'label']]
    
    if verbose:
        print(f'Writing span-level predictions to {output_file}')


    # determine the separator based on the output file extension
    if output_file.endswith('.feather'):
        df.to_feather(output_file)
    elif output_file.endswith('.tsv') or output_file.endswith('.tab'):
        sep = '\t'
        df.to_csv(output_file, sep=sep, index=False, encoding='utf-8')
    elif output_file.endswith('.csv'):
        sep = ','
        df.to_csv(output_file, sep=sep, index=False, encoding='utf-8')
    else:
        raise ValueError('output file must be a tab-separated (.tsv, .tab) or comma-separated (.csv) file')
    # df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Token classification inference for NER tasks')
    
    parser.add_argument('-m', '--model-path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='Input file path (TSV or CSV format)')
    parser.add_argument('--id-col', type=str, default='sentence_id',
                        help='Column name for IDs (default: sentence_id)')
    parser.add_argument('--text-col', type=str, default='text',
                        help='Column name for text (default: text)')
    parser.add_argument('--metadata-cols', nargs='+', default=[],
                        help='list of metadata columns')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for inference (default: 64)')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='Output file path for predictions')
    parser.add_argument('--return-spanlevel', action='store_true',
                        help='Output span-level predictions')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with limited samples')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    
    return parser.parse_args()


def main(args: argparse.Namespace):
    # validate input file path and format
    if not os.path.exists(args.input_file):
        raise ValueError('Input file does not exist')
    ext = args.input_file.split('.')[-1] if '.' in args.input_file else ''
    if ext not in ['tsv', 'tab', 'csv', 'feather']:
        raise ValueError('Input file must have a tabular extension (.tsv, .tab, .csv, or .feather)')
    
    # validate input and output file path and format
    ext = args.output_file.split('.')[-1] if '.' in args.output_file else ''
    if args.return_spanlevel and ext not in ['tsv', 'tab', 'csv', 'feather']:
        raise ValueError('Output file must have a tabular extension (.tsv, .tab, .csv, or .feather)')
    if not args.return_spanlevel and ext != 'jsonl':
        raise ValueError('Output file must have a .jsonl extension when not returning span-level output')

    # Parse metadata columns
    if isinstance(args.metadata_cols, str):
        args.metadata_cols = args.metadata_cols.split(',')
    metadata_cols = [c.strip() for c in args.metadata_cols if c.strip()]
    
    # Read input data
    df = read_input_data(
        input_file=args.input_file,
        test_mode=args.test,
        batch_size=args.batch_size
    )
    
    # Setup device
    device = get_device(verbose=args.verbose)
    
    # Load model and create pipeline
    classifier = load_model_and_pipeline(
        model_path=args.model_path,
        device=device,
        batch_size=args.batch_size
    )
    # Run predictions
    predictions = predict_labels(
        classifier=classifier,
        texts=df[args.text_col].to_list(),
        verbose=args.verbose
    )
    
    # Add spans to dataframe
    df = add_spans_to_dataframe(df, predictions)
    
    
    
    # Write span-level output if requested
    if args.return_spanlevel:
        write_spanlevel_output(
            df=df,
            output_file=args.output_file,
            id_col=args.id_col,
            text_col=args.text_col,
            metadata_cols=metadata_cols,
            verbose=args.verbose
        )
    else:
        # Write JSONL output
        write_jsonl_output(
            df=df,
            output_file=args.output_file,
            id_col=args.id_col,
            text_col=args.text_col,
            metadata_cols=metadata_cols,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main(parse_args())
