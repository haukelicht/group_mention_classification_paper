#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import regex
import torch
from setfit import SetFitModel
from src.finetuning.setfit_extensions import SetFitModelForSpanClassification


def get_device(verbose=False):
    """Setup and return the appropriate device for inference."""
    device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device)
    if verbose:
        print('using device:', str(device))
    return device


def load_classifier(model_path, model_revision=None, use_span_embeddings=False, device=None, verbose=False):
    """Load the SetFit classifier model."""
    if verbose:
        print(f'Loading model from {model_path}')
    
    if use_span_embeddings:
        classifier = SetFitModelForSpanClassification.from_pretrained(
            model_path,
            revision=model_revision
        )
    else:
        classifier = SetFitModel.from_pretrained(
            model_path,
            revision=model_revision
        )
    
    if device is not None:
        classifier.to(device)
    
    return classifier


def read_input_data(input_file, group_mention_types=None, group_mention_type_col='label', 
                    test_mode=False, batch_size=64, verbose=False):
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
        raise ValueError('input file must be a feather, tab-separated (.tsv, .tab) or comma-separated (.csv) file')
    
    # Filter by group mention types if specified
    if group_mention_type_col and group_mention_types:
        if group_mention_type_col in df.columns:
            df.rename(columns={group_mention_type_col: 'group_type'}, inplace=True)
        if 'group_type' in df.columns:
            df = df[df['group_type'].isin(group_mention_types)]
    
    # Sample data if in test mode
    if test_mode:
        n_ = batch_size * 10
        if n_ < len(df):
            df = df.sample(n=n_, random_state=42).reset_index(drop=True)
    
    if verbose:
        print(f'Processing {len(df)} texts')
    
    return df


def prepare_inputs(df, sentence_text_col, mention_text_col, use_span_embeddings=False,
                   concat_strategy=None, concat_sep_token=': ', classifier=None):
    """Prepare input texts for classification based on the specified strategy."""
    
    if use_span_embeddings:
        # Using span embedding strategy
        if "span" not in df.columns:
            df['span'] = df.apply(
                lambda x: regex.search(regex.escape(x[mention_text_col]), x[sentence_text_col]).span(), 
                axis=1
            )
        df['input'] = classifier._normalize_inputs(texts=df[sentence_text_col], spans=df['span'])
    
    elif concat_strategy is None:
        # Default: just the mention text
        df['input'] = df[mention_text_col]
    
    else:
        # Using concat strategy
        if concat_sep_token is None and classifier is not None:
            sep_tok = classifier.model_body.tokenizer.sep_token
        else:
            sep_tok = concat_sep_token
        
        if concat_strategy == 'prefix':
            df['input'] = df[mention_text_col] + sep_tok + df[sentence_text_col]
        elif concat_strategy == 'suffix':
            df['input'] = df[sentence_text_col] + sep_tok + df[mention_text_col]
        else:
            raise ValueError(f"Unknown concat strategy: {concat_strategy}")
    
    return df


def predict_labels(classifier, texts, batch_size=64, verbose=False):
    """Run inference on texts and return predictions."""
    if verbose:
        print(f'Predicting labels for {len(texts)} inputs')
    
    with torch.no_grad():
        predictions = classifier.predict(
            texts, 
            batch_size=batch_size, 
            as_numpy=True, 
            use_labels=False, 
            show_progress_bar=verbose
        )
    
    return predictions


def add_predictions_to_dataframe(df, predictions, classifier):
    """Add predicted labels as columns to the dataframe."""
    df[classifier.labels] = predictions
    if 'input' in df.columns:
        del df["input"]
    return df


def write_output(df, output_file, verbose=False):
    """Write predictions to file."""
    if verbose:
        print(f'Writing mention-level predicted labels to {output_file}')
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='SetFit multi-label classification inference for group mention attributes'
    )
    
    # Input/Output arguments
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input TSV file containing sentences with predicted group mentions.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output TSV file to save predictions.')
    
    # Column name arguments
    parser.add_argument('--sentence_text_col', type=str, default='sentence_text',
                        help='Name of the column containing the sentence text. Default is "sentence_text".')
    parser.add_argument('--mention_text_col', type=str, default='text',
                        help='Name of the column containing the mention text. Default is "text".')
    parser.add_argument('--group_mention_types', type=str, default=None,
                        help='Comma-separated list of group mention types to classify (e.g., "social group").')
    parser.add_argument('--group_mention_type_col', type=str, default='label',
                        help='Name of the column containing the group mention type labels. Default is "label".')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pre-trained SetFit model for classification.')
    parser.add_argument('--model_revision', type=str, default=None,
                        help='Revision of the pre-trained SetFit model for classification.')
    
    # Input preparation strategy arguments
    parser.add_argument('--use_span_embeddings', action='store_true',
                        help='Whether to use custom SetFitForSpanClassification instead of mention-only or concatenation strategies.')
    parser.add_argument('--concat_strategy', type=str, choices=[None, 'prefix', 'suffix'], default=None,
                        help='If not None, concatenate the mention text as prefix or suffix to the context text using --concat_sep_token')
    parser.add_argument('--concat_sep_token', type=str, default=': ',
                        help='Separator token to use when concatenating mention text to context text. Default is ": " (if --concat_strategy is prefix or suffix).')
    
    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference. Default is 64.')
    
    # Other arguments
    parser.add_argument('--test', action='store_true',
                        help='If set, run in test mode with a smaller subset of data.')
    parser.add_argument('--verbose', action='store_true',
                        help='If set, print verbose output during processing.')
    
    return parser.parse_args()


def main(args: argparse.Namespace):
    # Validate input file
    if not os.path.exists(args.input_file):
        raise ValueError('Input file does not exist')
    
    # Parse group mention types
    group_mention_types = None
    if args.group_mention_types:
        group_mention_types = [t.strip() for t in args.group_mention_types.split(',')]
    
    
    # Read input data
    df = read_input_data(
        input_file=args.input_file,
        group_mention_types=group_mention_types,
        group_mention_type_col=args.group_mention_type_col,
        test_mode=args.test,
        batch_size=args.batch_size,
        verbose=args.verbose
    )
    
    # Setup device
    device = get_device(verbose=args.verbose)
    
    # Load classifier
    classifier = load_classifier(
        model_path=args.model_path,
        model_revision=args.model_revision,
        use_span_embeddings=args.use_span_embeddings,
        device=device,
        verbose=args.verbose
    )

    # Prepare inputs based on strategy
    df = prepare_inputs(
        df=df,
        sentence_text_col=args.sentence_text_col,
        mention_text_col=args.mention_text_col,
        use_span_embeddings=args.use_span_embeddings,
        concat_strategy=args.concat_strategy,
        concat_sep_token=args.concat_sep_token,
        classifier=classifier
    )
    
    # Run predictions
    predictions = predict_labels(
        classifier=classifier,
        texts=df['input'].to_list(),
        batch_size=args.batch_size,
        verbose=args.verbose
    )
    
    # Add predictions to dataframe
    df = add_predictions_to_dataframe(df, predictions, classifier)
    
    # Write output
    write_output(df, args.output_file, verbose=args.verbose)


if __name__ == "__main__":
    main(parse_args())
