import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def classify_stance_batch(texts_and_targets, model, tokenizer, batch_size=32, verbose=False):
    """
    Classify stance towards target groups for multiple text-target pairs.

    Validated for Horne et al.' group mention stance NLI classifier `rwillh11/mdeberta_NLI_stance_NoContext`
    
    Parameters:
    -----------
    texts_and_targets : list of tuples
        Each tuple is (text, target_group)
    model : transformers model
        Pre-loaded NLI model
    tokenizer : transformers tokenizer
        Pre-loaded tokenizer
    batch_size : int
        Number of hypothesis pairs to process at once
    
    Returns:
    --------
    pd.DataFrame with columns: text, target_group, positive, negative, neutral, predicted_stance
    """
    # Create hypotheses for each stance
    hypotheses = {
        "positive": "The text is positive towards {target_group}.",
        "negative": "The text is negative towards {target_group}.",
        "neutral": "The text is neutral, or contains no stance, towards {target_group}."
    }
    
    stances = list(hypotheses.keys())
    
    # Build all hypothesis pairs upfront
    all_texts = []
    all_hypotheses = []
    metadata = []  # track which example each hypothesis belongs to
    
    for idx, (text, target_group) in enumerate(texts_and_targets):
        for stance, hypothesis in hypotheses.items():
            all_texts.append(text)
            all_hypotheses.append(hypothesis.format(target_group=target_group))
            metadata.append((idx, stance, text, target_group))
    
    # Process in batches
    all_probs = []
    n_batches = (len(all_texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(all_texts), batch_size), total=n_batches, disable=not verbose):
        batch_texts = all_texts[i:i+batch_size]
        batch_hyps = all_hypotheses[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts, 
            batch_hyps, 
            return_tensors="pt", 
            truncation=True,
            padding=True
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            entailment_probs = probs[:, 0].cpu().numpy()  # Probability of entailment
            all_probs.extend(entailment_probs)
    
    # Organize results by original example
    results = []
    for i, (text, target_group) in enumerate(texts_and_targets):
        example_results = {'text': text, 'target_group': target_group}
        
        # Get the 3 stance probabilities for this example
        for j, stance in enumerate(stances):
            idx = i * 3 + j  # Each example has 3 hypotheses
            example_results[stance] = all_probs[idx]
        
        # Get predicted stance (highest probability)
        stance_probs = [example_results[s] for s in stances]
        example_results['predicted_stance'] = stances[np.argmax(stance_probs)]
        
        results.append(example_results)
    
    return pd.DataFrame(results)