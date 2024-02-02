import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from processors.utils import *

import pandas as pd
import csv

from transformers import (
  WEIGHTS_NAME,
  AdamW,
  BertConfig,
  BertForSequenceClassification,
  BertTokenizer,
  XLMConfig,
  XLMForSequenceClassification,
  XLMTokenizer,
  XLMRobertaConfig,
  XLMRobertaTokenizer,
  XLMRobertaForSequenceClassification,
  get_linear_schedule_with_warmup,
)

from processors.utils import convert_examples_to_features
from processors.xnli import XnliProcessor

# explainer = LimeTextExplainer(class_names=label_list)
# exp = explainer.explain_instance(text, predictor)
# exp.save_to_file("/home/ashish/benchmark/xtreme/outputs-temp/xnli/exp_cons.html")

MODEL_CLASSES = {
  "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
  "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
  "xlmr": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}

BASE_PATH = "/home/ashish/benchmark/xtreme/"

PROCESSORS = {
  'xnli': XnliProcessor
}

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def get_maximum_head(attention_layer, start_index, end_index, indices):
    result = []
    for index in indices:
        result_ = []
        for i in range(12):  # iterating across all heads and picking the max val
            result_.append(attention_layer[i][0][start_index + index])
        result.append(np.max(result_))

    return np.mean(result)

def get_maximum_overall(attention, start_index, indices):
    scores = []
    for index in indices:
        scores_ = []
        for i in range(len(attention)):  # iterating over layers
            for j in range(len(attention[i][0])):  # iterating over heads
                scores_.append(attention[i][0][j][0][start_index + index])
        scores.append(np.max(scores_))
    return np.mean(scores)


def get_attention_scores(attention):
    scores = []

    for index in range(512):
        scores_ = []
        for i in range(len(attention)):  # iterating over layers
            for j in range(len(attention[i][0])):  # iterating over heads
                scores_.append(attention[i][0][j][0][index])
        scores.append(np.max(scores_))
    
    return scores

def search(token, j, cs_premise_tokens):
    found = False
    index = -1
    for k in range(j, len(cs_premise_tokens)):
        if cs_premise_tokens[k] == token:
            found = True
            index = k
            break
    return found, index


def get_diff_tokens(first_tokens, second_tokens):
    i = 0
    j = 0
    diff_tokens = []
    diff_indices = []
    
    while i < len(first_tokens) and j < len(second_tokens):
        if first_tokens[i] == second_tokens[j]:
            i += 1
            j += 1
        else:
            found, index = search(first_tokens[i], j, second_tokens)
            if found:
                j = index
                i += 1
            else:
                diff_tokens.append(first_tokens[i])
                diff_indices.append(i)
                i += 1
    return diff_tokens, diff_indices

def compute_diff_attentions(texts, high_cs_texts, lang_texts, tokenizer_, attentions):
    lang_diff_pre = []
    cs_diff_pre = []
    lang_diff_hypo = []
    cs_diff_hypo = []
    cs_scores = []
    lang_scores = []
    for q, text in enumerate(texts):
        premise_cs = high_cs_texts[q].split("\t")[0]
        hypothesis_cs = high_cs_texts[q].split("\t")[1]

        premise_lang = lang_texts[q].split("\t")[0]
        hypothesis_lang = lang_texts[q].split("\t")[1]
        
        cs_premise = tokenizer_.encode_plus(premise_cs, add_special_tokens=False, max_length=512)["input_ids"]
        cs_hypo = tokenizer_.encode_plus(hypothesis_cs, add_special_tokens=False, max_length=512)["input_ids"]
        lang_premise = tokenizer_.encode_plus(premise_lang, add_special_tokens=False, max_length=512)["input_ids"]
        lang_hypo = tokenizer_.encode_plus(hypothesis_lang, add_special_tokens=False, max_length=512)["input_ids"]

        lang_premise_len = len(lang_premise)
        cs_premise_len = len(cs_premise)
        lang_hypo_len = len(lang_hypo)
        cs_hypo_len = len(cs_hypo)

        cs_premise_tokens = tokenizer_.convert_ids_to_tokens(cs_premise)
        lang_premise_tokens = tokenizer_.convert_ids_to_tokens(lang_premise)
        cs_hypo_tokens = tokenizer_.convert_ids_to_tokens(cs_hypo)
        lang_hypo_tokens = tokenizer_.convert_ids_to_tokens(lang_hypo)
        lang_minus_cs_pre_tokens = []
        lang_minus_cs_pre_indices = []
        lang_minus_cs_hypo_tokens = []
        lang_minus_cs_hypo_indices = []

        lang_minus_cs_hypo_tokens, lang_minus_cs_hypo_indices = get_diff_tokens(lang_hypo_tokens, cs_hypo_tokens)
        lang_minus_cs_pre_tokens, lang_minus_cs_pre_indices = get_diff_tokens(lang_premise_tokens, cs_premise_tokens)
        cs_minus_lang_hypo_tokens, cs_minus_lang_hypo_indices = get_diff_tokens(cs_hypo_tokens, lang_hypo_tokens)
        cs_minus_lang_pre_tokens, cs_minus_lang_pre_indices = get_diff_tokens(cs_premise_tokens, lang_premise_tokens)

        print(tokenizer_.decode(tokenizer_.convert_tokens_to_ids(lang_minus_cs_hypo_tokens)))
        print(cs_minus_lang_hypo_tokens)
        print(lang_minus_cs_pre_tokens)
        print(cs_minus_lang_pre_tokens)
        lang_diff_pre.append(tokenizer_.decode(tokenizer_.convert_tokens_to_ids(lang_minus_cs_pre_tokens)))
        lang_diff_hypo.append(tokenizer_.decode(tokenizer_.convert_tokens_to_ids(lang_minus_cs_hypo_tokens)))
        cs_diff_pre.append(tokenizer_.decode(tokenizer_.convert_tokens_to_ids(cs_minus_lang_pre_tokens)))
        cs_diff_hypo.append(tokenizer_.decode(tokenizer_.convert_tokens_to_ids(cs_minus_lang_hypo_tokens)))
        #print(len(cs_premise_tokens), len(cs_premise), len(lang_premise_tokens), len(lang_premise))

        attn_lang_premise = 0.0
        attn_cs_premise = 0.0
        attn_lang_hypo = 0.0
        attn_cs_hypo = 0.0

        attn_lang_premise += get_maximum_overall(attentions[q], 1, lang_minus_cs_pre_indices)
        index = 1+lang_premise_len
        attn_cs_premise += get_maximum_overall(attentions[q], index, cs_minus_lang_pre_indices)
        index += cs_premise_len
        attn_lang_hypo += get_maximum_overall(attentions[q], index+1, lang_minus_cs_hypo_indices)
        index += 1+lang_hypo_len
        attn_cs_hypo += get_maximum_overall(attentions[q], index, cs_minus_lang_hypo_indices)

        """for i in range(12):  # averaging over all 12 layers
            attn_lang_premise += get_maximum_head(attentions[q][i][0], 1, 1+lang_premise_len, lang_minus_cs_pre_indices)
            index = 1+lang_premise_len
            attn_cs_premise += get_maximum_head(attentions[q][i][0], index, index+cs_premise_len, cs_minus_lang_pre_indices)
            index += cs_premise_len
            attn_lang_hypo += get_maximum_head(attentions[q][i][0], index+1, index+1+lang_hypo_len, lang_minus_cs_hypo_indices)
            index += 1+lang_hypo_len
            attn_cs_hypo += get_maximum_head(attentions[q][i][0], index, index+cs_hypo_len, cs_minus_lang_hypo_indices)"""

        # cs_scores.append(attn_cs_premise/12 + attn_cs_hypo/12)
        # lang_scores.append(attn_lang_premise/12 + attn_lang_hypo/12)
        # print(attn_lang_premise/12, attn_cs_premise/12, attn_lang_hypo/12, attn_cs_hypo/12)
        cs_scores.append(attn_cs_premise + attn_cs_hypo)
        lang_scores.append(attn_lang_premise + attn_lang_hypo)
        print(attn_lang_premise, attn_cs_premise, attn_lang_hypo, attn_cs_hypo)

    return lang_diff_pre, lang_diff_hypo, cs_diff_pre, cs_diff_hypo, lang_scores, cs_scores

def get_pre_hyp_scores(texts, model_, tokenizer_, model_type):
    probs = []
    attentions = []
    for q, text in enumerate(texts):
        premise = text.split("\t")[0]
        hypothesis = text.split("\t")[1]

        inputs = tokenizer_.encode_plus(premise, hypothesis, add_special_tokens=True, max_length=512)

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        
        attention_mask = [1] * len(input_ids)
        input_length = len(input_ids)
        
        # Zero-pad up to the sequence length.
        padding_length = 512 - len(input_ids)
        pad_token = 0
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([input_ids], dtype=torch.long)
        all_attention_mask = torch.tensor([attention_mask], dtype=torch.long)
        all_token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)
        all_labels = torch.tensor([1], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        ex = dataset[0]
        #device = torch.device("cuda:0" if torch.cuda.is_available() and not False else "cpu")
        device = torch.device("cuda:2")
        ex = tuple(t.to(device) for t in ex)

        model_.eval()
        with torch.no_grad():
            inputs = {"input_ids": ex[0].unsqueeze(0), "attention_mask": ex[1].unsqueeze(0),  "labels": ex[3].unsqueeze(0)}
            inputs["token_type_ids"] = (
                ex[2].unsqueeze(0) if model_type in ["bert"] else None
            )
            
            outputs = model_(**inputs)
            loss, logits, attention = outputs[:3]
            #loss, logits = outputs[:3]

        prob = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
        probs.append(prob[0])
        attention = [att.detach().cpu().numpy() for att in attention]
        #print(attention)
        attentions.append(attention)
    
    return probs, attentions, input_ids

def swapped_analysis(args, model_, tokenizer_):
    low_swapped_premises_ = args["swapped_low_premise"]
    low_swapped_hypo = args["swapped_low_hypo"]
    high_swapped_premises_ = args["swapped_high_premise"]
    high_swapped_hypo = args["swapped_high_hypo"]

    low_swapped_premises, high_swapped_premises = [], []
    idx = 0
    for i in range(len(low_swapped_hypo)):
        if i != 0 and i % 3 == 0:
            idx += 1
        low_swapped_premises.append(low_swapped_premises_[idx])
        high_swapped_premises.append(high_swapped_premises_[idx])

    swapped_low_texts = [low_swapped_premises[i] + "\t" + low_swapped_hypo[i] for i in range(len(low_swapped_premises))]
    swapped_high_texts = [high_swapped_premises[i] + "\t" + high_swapped_hypo[i] for i in range(len(high_swapped_premises))]

    low_premises_ = args["cs_low_premises"]
    low_hypo = args["cs_low_hypo"]
    high_premises_ = args["cs_high_premises"]
    high_hypo = args["cs_low_hypo"]

    low_premises, high_premises = [], []
    idx = 0
    for i in range(len(low_hypo)):
        if i != 0 and i % 3 == 0:
            idx += 1
        low_premises.append(low_premises_[idx])
        high_premises.append(high_premises_[idx])

    low_texts = [low_premises[i] + "\t" + low_hypo[i] for i in range(len(low_premises))]
    high_texts = [high_premises[i] + "\t" + high_hypo[i] for i in range(len(high_premises))] 

    #swapped_low_probs = compute_xnli_prob(swapped_low_texts, model_, tokenizer_, args["model_type"])
    swapped_high_probs = get_pre_hyp_scores(swapped_high_texts, model_, tokenizer_, args["model_type"])[0]
    #low_probs = compute_xnli_prob(low_texts, model_, tokenizer_, args["model_type"])
    high_probs = get_pre_hyp_scores(high_texts, model_, tokenizer_, args["model_type"])[0]

    dict_ = {}
    dict_["high_premises"] = high_premises
    dict_["high_hypo"] = high_hypo
    dict_["predictions"] = high_probs
    dict_["high_swapped_premises"] = high_swapped_premises
    dict_["high_swapped_hypo"] = high_swapped_hypo
    dict_["swapped_predictions"] = swapped_high_probs

    pd.DataFrame.from_dict(dict_).to_csv(args["output_dir"] + "swapped_high_csprobs.csv")

def diff_analysis(args, model_, tokenizer_):
    high_premises_ = args["cs_high_premises"]
    high_hypo = args["cs_high_hypo"]

    hi_premises_ = args["hindi_premises"]
    hi_hypo = args["hindi_hypo"]

    high_premises, hi_premises = [], []
    idx = 0
    for i in range(len(high_hypo)):
        if i != 0 and i % 3 == 0:
            idx += 1
        high_premises.append(high_premises_[idx])
        hi_premises.append(hi_premises_[idx])

    high_cs_texts = [high_premises[i] + "\t" + high_hypo[i] for i in range(len(hi_premises[0:100]))]
    hi_texts = [hi_premises[i] + "\t" + hi_hypo[i] for i in range(len(hi_premises[0:100]))]
    
    texts = [hi_premises[i] + " " + high_premises[i] + "\t" + hi_hypo[i] + " " + high_hypo[i] for i in range(len(hi_premises[0:100]))]

    probs, attentions = get_pre_hyp_scores(texts, model_, tokenizer_, args["model_type"])
    lang_diff_pre, lang_diff_hypo, cs_diff_pre, cs_diff_hypo, lang_scores, cs_scores = compute_diff_attentions(texts, high_cs_texts, hi_texts, tokenizer_, attentions)
    
    output = {"Hindi_premise": hi_premises[0:100], "Hindi_hypo": hi_hypo[0:100], "cs_high_premise": high_premises[0:100], "cs_high_hypo": high_hypo[0:100], 
    "Hindi_minus_cs_premise": lang_diff_pre, "Hindi_minus_cs_hypo": lang_diff_hypo, "cs_minus_hindi_premise": cs_diff_pre, "cs_minus_hindi_hypo": cs_diff_hypo,
    "lang_scores": lang_scores, "cs_scores": cs_scores}

    pd.DataFrame.from_dict(output).to_csv(args["output_dir"] + "finegrained_cshigh.csv")


def get_words(decoded):

    # group tokens
    tokens = []
    indices = []
    temp = []
    temp_indices = []
    premise_last = 0
    
    last_index = len(decoded) - 1
    for tok in reversed(decoded):
        if tok != "<s>": break
        last_index -= 1

    for i,tok in enumerate(decoded[1:last_index]):
        if tok == "</s>": 
            if decoded[i] != "</s>": premise_last = i
            continue
        elif not tok.startswith("▁"):
            temp.append(tok)
            temp_indices.append(i+1)
        else:
            if len(temp) > 0:
                tokens.append(temp)
                indices.append(temp_indices)
            temp = [tok]
            temp_indices = [i+1]
    
    if len(temp) > 0:
        tokens.append(temp)
        indices.append(temp_indices)

    result = []
    for i, temp in enumerate(tokens):
        word = ""
        for wo in temp:
            word += wo if "▁" not in wo else wo[1:]        
        result.append(word)
    
    return result, indices, premise_last

def get_important_words(sents, model, tokenizer, model_type):
    _, attentions, input_ids = get_pre_hyp_scores(sents, model, tokenizer, model_type)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    words, indices, premise_last = get_words(tokens)  # convert tokens to words
    
    # get premise and hypo indices
    premise_indices, hypo_indices = [], []
    flag = True
    premise_final_index = 0
    for i, index in enumerate(indices):
        if index[-1] != premise_last and flag:
            premise_indices.append(index)
        elif flag:
            premise_indices.append(index)
            flag = False
            premise_final_index = i
        else:
            hypo_indices.append(index)

    #print(premise_indices, hypo_indices, premise_final_index)

    # get attention scores over the tokens
    scores = get_attention_scores(attentions)

    # get word level scores
    attn_scores_premise = []
    attn_scores_hypo = []
    premise, hypo = [], []
    for i, index in enumerate(indices):
        score = 0
        for inx in index:
            score += scores[inx]
        
        if i <= premise_final_index:
            attn_scores_premise.append(score)
            premise.append(words[i])
        else:    
            attn_scores_hypo.append(score)
            hypo.append(words[i])

    # get final selected words for premise and hypothesis
    avg_pre = sum(attn_scores_premise)/len(attn_scores_premise)
    avg_hypo = sum(attn_scores_hypo)/len(attn_scores_hypo)
    final_words_premise = []
    final_words_hypo = []

    for i,attn in enumerate(attn_scores_premise):
        if attn >= avg_pre:
            #final_words_premise.append(premise[i])
            final_words_premise.append(i)

    for i,attn in enumerate(attn_scores_hypo):
        if attn >= avg_hypo:
            #final_words_hypo.append(hypo[i])
            final_words_hypo.append(i)

    #print(final_words_premise)
    #print(final_words_hypo)

    return final_words_premise, final_words_hypo

def get_alignment_data(lang):

    with open(BASE_PATH + 'awesome-align/xnli_data/probing/en-{}_premise.txt'.format(lang),'r') as f:
        enhi_prem = f.readlines()
    
    with open(BASE_PATH + 'awesome-align/xnli_data/probing/en-{}g_premise.txt'.format(lang),'r') as f:
        enhig_prem = f.readlines()

    en_premises = []
    hi_premises = []
    hig_premises = []
    for i,line in enumerate(enhi_prem):
        y = line.split('|||')
        yg = enhig_prem[i].split("|||")
        en_premises.append(y[0].strip().replace("\n", ""))
        hi_premises.append(y[1].strip().replace("\n", ""))
        hig_premises.append(yg[1].strip().replace("\n", ""))

    with open(BASE_PATH + 'awesome-align/xnli_data/probing/en-{}_premise-prob.txt'.format(lang),'r') as f:
        enhi_prem_probs = f.readlines()

    with open(BASE_PATH + 'awesome-align/xnli_data/probing/en-{}g_premise-prob.txt'.format(lang),'r') as f:
        enhig_prem_probs = f.readlines()

    premise_probs = []
    for line in enhi_prem_probs:
        line = line.replace("\n", "")
        tokens = line.split()
        lis = [float(token) for token in tokens]
        premise_probs.append(lis)

    premise_probsg = []
    for line in enhig_prem_probs:
        line = line.replace("\n", "")
        tokens = line.split()
        lis = [float(token) for token in tokens]
        premise_probsg.append(lis)

    with open(BASE_PATH + 'awesome-align/xnli_data/probing/en-{}_premise-alignmentsr.txt'.format(lang),'r') as f:
        enhi_prem_align = f.readlines()
    
    with open(BASE_PATH + 'awesome-align/xnli_data/probing/en-{}g_premise-alignmentsr.txt'.format(lang),'r') as f:
        enhig_prem_align = f.readlines()
    
    prem_ali = [] #list of lists of tuples
    for i,line in enumerate(enhi_prem_align):
        l = []
        for z in line.split():
            tup = z.split('-')
            #l.append((en_premises[i].split()[int(tup[0])], hi_premises[i].split()[int(tup[1])]))  # list of alignment tuples
            l.append((int(tup[0]), int(tup[1])))  # list of alignment tuples
        prem_ali.append(l)
    
    prem_alig = [] #list of lists of tuples
    for i,line in enumerate(enhig_prem_align):
        l = []
        for z in line.split():
            tup = z.split('-')
            # l.append((en_premises[i].split()[int(tup[0])], hig_premises[i].split()[int(tup[1])]))  # list of alignment tuples
            l.append((int(tup[0]), int(tup[1])))  # list of alignment tuples
        prem_alig.append(l)
    
    """ Read the dev hypothesis and alignment with english"""
    with open(BASE_PATH + 'awesome-align/xnli_data/probing/en-{}_hyp.txt'.format(lang),'r') as f:
        enhi_hyp = f.readlines()
    
    with open(BASE_PATH + 'awesome-align/xnli_data/probing/en-{}g_hyp.txt'.format(lang),'r') as f:
        enhig_hyp = f.readlines()

    en_hypothesis = []
    hi_hypothesis = []
    hig_hypothesis = []
    for i, line in enumerate(enhi_hyp):
        y = line.split('|||')
        yg = enhig_hyp[i].split('|||')
        en_hypothesis.append(y[0].strip())
        hi_hypothesis.append(y[1].strip())
        hig_hypothesis.append(yg[1].strip())

    with open(BASE_PATH + 'awesome-align/xnli_data/probing/en-{}_hyp-prob.txt'.format(lang),'r') as f:
        enhi_hyp_probs = f.readlines()

    with open(BASE_PATH + 'awesome-align/xnli_data/probing/en-{}g_hyp-prob.txt'.format(lang),'r') as f:
        enhig_hyp_probs = f.readlines()

    hypothesis_probs = []
    for line in enhi_hyp_probs:
        line = line.replace("\n", "")
        tokens = line.split()
        lis = [float(token) for token in tokens]
        hypothesis_probs.append(lis)
    
    hypothesis_probsg = []
    for line in enhig_hyp_probs:
        line = line.replace("\n", "")
        tokens = line.split()
        lis = [float(token) for token in tokens]
        hypothesis_probsg.append(lis)

    with open(BASE_PATH + 'awesome-align/xnli_data/probing/en-{}_hyp-alignmentsr.txt'.format(lang),'r') as f:
        enhi_hyp_align = f.readlines()
    
    with open(BASE_PATH + 'awesome-align/xnli_data/probing/en-{}g_hyp-alignmentsr.txt'.format(lang),'r') as f:
        enhig_hyp_align = f.readlines()
    
    hyp_ali = []
    for i, line in enumerate(enhi_hyp_align):
        l = []
        for z in line.split():
            tup = z.split('-')
            # l.append((en_hypothesis[i].split()[int(tup[0])], hi_hypothesis[i].split()[int(tup[1])]))
            l.append((int(tup[0]), int(tup[1])))
        hyp_ali.append(l)
    
    hyp_alig = []
    for i, line in enumerate(enhig_hyp_align):
        l = []
        for z in line.split():
            tup = z.split('-')
            # l.append((en_hypothesis[i].split()[int(tup[0])], hig_hypothesis[i].split()[int(tup[1])]))
            l.append((int(tup[0]), int(tup[1])))
        hyp_alig.append(l)

    print(len(en_premises), len(en_hypothesis), len(prem_ali), len(hyp_alig), len(hypothesis_probs), len(premise_probsg))

    dict_ = {"hi_premises": hi_premises, "hig_premises": hig_premises, "hi_hypothesis": hi_hypothesis, "hig_hypothesis": hig_hypothesis,
            "hyp_ali": hyp_ali, "hyp_alig": hyp_alig, "hypothesis_probs": hypothesis_probs, "hypothesis_probsg": hypothesis_probsg, 
            "prem_ali": prem_ali, "prem_alig": prem_alig, "premise_probs": premise_probs, "premise_probsg": premise_probsg, "en_premises": en_premises,
            "en_hypothesis": en_hypothesis}
    
    return dict_

def get_en(word, align):
    reqd = None
    for tup in align:
        #if tup[1].strip().replace("\n", "") == word.strip().replace("\n", ""):
        if tup[0] == word:  # word now is an index
            #return tup[0].strip().replace("\n", "")
            return tup[1]

    return reqd


def get_attentionoverlap(model, tokenizer, model_type):
    
    dict_ = get_alignment_data("hi")

    # iterate over all pairs of hi and en
    hi_premises, hi_hypos = dict_["hi_premises"], dict_["hi_hypothesis"]
    en_premises, en_hypos = dict_["en_premises"], dict_["en_hypothesis"]
    prem_ali = dict_["prem_ali"]
    hyp_ali = dict_["hyp_ali"]
    premise_probs = dict_["premise_probs"]
    hypothesis_probs = dict_["hypothesis_probs"]
    overlap = 0.0
    eng_count = 0
    for i in range(4472):
        if i % 50 == 0: print(i)
        en_premise = en_premises[i]
        en_hypo = en_hypos[i]
        hi_premise = hi_premises[i]
        hi_hypo = hi_hypos[i]

        premise_align = [ali for j, ali in enumerate(prem_ali[i]) if premise_probs[i][j] > 0.5]
        hypothesis_align = [ali for j, ali in enumerate(hyp_ali[i]) if hypothesis_probs[i][j] > 0.5]

        # these words are now indices
        en_final_words_premise, en_final_words_hypo = get_important_words([en_premise + "\t" + 
                en_hypo], model, tokenizer, model_type)
        hi_final_words_premise, hi_final_words_hypo = get_important_words([hi_premise + "\t" + 
                hi_hypo], model, tokenizer, model_type)
        
        eng_count += len(en_final_words_premise)
        cnt = 0
        for word in en_final_words_premise:
            hiword = get_en(word, premise_align)  # enword is an index now
            if hiword != None and hiword in hi_final_words_premise: cnt += 1
        
        overlap += (float(cnt)/float(len(en_final_words_premise)))

        eng_count += len(en_final_words_hypo)
        cnt = 0
        for word in en_final_words_hypo:
            hiword = get_en(word, hypothesis_align)
            if hiword != None and hiword in hi_final_words_hypo: cnt += 1
        
        overlap += (float(cnt)/float(len(en_final_words_hypo)))

    print("Overlap for mono hindi is {}".format(overlap/4472.0))

    # iterate over all pairs of hig and en
    hi_premises, hi_hypos = dict_["hig_premises"], dict_["hig_hypothesis"]
    en_premises, en_hypos = dict_["en_premises"], dict_["en_hypothesis"]
    prem_ali = dict_["prem_alig"]
    hyp_ali = dict_["hyp_alig"]
    premise_probs = dict_["premise_probsg"]
    hypothesis_probs = dict_["hypothesis_probsg"]
    overlap = 0.0
    for i in range(4472):
        if i % 50 == 0: print(i)
        en_premise = en_premises[i]
        en_hypo = en_hypos[i]
        hi_premise = hi_premises[i]
        hi_hypo = hi_hypos[i]

        premise_align = [ali for j, ali in enumerate(prem_ali[i]) if premise_probs[i][j] > 0.5]
        hypothesis_align = [ali for j, ali in enumerate(hyp_ali[i]) if hypothesis_probs[i][j] > 0.5]

        en_final_words_premise, en_final_words_hypo = get_important_words([en_premise + "\t" + 
                en_hypo], model, tokenizer, model_type)
        hi_final_words_premise, hi_final_words_hypo = get_important_words([hi_premise + "\t" + 
                hi_hypo], model, tokenizer, model_type)
        

        cnt = 0.0
        for word in en_final_words_premise:
            hiword = get_en(word, premise_align)
            if hiword != None and hiword in hi_final_words_premise: cnt += 1
        
        overlap += (float(cnt)/float(len(en_final_words_premise)))

        cnt = 0.0
        for word in en_final_words_hypo:
            hiword = get_en(word, hypothesis_align)
            if hiword != None and hiword in hi_final_words_hypo: cnt += 1
        
        overlap += (float(cnt)/float(len(en_final_words_hypo)))

    print("Overlap for gtranslated hindi is {}".format(overlap/float(4472)))

def main():

    bert_path = "bert-base-multilingual-cased-LR2e-5-epoch2-MaxLen128"
    xlmr_path = "xlm-roberta-large-LR5e-6-epoch2-MaxLen128"
    
    OUTPUT_DIR = "/home/ashish/benchmark/xtreme/outputs-temp/xnli/" + xlmr_path + "/"
    MODEL_DIR = "/run/user/1056/outputs-temp/xnli/" + xlmr_path + "/"
    #OUTPUT_DIR = "/home/ashish/benchmark/xtreme/outputs/xnli/bert_swapped/" + bert_path + "/"
    #OUTPUT_DIR = "/run/user/1056/outputs/xnli/bert_mixswapped/" + bert_path + "/"
    
    model_type = "xlmr"
    args = {"do_lower_case": False, "model_name_or_path": xlmr_path,
        "max_seq_length":512, "model_type": model_type, "task_name": "xnli", "batch_size": 8, "no_cuda": False, 
        "output_dir": OUTPUT_DIR}

    # Setup CUDA, GPU & distributed training
    #device = torch.device("cuda:0" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
    device = torch.device("cuda:2")
    args["n_gpu"] = torch.cuda.device_count()
    args["device"] = device


    # Set seed
    set_seed(42)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args["model_type"]]
    print("model class is ", model_class)
    best_checkpoint = os.path.join(MODEL_DIR, 'checkpoint-best')
    #best_checkpoint = OUTPUT_DIR
    tokenizer = tokenizer_class.from_pretrained(best_checkpoint, do_lower_case=False)
    model = model_class.from_pretrained(best_checkpoint)
    model.to(args["device"])

    if args["task_name"] not in PROCESSORS:
        raise ValueError("Task not found: %s" % (args["task_name"]))
    processor = PROCESSORS[args["task_name"]]()
    args["output_mode"] = "classification"
    label_list = processor.get_labels()

    lang2id = None
    print("label list is ", label_list)

    get_attentionoverlap(model, tokenizer, "xlmr")
    # probs, _ = get_pre_hyp_scores(["क्या यह बीस प्रतिशत ब्याज है" + "\t" + 
    #             "कोई दिलचस्पी नहीं है।"], model, tokenizer, "xlmr")
    #print(probs)

if __name__ == "__main__":
    main()