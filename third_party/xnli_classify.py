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
from sys import getsizeof

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
from processors.pawsx import PawsxProcessor
# from lime.lime_text import LimeTextExplainer

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter

from xnli_utils import *

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
  "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
  "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
  "xlmr": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}

PROCESSORS = {
  'xnli': XnliProcessor,
  'pawsx': PawsxProcessor,
}

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def get_examples(lang, hypothesis, premises, labels):
    examples = []

    print(len(hypothesis), len(premises), len(labels))
    for (i, line) in enumerate(premises):
        # iterate over three premises
        guid = "%s-%s" % (lang, i)
        text_a = premises[i]
        text_b = hypothesis[i]
        label = labels[i]
        #if not isinstance(text_a, str): continue
        #print(type(text_a), type(text_b), type(label))
        assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lang))
            
    return examples

def get_examples_probe(lang, hypothesis, premises, hi_hypothesis, hi_premises, labels):
    examples = []

    cnt = 0
    for (i, line) in enumerate(premises):
        if cnt == len(hypothesis): break
        # iterate over three premises
        guid = "%s-%s" % (lang, i)
        text_a = hi_premises[i] + " " + premises[i]
        text_b = hi_hypothesis[i] + " " + hypothesis[i]
        label = labels[i]
        assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lang))
        cnt += 1
            
            
    return examples

def load_and_cache_examples(args, task, tokenizer, split, language='en', lang2id=None, evaluate=False, probe=False):

    processor = PROCESSORS[task]()
    output_mode = "classification"
    
    # Load data features from cache or dataset file
    lc = '_lc' if args["do_lower_case"] else ''

    label_list = processor.get_labels()
    hypo, premises, labels = [],[],[]
    labels = args["labels"]
    hypo, premises = args["hyp"], args["pre"]
    
    # hi_hypo, hi_premises = args["hi_hyp"], args["hi_pre"]

    # print("lengths of hindi and english ---", len(hi_hypo), len(hi_premises), len(hypo), len(premises))

    if not probe:
        examples = get_examples(language, hypo, premises, labels)
    else:
        examples = get_examples_probe(language, hypo, premises, hi_hypo, hi_premises, labels)

    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=args["max_seq_length"],
        output_mode=output_mode,
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
        lang2id=lang2id,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    else:
        raise ValueError("No other `output_mode` for {}.".format(args["task_name"]))

    if args["model_type"] == 'xlm':
        all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_langs)
    else:  
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def compute_metrics(preds, labels):
  scores = {
    "acc": (preds == labels).mean(), 
    "num": len(
      preds), 
    "correct": (preds == labels).sum()
  }
  return scores

def get_data(path, args):
    hyp = []
    pre = []
    labels = []

    is_tsv = path.endswith(".tsv")

    if is_tsv:
        with open(path) as file:
            tsv_file = csv.reader(file, delimiter="\t")
            for line in tsv_file:
                #print(line)
                pre.append(line[0].replace("\n", ""))
                hyp.append(line[1].replace("\n", ""))
                labels.append(line[2].replace("\n", ""))
    else:
        f = open(path)
        lines = f.readlines()
        
        for i,line in enumerate(lines):
            pre.append(line.split("\t")[0].replace("\n", ""))
            hyp.append(line.split("\t")[1].replace("\n", ""))
            labels.append(line.split("\t")[2].replace("\n", ""))
        f.close()
    
    args["pre"] = pre
    args["hyp"] = hyp
    args["labels"] = labels

def evaluate(args, model, tokenizer, split, language='en', lang2id=None, output_file=None, label_list=None, output_only_prediction=True, probe=False):
    """Evalute the model."""
    results = {}
    
    # store hypothesis and premises of the input language
    get_data(language, args)
    hypo_, premises_lang, labels_lang = args["hyp"], args["pre"], args["labels"]
    
    # # store hypo and premise of hindi
    # get_data(args["data_dir"] + "dev-hi.tsv" if split == "dev" else args["data_dir"] + "test-hi.tsv", args)
    # hi_hypo, hi_premises, labels_hi = args["hyp"], args["pre"], args["labels"]
    
    # # store hypo and premise of english
    # get_data(args["data_dir"] + "dev-en.tsv" if split == "dev" else args["data_dir"] + "test-en.tsv", args)
    # en_hypo, en_premises, labels_en = args["hyp"], args["pre"], args["labels"]

    # args["hyp"] = hypo_
    # args["pre"] = premises_lang
    # args["hi_hyp"] = hi_hypo
    # args["hi_pre"] = hi_premises
    # args["en_hyp"] = en_hypo
    # args["en_pre"] = en_premises
    # args["labels"] = labels_lang
    # args["labels_hi"] = labels_hi
    # args["labels_en"] = labels_en

    # concat_hypo = [hyp_hi + " " + hyp for hyp_hi,hyp in zip(hi_hypo, hypo_)]
    # concat_premises = [pre_hi + " " + pre for pre_hi,pre in zip(hi_premises, premises_lang)]

    eval_dataset = load_and_cache_examples(args, "xnli", tokenizer, split, language=language, lang2id=lang2id, evaluate=True, probe=probe)

    args["eval_batch_size"] = args["batch_size"]
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

    # Eval!
    logger.info("***** Running evaluation {} *****".format(language))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args["eval_batch_size"])
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    sentences = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args["device"]) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args["model_type"] != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args["model_type"] in ["bert"] else None
                )  # XLM and DistilBERT don't use segment_ids
            if args["model_type"] == "xlm":
                inputs["langs"] = batch[4]
            
            #print(inputs)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            sentences = inputs["input_ids"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            sentences = np.append(sentences, inputs["input_ids"].detach().cpu().numpy(), axis=0)

    print(out_label_ids)
    eval_loss = eval_loss / nb_eval_steps
    if args["output_mode"] == "classification":
        preds = np.argmax(preds, axis=1)
    else:
        raise ValueError("No other `output_mode` for XNLI.")
    
    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    outputs = []
    predictions = []

    #output_file = True
    if False:
        logger.info("***** Save prediction ******")
        for p, l in zip(list(preds), list(out_label_ids)):
            if label_list:
                p = label_list[p]
                l = label_list[l]
            
            outputs.append(l)
            predictions.append(p)
        if len(hi_hypo) == len(hypo_):
            res = {"Concat Premises": concat_premises, "Concat Hypothesis": concat_hypo, "labels": outputs, "predictions": predictions, "Premises_lang": premises_lang, 
            "Premises_hindi": hi_premises, "Premises_english": en_premises, "Hypothesis_lang": hypo_, "Hypothesis_hindi": hi_hypo, "Hypothesis_english": en_hypo}
        else:
            res = {"Concat Premises": concat_premises, "Concat Hypothesis": concat_hypo, "labels": outputs, "predictions": predictions, "Premises_lang": premises_lang, "Hypothesis_lang": hypo_}

        df = pd.DataFrame(res)
        df.to_csv(output_file + ".csv")
    
    logger.info("***** Eval results {} *****".format(language))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    return results


def main():
    # read monolingual hindi data
    DATA_DIR = "/home/ashish/benchmark/xtreme/download/storycloze/"

    bert_path = "bert-base-multilingual-cased-LR2e-5-epoch2-MaxLen128"
    xlmr_path = "xlm-roberta-large-LR5e-6-epoch2-MaxLen128"
    
    #OUTPUT_DIR = "/home/ashish/benchmark/xtreme/outputs-temp/xnli/" + xlmr_path + "/"
    OUTPUT_DIR = "/home/ashish/benchmark/xtreme/outputs-temp/storycloze/enenesen/" + xlmr_path + "/"
    MODEL_DIR = "/home/ashish/benchmark/xtreme/outputs-temp/xnli/enenesen/" + xlmr_path + "/"
    #MODEL_DIR = "/run/user/1056/outputs-temp/xnli/" + xlmr_path + "/"
    #OUTPUT_DIR = "/home/ashish/benchmark/xtreme/outputs/xnli/bert_swapped/" + bert_path + "/"
    #OUTPUT_DIR = "/run/user/1056/outputs/xnli/bert_mixswapped/" + bert_path + "/"
    
    model_type = "xlmr"
    args = {"do_lower_case": False, "data_dir": DATA_DIR, "model_name_or_path": xlmr_path,
        "max_seq_length":512, "model_type": model_type, "task_name": "xnli", "batch_size": 8, "no_cuda": False, 
        "output_dir": OUTPUT_DIR}

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda:2" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
    #device = torch.device("cpu")
    args["n_gpu"] = torch.cuda.device_count()
    args["device"] = device

    logging.basicConfig(handlers=[logging.FileHandler(os.path.join(args["output_dir"], "test")), logging.StreamHandler()],
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Set seed
    set_seed(42)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args["model_type"]]
    print("model class is ", model_class)
    best_checkpoint = os.path.join(MODEL_DIR, 'checkpoint-best')
    #best_checkpoint = OUTPUT_DIR
    tokenizer = tokenizer_class.from_pretrained(best_checkpoint, do_lower_case=False)
    model = model_class.from_pretrained(best_checkpoint)
    model.to(args["device"])

    print("size is ", getsizeof(model))

    if args["task_name"] not in PROCESSORS:
        raise ValueError("Task not found: %s" % (args["task_name"]))
    processor = PROCESSORS[args["task_name"]]()
    args["output_mode"] = "classification"
    label_list = processor.get_labels()

    lang2id = None
    split = "dev"
    print("label list is ", label_list)

    # langs = ["dev-aren", "dev-bgen", "dev-deen", "dev-elen", "dev-esen", "dev-fren", "dev-hien", 
    #         "dev-ruen", "dev-swen", "dev-then", "dev-tren", "dev-uren", "dev-vien",
    #         "dev-zhen", "dev-enaren", "dev-enbgen", "dev-endeen", "dev-enelen", "dev-enesen", "dev-enfren", "dev-enhien", "dev-enruen",
    #         "dev-enswen", "dev-enthen", "dev-entren", "dev-enuren", "dev-envien", "dev-enzhen"]
    # paths = [DATA_DIR+"translated/dev-aren.txt", DATA_DIR+"translated/dev-bgen.txt", 
    #         DATA_DIR+"translated/dev-deen.txt", DATA_DIR+"translated/dev-elen.txt", DATA_DIR+"translated/dev-esen.txt", 
    #         DATA_DIR+"translated/dev-fren.txt", DATA_DIR+"translated/dev-hien.txt", DATA_DIR+"translated/dev-ruen.txt", DATA_DIR+"translated/dev-swen.txt", 
    #         DATA_DIR+"translated/dev-then.txt", DATA_DIR+"translated/dev-tren.txt", DATA_DIR+"translated/dev-uren.txt", DATA_DIR+"translated/dev-vien.txt", DATA_DIR+"translated/dev-zhen.txt",
    #         DATA_DIR+"translated/dev-enaren.txt", DATA_DIR+"translated/dev-enbgen.txt", DATA_DIR+"translated/dev-endeen.txt", DATA_DIR+"translated/dev-enelen.txt", DATA_DIR+"translated/dev-enesen.txt", 
    #         DATA_DIR+"translated/dev-enfren.txt", DATA_DIR+"translated/dev-enhien.txt", DATA_DIR+"translated/dev-enruen.txt", DATA_DIR+"translated/dev-enswen.txt", DATA_DIR+"translated/dev-enthen.txt", DATA_DIR+"translated/dev-entren.txt",
    #         DATA_DIR+"translated/dev-enuren.txt", DATA_DIR+"translated/dev-envien.txt", DATA_DIR+"translated/dev-enzhen.txt"]
    # langs = ["test-enar", "test-enbg", "test-ende", "test-enel", "test-enes", "test-enfr", "test-enhi", "test-enru", "test-ensw", "test-enth", 
    #         "test-entr", "test-enur", "test-envi", "test-enzh", "test-aren", "test-bgen", "test-deen", "test-elen", "test-esen", "test-fren", 
    #         "test-hien", "test-then", "test-tren", "test-uren", "test-vien",
    #         "test-zhen", "test-enaren", "test-enbgen", "test-endeen", "test-enelen", "test-enesen", "test-enfren", "test-enhien", "test-enruen",
    #         "test-enswen", "test-enthen", "test-entren", "test-enuren", "test-envien", "test-enzhen"]
    # paths = [DATA_DIR+"translated/test-enar.txt",DATA_DIR+"translated/test-enbg.txt",DATA_DIR+"translated/test-ende.txt",DATA_DIR+"translated/test-enel.txt",
    #         DATA_DIR+"translated/test-enes.txt",DATA_DIR+"translated/test-enfr.txt",DATA_DIR+"translated/test-enhi.txt",
    #         DATA_DIR+"translated/test-enru.txt", DATA_DIR+"translated/test-ensw.txt", 
    #         DATA_DIR+"translated/test-enth.txt", DATA_DIR+"translated/test-entr.txt", DATA_DIR+"translated/test-enur.txt", DATA_DIR+"translated/test-envi.txt", DATA_DIR+"translated/test-enzh.txt",
    #         DATA_DIR+"translated/test-aren.txt", DATA_DIR+"translated/test-bgen.txt", 
    #         DATA_DIR+"translated/test-deen.txt", DATA_DIR+"translated/test-elen.txt", DATA_DIR+"translated/test-esen.txt", 
    #         DATA_DIR+"translated/test-fren.txt", DATA_DIR+"translated/test-hien.txt", DATA_DIR+"translated/test-then.txt", DATA_DIR+"translated/test-tren.txt", DATA_DIR+"translated/test-uren.txt", DATA_DIR+"translated/test-vien.txt", DATA_DIR+"translated/test-zhen.txt",
    #         DATA_DIR+"translated/test-enaren.txt", DATA_DIR+"translated/test-enbgen.txt", DATA_DIR+"translated/test-endeen.txt", DATA_DIR+"translated/test-enelen.txt", DATA_DIR+"translated/test-enesen.txt", 
    #         DATA_DIR+"translated/test-enfren.txt", DATA_DIR+"translated/test-enhien.txt", DATA_DIR+"translated/test-enruen.txt", DATA_DIR+"translated/test-enswen.txt", DATA_DIR+"translated/test-enthen.txt", DATA_DIR+"translated/test-entren.txt",
    #         DATA_DIR+"translated/test-enuren.txt", DATA_DIR+"translated/test-envien.txt", DATA_DIR+"translated/test-enzhen.txt"]
    # langs = ["englishar_dev", "englishbg_dev", "englishzh_dev", "englishfr_dev", "englishde_dev", "englishel_dev", "english_dev", "englishru_dev", 
    #         "englishes_dev", "englishsw_dev", "englishth_dev", "englishtr_dev", "englishur_dev", "englishvi_dev"]
    # paths = [DATA_DIR+"translated/englishar_dev.txt", DATA_DIR+"translated/englishbg_dev.txt", DATA_DIR+"translated/englishzh_dev.txt", DATA_DIR+"translated/englishfr_dev.txt", 
    #         DATA_DIR+"translated/englishde_dev.txt", DATA_DIR+"translated/englishel_dev.txt", DATA_DIR+"translated/english_dev.txt", DATA_DIR+"translated/englishru_dev.txt", 
    #         DATA_DIR+"translated/englishes_dev.txt", DATA_DIR+"translated/englishsw_dev.txt", DATA_DIR+"translated/englishth_dev.txt", DATA_DIR+"translated/englishtr_dev.txt", 
    #         DATA_DIR+"translated/englishur_dev.txt", DATA_DIR+"translated/englishvi_dev.txt"]
    # langs = ["test-enfr", "test-enes", "test-ende", "test-enel", "test-enbg", "test-enru", "test-entr", "test-enar", "test-envi", 
    #         "test-enth", "test-enzh", "test-enhi", "test-ensw", "test-enur"]
    # paths = [DATA_DIR+"nllb/test-enfr.txt", DATA_DIR+"nllb/test-enes.txt", DATA_DIR+"nllb/test-ende.txt", DATA_DIR+"nllb/test-enel.txt", 
    #         DATA_DIR+"nllb/test-enbg.txt", DATA_DIR+"nllb/test-enru.txt", DATA_DIR+"nllb/test-entr.txt", DATA_DIR+"nllb/test-enar.txt", DATA_DIR+"nllb/test-envi.txt", 
    #         DATA_DIR+"nllb/test-enth.txt", DATA_DIR+"nllb/test-enzh.txt", DATA_DIR+"nllb/test-enhi.txt", DATA_DIR+"nllb/test-ensw.txt", DATA_DIR+"nllb/test-enur.txt"]

    langs = ["monoesen", "monoenesen", "monoenes"]
    paths = [DATA_DIR+"monoesen.txt", DATA_DIR+"monoenesen.txt", DATA_DIR+"monoenes.txt"]
    
    probes = [False]
    for i,lang in enumerate(langs):
        output_file = args["output_dir"]+"{}-results".format(lang)
        for probe in probes:
            result = evaluate(args, model, tokenizer, split, language=paths[i], lang2id=lang2id, output_file=output_file, label_list=label_list, probe=probe)
    
    #diff_analysis(args, model, tokenizer)

    #probe_input(args, model, tokenizer, "en")
    #swapped_analysis(args, model, tokenizer)


model_ = None
tokenizer_ = None
model_type = None

if __name__ == "__main__":
  main()