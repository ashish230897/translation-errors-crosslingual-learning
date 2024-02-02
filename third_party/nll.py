from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import csv
import torch
import string

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="swh_Latn", tgt_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-1.3B")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

from indictrans import Transliterator
trn = Transliterator(source='eng', target='urd', build_lookup=True)

BASE_PATH = "/home/ashish/benchmark/xtreme/"

def if_number(char):
    try:
        float(char)
    except ValueError:
        return False
    
    return True


def is_english(token):

    try:
        val = int(token)
        return True
    except ValueError:
        pass

    for char in token.lower():
        if char >= 'a' and char <= 'z':
            return True 
        elif if_number(char):
            pass
        else: return False
    
    return True
 

def check_transliterate(text):
    
    tokens = text.split(" ")
    flag = True
    
    for token in tokens:
        
        token = token.strip().lower().replace("\n", "").translate(str.maketrans('', '', string.punctuation+'\u0964'))
        if len(token) > 0 and not is_english(token):
            flag = False
            break
    
    return flag

def transliterate(text):
    input = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(
        input_ids=input["input_ids"].to(device), attention_mask=input["attention_mask"].to(device), 
        forced_bos_token_id=tokenizer.lang_code_to_id["swh_Latn"])
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def translate_lang_en():

    try:
        with open(BASE_PATH + "download/xnli/test-sw.tsv") as file:
            tsv_file = csv.reader(file, delimiter="\t")
            premises, en_premises = [], []
            hypothesis, en_hypothesis = [], []
            labels = []
            
            for i,line in enumerate(tsv_file):
                #if i == 4: break
                if i%3 == 0: premises.append(line[0])
                hypothesis.append(line[1])
                labels.append(line[2])
            
            print("Processing premises")
            for i,pre in enumerate(premises):
                if i % 100 == 0: print(i)
                if check_transliterate(pre):
                    #pre = trn.transform(pre)
                    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="eng_Latn", tgt_lang="swh_Latn")
                    pre = transliterate(pre)
                    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="swh_Latn", tgt_lang="eng_Latn")

                pre = tokenizer(pre, return_tensors="pt")
                
                translated_tokens = model.generate(
                    input_ids=pre["input_ids"].to(device), attention_mask=pre["attention_mask"].to(device), 
                    forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
                pre_en = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                en_premises.append(pre_en)
            
            print("Processing hypothesis")
            for i,hypo in enumerate(hypothesis):
                if i % 100 == 0: print(i)
                if check_transliterate(hypo):
                    #hypo = trn.transform(hypo)
                    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="eng_Latn", tgt_lang="swh_Latn")
                    hypo = transliterate(hypo)
                    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="swh_Latn", tgt_lang="eng_Latn")
                
                hyp = tokenizer(hypo, return_tensors="pt")
                translated_tokens = model.generate(
                    input_ids=hyp["input_ids"].to(device), attention_mask=hyp["attention_mask"].to(device), 
                    forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
                hyp_en = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                en_hypothesis.append(hyp_en)
    except Exception as e:
        print(e)

    file = open(BASE_PATH + "download/xnli/nllb/swen13_test.txt", "w+")
    for i,hypo in enumerate(en_hypothesis):
        file.write(en_premises[int(i/3)] + "\t" + hypo + "\t" + labels[i] + "\n")
    file.close()


def translate_text(text):
    tokens = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(
        input_ids=tokens["input_ids"].to(device), attention_mask=tokens["attention_mask"].to(device), 
        forced_bos_token_id=tokenizer.lang_code_to_id["hin_Deva"])
    gen = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(gen)

def main():
    translate_lang_en()
    #translate_text("Uski maa ne bataya ki wo ghar pahuch gaya")
    #print(trn.transform("Uski maa ne bataya ki wo ghar pahuch gaya"))

if __name__ == "__main__":
    main()