from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import csv
import time
import traceback

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
BASE_PATH = "/home/ashish/benchmark/xtreme/"


def translate(target, batch, tokenizer):
    input = tokenizer(batch, return_tensors="pt", padding=True)
    translated_tokens = model.generate(
        input_ids=input["input_ids"].to(device), attention_mask=input["attention_mask"].to(device), 
        forced_bos_token_id=tokenizer.get_lang_id(target))
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

def translate_en_lang_en():

    lang_code = {"zh": "zh"}
    batch_size = 10
    for lang in ["zh"]:
        try:
            print("Processing lang {}".format(lang))
            with open(BASE_PATH + "download/xnli/test-en.tsv") as file:
                tsv_file = csv.reader(file, delimiter="\t")
                premises, en_premises, zh_premises = [], [], []
                hypothesis, en_hypothesis, zh_hypothesis = [], [], []
                labels = []
                
                
                for i,line in enumerate(tsv_file):
                    #if i == 8: break
                    if i%3 == 0:
                        premises.append(line[0])
                    hypothesis.append(line[1])
                    labels.append(line[2])
                

                num_batches = int(len(premises)/batch_size)
                print("Number of batches are {}".format(num_batches))
                print("Processing premises")
                for i in range(num_batches):
                    t1 = time.time()
                    print("batch {}".format(i))
                    if i != num_batches - 1:
                        batch_pre = premises[i*batch_size: i*batch_size + batch_size]
                    else:
                        batch_pre = premises[i*batch_size:]

                    tokenizer.src_lang = "en"
                    lang_pre = translate(lang_code[lang], batch_pre, tokenizer)
                    zh_premises.extend(lang_pre)
                    tokenizer.src_lang = "zh"
                    en_pre = translate("en", lang_pre, tokenizer)
                    en_premises.extend(en_pre)

                    print("time taken is {}".format(time.time() - t1))

                num_batches = int(len(hypothesis)/batch_size)
                print("Number of batches are {}".format(num_batches))
                print("Processing hypothesis")
                for i in range(num_batches):
                    t1 = time.time()
                    print("batch {}".format(i))
                    if i != num_batches - 1:
                        batch_hypo = hypothesis[i*batch_size: i*batch_size + batch_size]
                    else:
                        batch_hypo = hypothesis[i*batch_size:]

                    tokenizer.src_lang = "en"
                    lang_hypo = translate(lang_code[lang], batch_hypo, tokenizer)
                    zh_hypothesis.extend(lang_hypo)
                    tokenizer.src_lang = "zh"
                    en_hypo = translate("en", lang_hypo, tokenizer)
                    en_hypothesis.extend(en_hypo)

                    print("time taken is {}".format(time.time() - t1))

        except Exception as e:
            print("error is {}".format(e))
            traceback.print_exc()

        file = open(BASE_PATH + "download/xnli/m2m/test-en{}en.txt".format(lang), "w+")
        for i,hypo in enumerate(en_hypothesis):
            file.write(en_premises[int(i/3)] + "\t" + hypo + "\t" + labels[i] + "\n")
        file.close()

        file = open(BASE_PATH + "download/xnli/m2m/test-en{}.txt".format(lang), "w+")
        for i,hypo in enumerate(zh_hypothesis):
            file.write(zh_premises[int(i/3)] + "\t" + hypo + "\t" + labels[i] + "\n")
        file.close()

def translate_lang_en():

    lang_code = {"zh": "zh"}
    batch_size = 10
    for lang in ["zh"]:
        try:
            print("Processing lang {}".format(lang))
            with open(BASE_PATH + "download/xnli/test-{}.tsv".format(lang)) as file:
                tsv_file = csv.reader(file, delimiter="\t")
                premises, zh_premises = [], []
                hypothesis, zh_hypothesis = [], []
                labels = []
                
                
                for i,line in enumerate(tsv_file):
                    #if i == 8: break
                    if i%3 == 0:
                        premises.append(line[0])
                    hypothesis.append(line[1])
                    labels.append(line[2])
                

                num_batches = int(len(premises)/batch_size)
                print("Number of batches are {}".format(num_batches))
                print("Processing premises")
                for i in range(num_batches):
                    t1 = time.time()
                    print("batch {}".format(i))
                    if i != num_batches - 1:
                        batch_pre = premises[i*batch_size: i*batch_size + batch_size]
                    else:
                        batch_pre = premises[i*batch_size:]

                    tokenizer.src_lang = "zh"
                    lang_pre = translate("en", batch_pre, tokenizer)
                    zh_premises.extend(lang_pre)

                    print("time taken is {}".format(time.time() - t1))

                num_batches = int(len(hypothesis)/batch_size)
                print("Number of batches are {}".format(num_batches))
                print("Processing hypothesis")
                for i in range(num_batches):
                    t1 = time.time()
                    print("batch {}".format(i))
                    if i != num_batches - 1:
                        batch_hypo = hypothesis[i*batch_size: i*batch_size + batch_size]
                    else:
                        batch_hypo = hypothesis[i*batch_size:]

                    tokenizer.src_lang = "zh"
                    lang_hypo = translate("en", batch_hypo, tokenizer)
                    zh_hypothesis.extend(lang_hypo)
                    
                    print("time taken is {}".format(time.time() - t1))

        except Exception as e:
            print("error is {}".format(e))
            traceback.print_exc()

        file = open(BASE_PATH + "download/xnli/m2m/test-{}en.txt".format(lang), "w+")
        for i,hypo in enumerate(zh_hypothesis):
            file.write(zh_premises[int(i/3)] + "\t" + hypo + "\t" + labels[i] + "\n")
        file.close()

translate_en_lang_en()
#translate_lang_en()

# translate english to chinese
# tokenizer.src_lang = "en"
# en_text = "And he said, Mama, I'm home."
# input = tokenizer(en_text, return_tensors="pt")
# translated_tokens = model.generate(
#         input_ids=input["input_ids"].to(device), attention_mask=input["attention_mask"].to(device), 
#         forced_bos_token_id=tokenizer.get_lang_id("zh"))
# chinese = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
# print(chinese)
# # => "La vie est comme une boîte de chocolat."

# # translate Chinese to English
# tokenizer.src_lang = "zh"
# input = tokenizer(chinese, return_tensors="pt")
# translated_tokens = model.generate(
#         input_ids=input["input_ids"].to(device), attention_mask=input["attention_mask"].to(device), 
#         forced_bos_token_id=tokenizer.get_lang_id("en"))
# english = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
# print(english)