import re
import sys
import tqdm
import torch
import logging
import argparse
import numpy as np
from prettytable import PrettyTable
from transformers import AutoModel, AutoTokenizer

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = '../SentEval'
PATH_TO_DATA = '../SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def cal_avg_cosine(k, n=100000):
    cos = torch.nn.CosineSimilarity(dim=-1)
    s = torch.tensor(k[:100000]).cuda()
    kk = []
    pbar = tqdm.tqdm(total=n)
    with torch.no_grad():
        for i in range(n):
            kk.append(cos(s[i:i+1], s).mean().item())
            pbar.set_postfix({'cosine': sum(kk)/len(kk)})
            pbar.update(1)
    return sum(kk) /len(kk)


def s_eval(args):
    se, task = args[0], args[1]
    return se.eval(task)


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def denoising(model, template, tokenizer, device, mask_num):
    model.eval()
    total_length = 200

    template = template.replace('*mask*', tokenizer.mask_token)\
                       .replace('*sep+*', '').replace('*cls*', '').replace('*sent_0*', ' ')
    
    template = template.split(' ')

    bs_str = template[0].replace('_', ' ')
    es_str = template[1].replace('_', ' ')
    bs = tokenizer.encode(bs_str, add_special_tokens=False)
    es = tokenizer.encode(es_str, add_special_tokens=False)
    
    input_ids, attention_mask = [], []
    template = tokenizer.encode(bs_str + es_str)
    for i in range(total_length - len(template) + 1):
        input_ids.append([template[0]] + 
                         bs + 
                         [tokenizer.pad_token_id] * i + 
                         es + 
                         [template[-1]] + 
                         [tokenizer.pad_token_id] * (total_length - len(template) - i))
        
        attention_mask.append([1] * (len(template) + i) + [0] * (total_length - len(template) - i))

    input_ids = torch.Tensor(input_ids).to(device).long()
    attention_mask = torch.Tensor(attention_mask).to(device).long()

    mask = input_ids == tokenizer.mask_token_id

    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True)
        
        last_hidden = outputs.hidden_states[-1]
        noise = last_hidden[mask]
        
        noise = noise.reshape(-1, mask_num, noise.shape[-1])
        noise = noise[:, mask_num-1, :]
    
    noise.requires_grad = False
    return noise, len(template)


def main():
    arg_list = ['--model_name_or_path', '../result/CoT-Roberta',
                '--mode', 'test',
                '--mask_embedding_sentence',
                '--mask_num', '2',
                '--mask_embedding_sentence_template', "*cls*_The_sentence_of_'_*sent_0*_'_means_*mask*_,_so_it_can_be_summarized_as_*mask*_._*sep+*"]  
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_only", action='store_true')
    parser.add_argument('--mlm_head_predict', action='store_true')
    parser.add_argument('--remove_continue_word', action='store_true')
    parser.add_argument('--mask_num', type=int, default=2)
    parser.add_argument('--mask_embedding_sentence', action='store_true')
    parser.add_argument('--mask_embedding_sentence_use_org_pooler', action='store_true')
    parser.add_argument('--mask_embedding_sentence_template', type=str, default=None)
    parser.add_argument('--mask_embedding_sentence_delta', action='store_true')
    parser.add_argument('--mask_embedding_sentence_use_pooler', action='store_true')
    parser.add_argument('--mask_embedding_sentence_autoprompt', action='store_true')
    parser.add_argument('--mask_embedding_sentence_org_mlp', action='store_true')
    parser.add_argument("--tokenizer_name", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str, help="Transformers' model name or path")
    
    parser.add_argument("--pooler", type=str,
                        choices=['cls', 'cls_before_pooler', 'avg',  'avg_first_last'],
                        default='cls', 
                        help="Which pooler to use")
    
    parser.add_argument("--mode", type=str, 
                        choices=['dev', 'test', 'fasttest'],
                        default='test', 
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    
    parser.add_argument('--calc_anisotropy', action='store_true')

    args = parser.parse_args(arg_list)

    # Load transformers' model checkpoint
    if args.mask_embedding_sentence_org_mlp:
        # only for bert-base
        from transformers import BertForMaskedLM, BertConfig
        config = BertConfig.from_pretrained("bert-base-uncased")
        mlp = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config).cls.predictions.transform

        if 'result' in args.model_name_or_path:
            state_dict = torch.load(args.model_name_or_path + '/pytorch_model.bin')
            new_state_dict = {}
            
            for key, param in state_dict.items():
                # Replace "mlp" to "pooler"
                if 'pooler' in key:
                    key = key.replace("pooler.", "")
                    new_state_dict[key] = param

            mlp.load_state_dict(new_state_dict)

    model = AutoModel.from_pretrained(args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    if args.mask_embedding_sentence_autoprompt:
        state_dict = torch.load(args.model_name_or_path+'/pytorch_model.bin')

        p_mbv = state_dict['p_mbv']
        template = args.mask_embedding_sentence_template
        template = template.replace('*mask*', tokenizer.mask_token)\
                           .replace('*sep+*', '').replace('*cls*', '').replace('*sent_0*', ' ').replace('_', ' ')
        
        mask_embedding_template = tokenizer.encode(template)
        mask_index = mask_embedding_template.index(tokenizer.mask_token_id)
        index_mbv = mask_embedding_template[1:mask_index] + mask_embedding_template[mask_index + 1: -1]

        dict_mbv = index_mbv
        fl_mbv = [i <= 3 for i, _ in enumerate(index_mbv)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    if args.mask_embedding_sentence_org_mlp:
        mlp = mlp.to(device)

    if args.mask_embedding_sentence_delta:
        with torch.no_grad():
            delta, template_len = denoising(model, args.mask_embedding_sentence_template, tokenizer, device, args.mask_num)

    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    if args.remove_continue_word:
        pun_remove_set = {'?', '*', '#', '´', '’', '=', '…', '|', '~', '/', '‚', '¿', '–', '»', '-', '€', '‘', '"', '(', '•', '`', '$', ':', '[', '”', '%', '£', '<', '[UNK]', ';', '“', '@', '_', '{', '^', ',', '.', '!', '™', '&', ']', '>', '\\', "'", ')', '+', '—'}
        if args.model_name_or_path == 'roberta-base':
            remove_set = {'Ġ.', 'Ġa', 'Ġthe', 'Ġin', 'a', 'Ġ, ', 'Ġis', 'Ġto', 'Ġof', 'Ġand', 'Ġon', 'Ġ\'', 's', '.', 'the', 'Ġman', '-', 'Ġwith', 'Ġfor', 'Ġat', 'Ġwoman', 'Ġare', 'Ġ"', 'Ġthat', 'Ġit', 'Ġdog', 'Ġsaid', 'Ġplaying', 'Ġwas', 'Ġas', 'Ġfrom', 'Ġ:', 'Ġyou', 'Ġan', 'i', 'Ġby'}
        else:
            remove_set = {".", "a", "the", "in", ",", "is", "to", "of", "and", "'", "on", "man", "-", "s", "with", "for", "\"", "at", "##s", "woman", "are", "it", "two", "that", "you", "dog", "said", "playing", "i", "an", "as", "was", "from", ":", "by", "white"}

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        if args.mask_embedding_sentence and args.mask_embedding_sentence_template is not None:
            template = args.mask_embedding_sentence_template
            template = template.replace('*mask*', tokenizer.mask_token )\
                               .replace('_', ' ').replace('*sep+*', '').replace('*cls*', '')

            for i, s in enumerate(sentences):
                if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
                sentences[i] = template.replace('*sent 0*', s).strip()
        elif args.remove_continue_word:
            for i, s in enumerate(sentences):
                sentences[i] = ' ' if args.model_name_or_path == 'roberta-base' else ''
                es = tokenizer.encode(' ' + s, add_special_tokens=False)
                for _, w in enumerate(tokenizer.convert_ids_to_tokens(es)):
                    if args.model_name_or_path == 'roberta-base':
                        # roberta base
                        if 'Ġ' not in w or w in remove_set:
                            pass
                        else:
                            if re.search('[a-zA-Z0-9]', w) is not None:
                                sentences[i] += w.replace('Ġ', '').lower() + ' '
                    elif w not in remove_set and w not in pun_remove_set and '##' not in w:
                        # bert base
                        sentences[i] += w.lower() + ' '
                if len(sentences[i]) == 0: sentences[i] = '[PAD]'

        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )

        for k in batch:
            batch[k] = batch[k].to(device) if batch[k] is not None else None
        
        # Get raw embeddings
        with torch.no_grad():
            if args.embedding_only:
                hidden_states = None
                pooler_output = None
                last_hidden = model.embeddings.word_embeddings(batch['input_ids'])

                if args.remove_continue_word:
                    batch['attention_mask'][batch['input_ids'] == tokenizer.cls_token_id] = 0
                    batch['attention_mask'][batch['input_ids'] == tokenizer.sep_token_id] = 0
            elif args.mask_embedding_sentence_autoprompt:
                input_ids = batch['input_ids']
                inputs_embeds = model.embeddings.word_embeddings(input_ids)
                p = torch.arange(input_ids.shape[1]).to(input_ids.device).view(1, -1)
                b = torch.arange(input_ids.shape[0]).to(input_ids.device)

                for i, k in enumerate(dict_mbv):
                    if fl_mbv[i]:
                        index = ((input_ids == k) * p).max(-1)[1]
                    else:
                        index = ((input_ids == k) * -p).min(-1)[1]
                        
                    inputs_embeds[b, index] = p_mbv[i]
                
                batch['input_ids'], batch['inputs_embeds'] = None, inputs_embeds
                outputs = model(**batch, output_hidden_states=True, return_dict=True)
                batch['input_ids'] = input_ids

                last_hidden = outputs.last_hidden_state
                pooler_output = last_hidden[input_ids == tokenizer.mask_token_id]

                if args.mask_embedding_sentence_org_mlp:
                    pooler_output = mlp(pooler_output)
                if args.mask_embedding_sentence_delta:
                    blen = batch['attention_mask'].sum(-1) - template_len
                    
                    if args.mask_embedding_sentence_org_mlp:
                        pooler_output -= mlp(delta[blen])
                    else:
                        pooler_output -= delta[blen]
                if args.mask_embedding_sentence_use_pooler:
                    pooler_output = model.pooler.dense(pooler_output)
                    pooler_output = model.pooler.activation(pooler_output)

            else:
                outputs = model(**batch, output_hidden_states=True, return_dict=True)

                try:
                    pooler_output = outputs.pooler_output
                except AttributeError:
                    pooler_output = outputs['last_hidden_state'][:, 0, :]

                if args.mask_embedding_sentence:
                    last_hidden = outputs.last_hidden_state
                    pooler_output = last_hidden[batch['input_ids'] == tokenizer.mask_token_id]

                    pooler_output = pooler_output.view(-1, args.mask_num, pooler_output.shape[-1])
                    pooler_output = pooler_output[:, args.mask_num - 1, :]  
                    
                    # for unsupervised bert, none of those ifs are satisfied
                    if args.mask_embedding_sentence_org_mlp:
                        pooler_output = mlp(pooler_output)
                    if args.mask_embedding_sentence_delta:
                        blen = batch['attention_mask'].sum(-1) - template_len

                        if args.mask_embedding_sentence_org_mlp:
                            pooler_output -= mlp(delta[blen])
                        else:
                            pooler_output -= delta[blen]

                    if args.mask_embedding_sentence_use_org_pooler:
                        pooler_output = mlp(pooler_output)
                    if args.mask_embedding_sentence_use_pooler:
                        pooler_output = model.pooler.dense(pooler_output)
                        pooler_output = model.pooler.activation(pooler_output)
                else:
                    last_hidden = outputs.last_hidden_state
                    hidden_states = outputs.hidden_states

        # Apply different pooler
        if args.mask_embedding_sentence:
            return pooler_output.view(batch['input_ids'].shape[0], -1).cpu()
        elif args.pooler == 'cls':
            # There is a linear + activation layer after CLS representation
            return pooler_output.cpu()
        elif args.pooler == 'cls_before_pooler':
            batch['input_ids'][(batch['input_ids'] == 0) | (batch['input_ids'] == 101) | (batch['input_ids'] == 102)] = batch['input_ids'].max()
            index = batch['input_ids'].topk(3, dim=-1, largest=False)[1]
            index2 = torch.arange(batch['input_ids'].shape[0]).to(index.device)
            r = last_hidden[index2, index[:, 0], :]
            for i in range(1, 3):
                r += last_hidden[index2, index[:, i], :]
            return (r / 3).cpu()
        elif args.pooler == "avg":
            return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
        elif args.pooler == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        elif args.pooler == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        else:
            raise NotImplementedError

    if args.calc_anisotropy:
        with open('./data/wiki1m_for_simcse.txt') as f:
            lines = f.readlines()[:100000]
        batch, embeds = [], []
        print('Get Sentence Embeddings....')
        for line in tqdm.tqdm(lines):
            batch.append(line.replace('\n', '').lower().split()[:32])
            if len(batch) >= 128:
                embeds.append(batcher(None, batch).detach().numpy())
                batch = []
        embeds.append(batcher(None, batch).detach().numpy())
        print('Calculate anisotropy....')
        embeds = np.concatenate(embeds, axis=0)
        cosine = cal_avg_cosine(embeds)
        print('Avg. Cos:', cosine)
        exit(0)

    results = {}
    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        scores = []
        task_names = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        scores = []
        task_names = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        scores = []
        task_names = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        scores = []
        task_names = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")

        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

if __name__ == "__main__":
    main()
