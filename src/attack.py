from sentence_transformers import SentenceTransformer
import torch
import random
from tqdm import tqdm
from src.utils import load_json
import json
import os

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient

def get_embeddings(model):
    """Returns the wordpiece embedding module."""
    # base_model = getattr(model, config.model_type)
    # embeddings = base_model.embeddings.word_embeddings

    # This can be different for different models; the following is tested for Contriever
    if isinstance(model, SentenceTransformer):
        embeddings = model[0].auto_model.embeddings.word_embeddings
    else:
        embeddings = model.embeddings.word_embeddings
    return embeddings

def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids


class Attacker():
    def __init__(self, args, **kwargs) -> None:
        # assert args.attack_method in ['default', 'whitebox']
        self.args = args
        self.attack_method = args.attack_method
        self.adv_per_query = args.adv_per_query
        
        self.model = kwargs.get('model', None)
        self.c_model = kwargs.get('c_model', None)
        self.tokenizer = kwargs.get('tokenizer', None)
        self.get_emb = kwargs.get('get_emb', None)
        
        if args.attack_method == 'hotflip':
            self.max_seq_length = kwargs.get('max_seq_length', 128)
            self.pad_to_max_length = kwargs.get('pad_to_max_length', True)
            self.per_gpu_eval_batch_size = kwargs.get('per_gpu_eval_batch_size', 64)
            self.num_adv_passage_tokens = kwargs.get('num_adv_passage_tokens', 30)            

            self.num_cand = kwargs.get('num_cand', 100)
            self.num_iter = kwargs.get('num_iter', 30)
            self.gold_init = kwargs.get('gold_init', True)
            self.early_stop = kwargs.get('early_stop', False)
    
        self.all_adv_texts = load_json(f'results/adv_targeted_results/{args.eval_dataset}.json')

    def get_attack(self, target_queries) -> list:
        '''
        This function returns adv_text_groups, which contains adv_texts for M queries
        For each query, if adv_per_query>1, we use different generated adv_texts or copies of the same adv_text
        '''
        adv_text_groups = [] # get the adv_text for the iter
        if self.attack_method == "LM_targeted":
            for i in range(len(target_queries)):
                question = target_queries[i]['query']
                id = target_queries[i]['id']
                adv_texts_b = self.all_adv_texts[id]['adv_texts'][:self.adv_per_query]
                adv_text_a = question + "."
                adv_texts = [adv_text_a + i for i in adv_texts_b]
                adv_text_groups.append(adv_texts)  
        elif self.attack_method == 'hotflip':
            adv_text_groups = self.hotflip(target_queries)
        else: raise NotImplementedError
        return adv_text_groups       
     

    def hotflip(self, target_queries, adv_b=None, **kwargs) -> list:
        device = 'cuda'
        print('Doing HotFlip attack!')
        adv_text_groups = []
        for query_score in tqdm(target_queries):
            query = query_score['query']
            top1_score = query_score['top1_score']
            id = query_score['id']
            adv_texts_b = self.all_adv_texts[id]['adv_texts']

            adv_texts=[]
            for j in range(self.adv_per_query):
                adv_b = adv_texts_b[j]
                adv_b = self.tokenizer(adv_b, max_length=self.max_seq_length, truncation=True, padding=False)['input_ids']
                if self.gold_init:
                    adv_a = query
                    adv_a = self.tokenizer(adv_a, max_length=self.max_seq_length, truncation=True, padding=False)['input_ids']

                else: # init adv passage using [MASK]
                    adv_a = [self.tokenizer.mask_token_id] * self.num_adv_passage_tokens

                embeddings = get_embeddings(self.c_model)
                embedding_gradient = GradientStorage(embeddings)
                
                adv_passage = adv_a + adv_b # token ids
                adv_passage_ids = torch.tensor(adv_passage, device=device).unsqueeze(0)
                adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
                adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device)  

                q_sent = self.tokenizer(query, max_length=self.max_seq_length, truncation=True, padding="max_length" if self.pad_to_max_length else False, return_tensors="pt")
                q_sent = {key: value.cuda() for key, value in q_sent.items()}
                q_emb = self.get_emb(self.model, q_sent).detach()            
                
                for it_ in range(self.num_iter):
                    grad = None   
                    self.c_model.zero_grad()

                    p_sent = {'input_ids': adv_passage_ids, 
                            'attention_mask': adv_passage_attention, 
                            'token_type_ids': adv_passage_token_type}
                    p_emb = self.get_emb(self.c_model, p_sent)  

                    if self.args.score_function == 'dot':
                        sim = torch.mm(p_emb, q_emb.T)
                    elif self.args.score_function == 'cos_sim':
                        sim = torch.cosine_similarity(p_emb, q_emb)
                    else: raise KeyError
                    
                    loss = sim.mean()
                    if self.early_stop and sim.item() > top1_score + 0.1: break
                    loss.backward()                

                    temp_grad = embedding_gradient.get()
                    if grad is None:
                        grad = temp_grad.sum(dim=0)
                    else:
                        grad += temp_grad.sum(dim=0)

                    token_to_flip = random.randrange(len(adv_a))
                    candidates = hotflip_attack(grad[token_to_flip],
                                                embeddings.weight,
                                                increase_loss=True,
                                                num_candidates=self.num_cand,
                                                filter=None)                
                    current_score = 0
                    candidate_scores = torch.zeros(self.num_cand, device=device) 

                    temp_score = loss.sum().cpu().item()
                    current_score += temp_score

                    for i, candidate in enumerate(candidates):
                        temp_adv_passage = adv_passage_ids.clone()
                        temp_adv_passage[:, token_to_flip] = candidate
                        temp_p_sent = {'input_ids': temp_adv_passage, 
                            'attention_mask': adv_passage_attention, 
                            'token_type_ids': adv_passage_token_type}
                        temp_p_emb = self.get_emb(self.c_model, temp_p_sent)
                        with torch.no_grad():
                            if self.args.score_function == 'dot':
                                temp_sim = torch.mm(temp_p_emb, q_emb.T)
                            elif self.args.score_function == 'cos_sim':
                                temp_sim = torch.cosine_similarity(temp_p_emb, q_emb)
                            else: raise KeyError                        
                            can_loss = temp_sim.mean()
                            temp_score = can_loss.sum().cpu().item()
                            candidate_scores[i] += temp_score

                    # if find a better one, update
                    if (candidate_scores > current_score).any():
                        best_candidate_idx = candidate_scores.argmax()
                        adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
                    else:
                        continue      
                
                adv_text = self.tokenizer.decode(adv_passage_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                adv_texts.append(adv_text)
            adv_text_groups.append(adv_texts)
        
        return adv_text_groups
