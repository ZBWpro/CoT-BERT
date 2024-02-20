import torch
import torch.nn as nn
import torch.distributed as dist

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    def __init__(self, config, scale=1):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*scale, config.hidden_size*scale)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

def denoising(cls, encoder, template, type='pos-1', device='cuda', evaluation=False):
    with torch.set_grad_enabled(not cls.model_args.mask_embedding_sentence_delta_freeze and not evaluation):
        if type == 'pos-1':
            bs = cls.bs
            es = cls.es
        elif type == 'pos-2':
            bs = cls.bs2
            es = cls.es2
        elif type == 'neg-1':
            bs = cls.bs3
            es = cls.es3
        elif type == 'neg-2':
            bs = cls.bs4
            es = cls.es4
        else:
            raise ValueError(f'unknown type {type}')
        
        input_ids, attention_mask = [], []
        for i in range(cls.total_length - len(template) + 1):
            input_ids.append([template[0]] + 
                             bs + 
                             [cls.pad_token_id] * i +
                             es +
                             [template[-1]] +
                             [cls.pad_token_id] * (cls.total_length - len(template) - i))
            
            attention_mask.append([1] * (len(template) + i) + [0] * (cls.total_length - len(template) - i))

        input_ids = torch.Tensor(input_ids).to(device).long()
        attention_mask = torch.Tensor(attention_mask).to(device).long()

        # CoT-BERT Authors: Since we haven't made any modifications related to the auto-prompt, 
        #                   there's a high probability that the following code may not function correctly.        
        if cls.model_args.mask_embedding_sentence_autoprompt:
            inputs_embeds = encoder.embeddings.word_embeddings(input_ids)
            p = torch.arange(input_ids.shape[1]).to(device).view(1, -1)
            b = torch.arange(input_ids.shape[0]).to(device)

            for i, k in enumerate(cls.dict_mbv):
                if cls.fl_mbv[i]:
                    index = ((input_ids == k) * p).max(-1)[1]
                else:
                    index = ((input_ids == k) * - p).min(-1)[1]

                inputs_embeds[b, index] = cls.p_mbv[i]
        else:
            inputs_embeds = None

        if evaluation:
            with torch.no_grad():
                mask = input_ids == cls.mask_token_id    
                outputs = encoder(input_ids=input_ids if inputs_embeds is None else None,
                                  inputs_embeds=inputs_embeds,
                                  attention_mask=attention_mask,
                                  output_hidden_states=True, return_dict=True)            
                
                last_hidden = outputs.last_hidden_state
                noise = last_hidden[mask]
        else:
            mask = input_ids == cls.mask_token_id    
            outputs = encoder(input_ids=input_ids if inputs_embeds is None else None,
                              inputs_embeds=inputs_embeds,
                              attention_mask=attention_mask,
                              output_hidden_states=True, return_dict=True)            
            
            last_hidden = outputs.last_hidden_state
            noise = last_hidden[mask]

        noise = noise.view(-1, cls.mask_num, noise.shape[-1])
        noise = noise[:, cls.mask_num - 1, :]

        return noise, len(template)


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    if cls.model_args.mask_embedding_sentence_org_mlp:
        from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
        cls.mlp = BertPredictionHeadTransform(config)
    else:
        cls.mlp = MLPLayer(config, scale=cls.model_args.mask_embedding_sentence_num_masks)
    
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_forward(cls,
               encoder,
               input_ids=None,
               attention_mask=None,
               token_type_ids=None,
               position_ids=None,
               head_mask=None,
               inputs_embeds=None,
               output_attentions=None,
               output_hidden_states=None,
               labels=None,
               return_dict=None,
):
    if cls.model_args.mask_embedding_sentence_delta:
        noise1, template_length1 = denoising(cls=cls, encoder=encoder, template=cls.mask_embedding_template, type='pos-1', device=input_ids.device)

        if len(cls.model_args.mask_embedding_sentence_different_template) > 0:
            noise2, template_length2 = denoising(cls=cls, encoder=encoder, template=cls.mask_embedding_template2, type='pos-2', device=input_ids.device)
        
        if len(cls.model_args.mask_embedding_sentence_negative_template) > 0:
            noise3, template_length3 = denoising(cls=cls, encoder=encoder, template=cls.mask_embedding_template3, type='neg-1', device=input_ids.device)

        if len(cls.model_args.mask_embedding_sentence_different_negative_template) > 0:
            noise4, template_length4 = denoising(cls=cls, encoder=encoder, template=cls.mask_embedding_template4, type='neg-2', device=input_ids.device)

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    
    batch_size = input_ids.size(0)

    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (batch_size * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (batch_size * num_sent, len)

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (batch_size * num_sent, len)

    if cls.model_args.mask_embedding_sentence_autoprompt:
        inputs_embeds = encoder.embeddings.word_embeddings(input_ids)
        p = torch.arange(input_ids.shape[1]).to(input_ids.device).view(1, -1)
        b = torch.arange(input_ids.shape[0]).to(input_ids.device)

        for i, k in enumerate(cls.dict_mbv):
            if cls.model_args.mask_embedding_sentence_autoprompt_continue_training_as_positive and i%2 == 0:
                continue

            if cls.fl_mbv[i]:
                index = ((input_ids == k) * p).max(-1)[1]
            else:
                index = ((input_ids == k) * -p).min(-1)[1]
            
            inputs_embeds[b, index] = cls.p_mbv[i]

    outputs = encoder(
        None if cls.model_args.mask_embedding_sentence_autoprompt else input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
    )

    if cls.model_args.mask_embedding_sentence:
        last_hidden = outputs.last_hidden_state

        pooler_output = last_hidden[input_ids == cls.mask_token_id]

        pooler_output = pooler_output.view(-1, cls.mask_num, pooler_output.shape[-1])
        pooler_output = pooler_output[:, cls.mask_num - 1, :]

        if cls.model_args.mask_embedding_sentence_delta:
            if cls.model_args.mask_embedding_sentence_org_mlp:
                pooler_output = cls.mlp(pooler_output)

            if len(cls.model_args.mask_embedding_sentence_different_template) > 0:
                pooler_output = pooler_output.view(batch_size, num_sent, -1)
                attention_mask = attention_mask.view(batch_size, num_sent, -1)

                entire_length = attention_mask.sum(-1)

                token_length = entire_length - template_length1
                pooler_output[:, 0, :] -= noise1[token_length[:, 0]]

                token_length = entire_length - template_length2
                pooler_output[:, 1, :] -= noise2[token_length[:, 1]]
                
                if num_sent == 3 and len(cls.model_args.mask_embedding_sentence_negative_template) == 0:
                    pooler_output[:, 2, :] -= noise3[token_length[:, 2]]
                
                if len(cls.model_args.mask_embedding_sentence_negative_template) > 0:
                    token_length = entire_length - template_length3
                    pooler_output[:, 2, :] -= noise3[token_length[:, 2]]

                if len(cls.model_args.mask_embedding_sentence_different_negative_template) > 0:
                    token_length = entire_length - template_length4
                    pooler_output[:, 3, :] -= noise4[token_length[:, 3]]
            else:
                token_length = attention_mask.sum(-1) - template_length1
                pooler_output -= noise1[token_length]

        pooler_output = pooler_output.view(batch_size * num_sent, -1)
    
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if (
        not cls.model_args.mask_embedding_sentence_delta
        or not cls.model_args.mask_embedding_sentence_org_mlp
    ):
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]
    elif num_sent == 4:
        z3, z4 = pooler_output[:, 2], pooler_output[:, 3]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent == 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)
        elif num_sent == 4:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            z4_list = [torch.zeros_like(z4) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            dist.all_gather(tensor_list=z4_list, tensor=z4.contiguous())
            z3_list[dist.get_rank()] = z3
            z4_list[dist.get_rank()] = z4
            z3 = torch.cat(z3_list, 0)
            z4 = torch.cat(z4_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]

        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2

        # Get full batch embeddings: (batch size x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    if cls.model_args.dot_sim:
        cos_sim = torch.mm(torch.sigmoid(z1), torch.sigmoid(z2.permute(1, 0)))
    else:
        cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    if cls.model_args.norm_instead_temp:
        cos_sim *= cls.sim.temp
        cmin, cmax = cos_sim.min(), cos_sim.max()
        cos_sim = (cos_sim - cmin) / (cmax - cmin) / cls.sim.temp

    if num_sent == 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        z2_z3_cos = cls.sim(z2.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos, z2_z3_cos], 1)
    elif num_sent == 4:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        z3_z4_cos = cls.sim(z3.unsqueeze(1), z4.unsqueeze(0))
        z2_z4_cos = cls.sim(z2.unsqueeze(1), z4.unsqueeze(0))
        cos_sim2 = torch.cat([z3_z4_cos, z2_z4_cos], 1)

    loss_fct = nn.CrossEntropyLoss()

    labels = torch.arange(cos_sim.size(0)).long().to(input_ids.device)

    # Calculate loss with hard negatives
    # if num_sent == 3:
    #     # Note that weights are actually logits of weights
    #     # z3_weight = cls.model_args.hard_negative_weight
        
    #     z3_weight = 0.0

    #     weights1 = torch.tensor(
    #         [[0.0] * z1_z3_cos.size(-1) + 
    #          [0.0] * i +
    #          [z3_weight] + 
    #          [0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
    #     ).to(input_ids.device)
        
    #     weights2 = torch.tensor(
    #         [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + 
    #          [0.0] * i + 
    #          [z3_weight] + 
    #          [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
    #     ).to(input_ids.device)

    #     cos_sim = cos_sim + weights1 + weights2

    loss = loss_fct(cos_sim, labels)

    # Calculate loss for MLM
    # if not cls.model_args.add_pseudo_instances and mlm_outputs is not None and mlm_labels is not None:
    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states if not cls.model_args.only_embedding_training else None,
        attentions=outputs.attentions if not cls.model_args.only_embedding_training else None,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    if cls.model_args.mask_embedding_sentence_delta and not cls.model_args.mask_embedding_sentence_delta_no_delta_eval :
        noise, template_length = denoising(cls=cls, encoder=encoder, template=cls.mask_embedding_template, type='pos-2', device=input_ids.device, evaluation=True)

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    if cls.model_args.mask_embedding_sentence and hasattr(cls, 'bs'):
        new_input_ids = []
        bs = torch.LongTensor(cls.bs).to(input_ids.device)
        es = torch.LongTensor(cls.es).to(input_ids.device)

        for i in input_ids:
            ss = i.shape[0]
            ii = i[i != cls.pad_token_id]

            ni = [ii[:1], bs]
            if ii.shape[0] > 2:
                ni += [ii[1:-1]]
            
            ni += [es, ii[-1:]]
            if ii.shape[0] < i.shape[0]:
                ni += [i[i == cls.pad_token_id]]
            
            ni = torch.cat(ni)
            
            try:
                assert ss + bs.shape[0] + es.shape[0] == ni.shape[0]
            except:
                print(ss + bs.shape[0] + es.shape[0])
                print(ni.shape[0])
                print(i.tolist())
                print(ni.tolist())
                assert 0

            new_input_ids.append(ni)

        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = (input_ids != cls.pad_token_id).long()
        token_type_ids = None

    if cls.model_args.mask_embedding_sentence_autoprompt:
        inputs_embeds = encoder.embeddings.word_embeddings(input_ids)

        with torch.no_grad():
            p = torch.arange(input_ids.shape[1]).to(input_ids.device).view(1, -1)
            b = torch.arange(input_ids.shape[0]).to(input_ids.device)
            for i, k in enumerate(cls.dict_mbv):
                if cls.fl_mbv[i]:
                    index = ((input_ids == k) * p).max(-1)[1]
                else:
                    index = ((input_ids == k) * -p).min(-1)[1]
                inputs_embeds[b, index] = cls.p_mbv[i]

    outputs = encoder(
        None if cls.model_args.mask_embedding_sentence_autoprompt else input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
    )

    if cls.model_args.mask_embedding_sentence and hasattr(cls, 'bs'):
        last_hidden = outputs.last_hidden_state
        pooler_output = last_hidden[input_ids == cls.mask_token_id]

        pooler_output = pooler_output.view(-1, cls.mask_num, pooler_output.shape[-1])
        pooler_output = pooler_output[:, cls.mask_num - 1, :]

        if cls.model_args.mask_embedding_sentence_delta and not cls.model_args.mask_embedding_sentence_delta_no_delta_eval :
            token_length = attention_mask.sum(-1) - template_length

            if cls.model_args.mask_embedding_sentence_org_mlp and not cls.model_args.mlp_only_train:
                pooler_output, noise = cls.mlp(pooler_output), cls.mlp(noise)

            pooler_output -= noise[token_length]

        if cls.model_args.mask_embedding_sentence_avg:
            pooler_output = pooler_output.view(input_ids.shape[0], -1)
        else:
            pooler_output = pooler_output.view(input_ids.shape[0], -1, pooler_output.shape[-1]).mean(1)
            
    if not cls.model_args.mlp_only_train and not cls.model_args.mask_embedding_sentence_org_mlp:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )

class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config)
        self.total_length = 80

        if self.model_args.mask_embedding_sentence_autoprompt:
            # register p_mbv in init, avoid not saving weight
            self.p_mbv = torch.nn.Parameter(torch.zeros(10))
            for param in self.bert.parameters():
                param.requires_grad = False

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )


class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config)
        self.total_length = 80

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
