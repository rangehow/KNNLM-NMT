import os

import logging
import time
import numpy as np
import torch
from torch import nn
from enum import Enum, auto
from pathlib import Path
from knnretriever import knnretriver
import faiss
import faiss.contrib.torch_utils
import torch.nn.functional as F
import config
logger = logging.getLogger(__name__)
logger.setLevel(20)


class DIST(Enum):
    l2 = auto()
    dot = auto()

    @staticmethod
    def from_string(s):
        try:
            return DIST[s.lower()]
        except KeyError:
            raise ValueError()


class KEY_TYPE(Enum):
    last_ffn_input = auto()
    last_ffn_output = auto()

    @staticmethod
    def from_string(s):
        try:
            return KEY_TYPE[s.lower()]
        except KeyError:
            raise ValueError()


class KNNWrapper(object):
    def __init__(
        self,
        dstore_dir,
        embeddings=None,
        knn_keytype=None,
        lmbda=0.25,
        knn_temp=10,
        assistant_model=None,
        k=10,
        vdb_type=None
    ):
        self.embeddings = embeddings  # 只需要是类，不需要被实例化，实例化放到setup里面做。但这个其实只需要在存在辅助模型的时候才有用，如果是自己不需要。
        self.dstore_dir = dstore_dir
        self.lmbda = lmbda
        self.knn_temperature = knn_temp
        self.knn_keytype = (
            KEY_TYPE.last_ffn_output if knn_keytype is None else knn_keytype
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k=k
        self.vdb_type=vdb_type

        # lazy init
        self.model = None
        self.retriever = None
        self.vocab_size = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.hook_handles = []
        self.assistant_model = None
        self.input_ids = None
        self.assistant_pastkv = None

    def setup_faiss(self,domain,vdb_type):
        # 这个函数需要完成的是检索器的导入
        prefix = "Translate this from Deutsch to English:"
        shot = {
            "it": """\nDeutsch:Zeigt den aktuellen Wert der Feldvariable an.\nEnglish:Displays the current value of the field variable.\nDeutsch:In diesem Bereich wählen Sie die relativen Größen bezogen auf die Basisgröße.\nEnglish:In this section, you can determine the relative sizes for each type of element with reference to the base size.\nDeutsch:Geben Sie einen kurzen, beschreibenden Namen für die Schnittstelle ein.\nEnglish:Simply enter a short human-readable description for this device.""",
            "koran": """\nDeutsch:So führt Gott (im Gleichnis) das Wahre und das Falsche an.\nEnglish:This is how God determines truth and falsehood.\nDeutsch:Da kamen sie auf ihn zu geeilt.\nEnglish:So the people descended upon him.\nDeutsch:Wir begehren von euch weder Lohn noch Dank dafür.\nEnglish:We wish for no reward, nor thanks from you.""",
            "law": """\nDeutsch:Deshalb ist die Regelung von der Ausfuhrleistung abhängig.\nEnglish:In this regard, the scheme is contingent upon export performance.\nDeutsch:Das Mitglied setzt gleichzeitig den Rat von seinem Beschluß in Kenntnis.\nEnglish:That member shall simultaneously inform the Council of the action it has taken.\nDeutsch:Dies gilt auch für die vorgeschlagene Sicherheitsleistung.\nEnglish:The same shall apply as regards the security proposed.""",
            "medical": """\nDeutsch:Das Virus wurde zuerst inaktiviert (abgetötet), damit es keine Erkrankungen verursachen kann.\nEnglish:This may help to protect against the disease caused by the virus.\nDeutsch:Desirudin ist ein rekombinantes DNS-Produkt, das aus Hefezellen hergestellt wird.\nEnglish:Desirudin is a recombinant DNA product derived from yeast cells.\nDeutsch:Katzen erhalten eine intramuskuläre Injektion.\nEnglish:In cats, it is given by intramuscular injection.""",
        }
        postfix = "\nDeutsch:{de}\nEnglish:{en}"
        prompt = prefix + shot[domain] + postfix
        # BUG 在我们的实验里，embedding本身什么是什么模型不太重要，因为我们会在外面安排额外的前向
        # 所以最后只利用里面db的直接搜索，不会传入一个字符串进去。所以这里随便设置一个就好了。
        embeddings = self.embeddings(model=self.model,prompt=prompt,vdb_type=vdb_type)
        self.retriever = knnretriver(index_dir=self.dstore_dir, embeddings=embeddings)

    def update_encoder_input(self, input_for_seq2seq_model):
        # seq2seq模型的encoder输入只能这么获取了，叹气
        self.input_ids = input_for_seq2seq_model.input_ids[:, :-1]
        beam_size = self.model.config.num_beams
        self.input_ids = torch.repeat_interleave(self.input_ids, beam_size, dim=0)

        # 其实这个是需要的，不然没法去掉pad部分
        self.attention_mask = input_for_seq2seq_model.attention_mask[:, :-1]
        self.attention_mask = torch.repeat_interleave(
            self.attention_mask, beam_size, dim=0
        )
        # print(beam_size,input_for_seq2seq_model,self.input_ids,self.attention_mask)
        self.assistant_pastkv = None

    def break_into(self, model, assistant_model=None,domain=None,vdb_type=None):
        self.model = model
        self.assistant_model = assistant_model  
        model.broken_into = True
        self.setup_faiss(domain=domain,vdb_type=vdb_type)
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # NOTE 挂在整个模型最前面的钩子，用于捕获labels，
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        
        if self.assistant_model is None:
            # NOTE 其实这个地方如果只捕获last ffn output是可以和下面的钩子融合的，只是考虑到这里还能钩最后一个注意力层和ffn中间的那个激活值就分开了
            # 被获取的值放在 self.activation_capturer.captured里头
            layer_to_capture_fn, capture_input,reverse = KNNWrapper.model_layer_to_capture[
                model.config.model_type
            ][self.knn_keytype]
            layer_to_capture = layer_to_capture_fn(model)
            self.activation_capturer = ActivationCapturer(
                layer_to_capture, capture_input=capture_input,reverse=reverse
            )
            self.register_hook(layer_to_capture, self.activation_capturer)
        # 如果是assistant模式的话,这个钩子应该挂在assistant_model上
        else:
            layer_to_capture_fn, capture_input,reverse = KNNWrapper.model_layer_to_capture[
                assistant_model.config.model_type
            ][self.knn_keytype]
            layer_to_capture = layer_to_capture_fn(assistant_model)
            self.activation_capturer = ActivationCapturer(
                layer_to_capture, capture_input=capture_input,reverse=reverse
            )
            self.register_hook(layer_to_capture, self.activation_capturer)

        # NOTE 一般情况下获取的是lm_head，如果不是的话需要自己改写这个函数
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        # 这个后钩子也要魔改,这块主要是搜knn和插值
        self.register_hook(final_layer, self.post_forward_hook)
        self.vocab_size = final_layer.out_features

    def get_knns(self, queries, queries_is_str):
        start = time.time()
        # 如果query是str，就需要给检索器里的embedding编码，否则可以直接调db的检索。
        if queries_is_str:
            result = self.retriever.retrieve(queries,k=self.k)
        else:
            result = self.retriever.db.similarity_search_with_score_by_vector(
                queries.to(torch.float).to("cpu"), k=self.k
            )
        # result是bsz x k x (2) ->0是document ,1是分数
        # print('result',result[0][0])
        dists, knns = [], []
        for i in range(len(result)):
            tempdists, tempknns = [], []
            for j in range(len(result[i])):
                tempdists.append(result[i][j][1])
                tempknns.append(result[i][j][0]["page_content"])
            dists.append(tempdists)
            knns.append(tempknns)

        end = time.time()
        # print('dists',dists)
        # print('检索耗时',end-start)
        return torch.tensor(dists), torch.tensor(knns)

    def pre_forward_hook(
        self, input_ids=None, attention_mask=None, labels=None, **kwargs
    ):
        self.labels = labels
        self.decoder_input_ids = kwargs.get("decoder_input_ids", None)
        # print('self.decoder_input_ids',self.decoder_input_ids)
        if self.assistant_model is not None:
            # nmt模型走generate的时候拿不到input_ids,因为会直接过encoder前处理成encoder_output
            self.decoder_input_ids = kwargs.get("decoder_input_ids", None)

        else:
            pass
            # print("self.assistant_model is None", input_ids)

        return self.original_forward_func(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs
        )

    def preprare_input_for_assistant_model(
        self,
    ):
        # print('self.decoder_input_ids',self.decoder_input_ids)
        # decoder的输入不要第一个，因为那个是eos
        real_input = torch.cat((self.input_ids, self.decoder_input_ids[:, 1:]), dim=1)
        decoder_attn = torch.ones_like(self.decoder_input_ids[:, 1:])
        real_attn = torch.cat((self.attention_mask, decoder_attn), dim=1)
        if self.assistant_pastkv is None:
            return real_input, real_attn
        else:
            return real_input[:, -1:], real_attn

    def post_forward_hook(self, module, input, output):
        # print('input',input)
        start = time.time()
        batch, time_dim, vocab_size = output.shape
        shift = 0 if self.is_encoder_decoder else 1
        lm_logits = output
        lm_logits = torch.nn.functional.log_softmax(
            lm_logits, dim=-1
        )  # (batch, time, vocab)

        # 这里需要安排一次辅助模型的前向
        if self.assistant_model is not None:

            input_ids, attention_mask = self.preprare_input_for_assistant_model()
            from transformers import AutoTokenizer

            llama_tokenizer = AutoTokenizer.from_pretrained(
                config.llama_path[self.vdb_type],
                use_fast=True,
                padding_side="left",
            )
            llama_tokenizer.pad_token_id = 0
            # print('输入给llama的东西',llama_tokenizer.batch_decode(input_ids),)
            # if self.assistant_pastkv is not None:
            # print('self.assistant_pastkv.shape',len(self.assistant_pastkv),self.assistant_pastkv[0][0].shape)
            assistant_output = self.assistant_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=self.assistant_pastkv,
                return_dict=True,
            )
            # print('logits',assistant_output['logits'])
            self.assistant_pastkv = assistant_output["past_key_values"]
        # 因为我已经设置了assistant模式挂在辅助模型上了,所以这里应该不需要修改
        queries = self.activation_capturer.captured  # (batch, time, dim)

        if self.labels is None:
            # 应该就是一个bsz\seqlen形状的东西
            nonpad_mask = torch.cat(
                [
                    torch.zeros([batch, time_dim - 1], dtype=torch.bool),
                    torch.ones([batch, 1], dtype=torch.bool),
                ],
                axis=-1,
            ).to(self.device)
        else:
            nonpad_mask = torch.cat(
                [
                    self.labels[:, shift:] != -100,
                    torch.zeros([self.labels.shape[0], shift], dtype=torch.bool).to(
                        self.device
                    ),
                ],
                axis=-1,
            )

        # NOTE 我认为当labels不存在的时候,这个地方实际上是取了时间维度的最后一个,相当于lm_logits[:,-1,:]
        # 验证了确实一样
        # print('debug',lm_logits.shape,output.shape,nonpad_mask.shape)
        lm_logits = lm_logits[nonpad_mask]
        # print('debug',lm_logits)

        if self.assistant_model is None:
            queries = queries[nonpad_mask]  # (nonpad, dim)
        else:
            queries = queries[:, -1, :]

        # print(queries)
        if self.assistant_model is None:
            dists, knns = self.get_knns(
                queries, queries_is_str=False
            )  # (nonpad batch * time, k)
        else:
            # 主要我把辅助模型的前向安排在这个外面了
            dists, knns = self.get_knns(
                queries, queries_is_str=False
            )
        # print(dists,knns)
        neg_dists = -dists
        knn_log_probs, _ = self.knns_to_log_prob(knns, neg_dists)
        # print('lm_logits',torch.topk(lm_logits,k=5))
        interpolated_scores = KNNWrapper.interpolate(
            knn_log_probs, lm_logits, self.lmbda
        )  # (nonpad, vocab)
        # output是bf16
        # print(output,interpolated_scores)
        # 因为有可能output来自nmt，那就是float，来自llama就是bf，所以直接根据output转变就行
        output[nonpad_mask] = interpolated_scores.to(output.dtype)
        
        # debug------------------------------------------------------------
        torch.set_printoptions(edgeitems =7)
        # print('topk output',torch.topk(output,k=5,))
        # nonzero_indices=torch.nonzero(knn_log_probs)
        # print('torch.nonzero',nonzero_indices)
        # print(nonzero_indices.shape)
        # print('non zero value',knn_log_probs[nonzero_indices[:, 0], nonzero_indices[:, 1]])
        # print('output.shape',output.shape)
        end = time.time()
        # print('后钩子耗时',end-start)

        return output

    def knns_to_log_prob(self, knns, neg_dists):
        # NOTE BUG?
        probs = torch.nn.functional.softmax(neg_dists / self.knn_temperature, dim=-1)
        vals_at_knns = knns.squeeze(-1)  # (nonpad batch * time, k)
        # print('neg_dists',neg_dists)
        # print(probs,vals_at_knns,vals_at_knns.shape,vals_at_knns.shape[:-1])
        # print( vals_at_knns.device,probs.device)
        knn_log_probs = (
            torch.full(
                size=(vals_at_knns.shape[:-1] + (self.vocab_size,)), fill_value=0.0
            )
            .scatter_add(dim=-1, index=vals_at_knns, src=probs)
            .to(self.device)
            .log()
        )  # (nonpad_batch * time, vocab)
        knn_log_probs = torch.nan_to_num(knn_log_probs, nan=None, neginf=-10000.0)
        # print('knn_log_probs',torch.topk(knn_log_probs,k=5))
        return knn_log_probs, vals_at_knns

    def register_hook(self, layer, func, pre=False):
        handle = (
            layer.register_forward_pre_hook(func)
            if pre
            else layer.register_forward_hook(func)
        )
        self.hook_handles.append(handle)

    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None

    def get_metrics(self):
        return {}

    @staticmethod
    def l2(query, keys):
        # query: (batch*time, dim)
        # keys:  (batch*time, k, dim)
        # returns: (batch*time, k)
        return torch.sum((query.unsqueeze(-2) - keys) ** 2, dim=-1)

    @staticmethod
    def dotprod(query, keys):
        # query: (batch, beams, dim)
        # keys:  (batch, 1, time, dim)
        # returns: (batch, beams, time)
        return torch.sum((query.unsqueeze(-2) * keys), dim=-1)

    @staticmethod
    def interpolate(knn_log_probs, lm_log_probs, lmbda):
        interpolated = torch.logaddexp(
            lm_log_probs + np.log(1 - lmbda), knn_log_probs + np.log(lmbda)
        )

        return interpolated

    @staticmethod
    def get_model_last_layer(model_type):
        # works for gpt2, marian, t5. If a model does not have an ".lm_head" layer,
        # add an "if model_type is ..." statement here, and return the output embedding layer
        if model_type == "fsmt":
            return lambda model: model.base_model.decoder.output_projection
        return lambda model: model.lm_head

    @staticmethod
    def get_model_embedding_layer(model_type):
        if model_type.startswith("gpt2"):
            return lambda model: model.transformer.wte

    # For every model name and key type, returns a lambda that returns the relevant layer in the model,
    # and whether the input of that layer should be captured (True) or the output (False)
    model_layer_to_capture = {
        "bart": {
            KEY_TYPE.last_ffn_input: (
                lambda model: model.base_model.decoder.layers[-1].fc1,
                True,
            ),
            KEY_TYPE.last_ffn_output: (
                lambda model: model.base_model.decoder.layers[-1],
                False,
            ),
        },
        "gpt2": {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.h[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.h[-1], False),
        },
        "marian": {
            KEY_TYPE.last_ffn_input: (
                lambda model: model.base_model.decoder.layers[-1].fc1,
                True,
            ),
            KEY_TYPE.last_ffn_output: (
                lambda model: model.base_model.decoder.layers[-1],
                False,
            ),
        },
        "t5": {
            KEY_TYPE.last_ffn_input: (
                lambda model: model.base_model.decoder.block[-1]
                .layer[2]
                .DenseReluDense,
                True,
            ),
            KEY_TYPE.last_ffn_output: (
                lambda model: model.base_model.decoder.block[-1].layer[2],
                False,
            ),
        },
        "llama": {
            KEY_TYPE.last_ffn_input: (
                lambda model: model.base_model.layers[-1].mlp,
                True,
                False,
            ),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.norm, False,False,),
        },
        "fsmt": {
            KEY_TYPE.last_ffn_input: (
                lambda model: model.base_model.decoder.layers[-1].fc1,
                True,
            ),
            KEY_TYPE.last_ffn_output: (
                lambda model: model.base_model.decoder.layers[-1].final_layer_norm,
                False,
                True,
            )
        },
        
    }


class ActivationCapturer(nn.Module):
    def __init__(self, layer, capture_input=False,reverse=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input
        self.reverse=reverse
        self.captured = None

    def forward(self, module, input, output):
        start = time.time()
        if self.capture_input:
            self.captured = input[0].detach()
        else:
            # 还挺麻烦的,一方面,模型的output本身是个tuple(tensor)
            # 里面的tensor可能因为束搜索变成很多个,like  bbsz x seq x 4096
            if  isinstance(output,tuple): # NOTE 这里如果不挂在layer[-1]上,挂在那种layernorm上就不会是tuple,但为了泛化就先这么写吧
                # print(len(output)) # fsmt这种应该是四个.x,self_attn_weights,layer_state(kv cache+k的padding mask),cross_attn_weights,
                self.captured = output[0].detach() # 修改一下改成层输出就好
                # print(output[0].shape) 
            else:
                self.captured = output.detach()
            
            # 我想啸,这个地方如果是fsmt,他是 seqlen x bbsz 和外面有冲突.
            if self.reverse:
                self.captured=self.captured.transpose(0, 1) 
        end = time.time()
        # print('获取隐藏状态耗时',end-start)


def get_dstore_path(dstore_dir, model_type, dstore_size, dimension):
    return f"{dstore_dir}/dstore_{model_type}_{dstore_size}_{dimension}"


def get_index_path(dstore_dir, model_type, dstore_size, dimension):
    return f"{dstore_dir}/index_{model_type}_{dstore_size}_{dimension}.indexed"
