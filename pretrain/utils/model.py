import torch
from torch import nn, tensor
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration, AutoTokenizer
import copy
from fastNLP import seq_len_to_mask


class CoNTGenerator(nn.Module):

    def __init__(self, PTM, model_name, pad_id, args):
        super(CoNTGenerator, self).__init__()
        self.PTM = PTM
        if self.PTM == "pegasus":
            self.generator = PegasusForConditionalGeneration.from_pretrained(model_name)
        
        elif self.PTM == "t5" or self.PTM == "codet5":
            print(model_name)
            # self.generator = T5ForConditionalGeneration.from_pretrained('/media/HD0/CoNT/t5-robust')
            self.generator = T5ForConditionalGeneration.from_pretrained(model_name)

            if args.scratch:
                print("random initialize...")
                self.generator = T5ForConditionalGeneration(self.generator.config)
        else:
            raise NotImplementedError("not support this PTM yet")
        self.pad_id = pad_id
        self.hidden_size = self.generator.config.hidden_size
        self.vocab_size = self.generator.config.vocab_size
        self.linear_layer = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.xavier_uniform_(self.linear_layer.weight)
        self.ignore_index = args.ignore_index
        self.loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)
        self.args = args

    def form_ngram(self, input_tensor, n=2):
        """
        input_tensor: batch x sample_num x seq_len
        return: batch x seq_len-3 x 4
        """
        bsz, cand_num, seq_len = input_tensor.size(0), input_tensor.size(1), input_tensor.size(2)
        seq_len_clip = seq_len - n + 1
        input_tensor_repeated = input_tensor[:, :, None, :].repeat(1, 1, seq_len_clip, 1)
        help_matrix_1 = torch.triu(torch.ones(seq_len, seq_len))
        help_matrix_2 = torch.triu(torch.ones(seq_len, seq_len), diagonal=n)
        help_matrix = (help_matrix_1 - help_matrix_2)[:seq_len_clip].bool()[None, None, :, :]
        ret_tensor = torch.masked_select(input_tensor_repeated, help_matrix.to(input_tensor.device))
        return ret_tensor.view(bsz, cand_num, seq_len_clip, n)

    def torch_bleu(self, ref_tensor, sys_tensor, pad_id, n_gram=2):
        """
        Calculates n-gram precision with brevity penalty. contributed by jinulee-v

        ref_tensor: batch x seq_len1
        sys_tensor: batch x sample_num x seq_len2
        """
        # Determine batch size, sample count(=beam size), n-gram
        bsz, sample_num = sys_tensor.size(0), sys_tensor.size(1)
        n = min(min(n_gram, ref_tensor.size(-1)), sys_tensor.size(-1))

        # Generate masks
        ref_padding = (~(ref_tensor == pad_id)).float()
        ref_padding[:, 0] = 1
        ref_ngram_mask = torch.arange(0, ref_padding.size(1), device=ref_padding.device) * torch.ones_like(ref_padding)
        ref_ngram_mask = torch.where(
            ref_ngram_mask < (torch.sum(ref_padding, dim=-1, keepdims=True) - n + 1),
            ref_padding, torch.zeros_like(ref_padding)
        )[:, :ref_ngram_mask.size(-1) - n + 1]
        sys_padding = (~(sys_tensor == pad_id)).float()
        sys_padding[:, 0] = 1
        sys_ngram_mask = torch.arange(0, sys_padding.size(-1), device=sys_padding.device) * torch.ones_like(sys_padding)
        sys_ngram_mask = torch.where(
            sys_ngram_mask < (torch.sum(sys_padding, dim=-1, keepdims=True) - n + 1),
            sys_padding, torch.zeros_like(sys_padding)
        )[:, :, :sys_ngram_mask.size(-1) - n + 1]

        # Get n-grams
        ref_tensor = ref_tensor * ref_padding  # mask out paddings
        sys_tensor = sys_tensor * sys_padding
        ref_tensor = ref_tensor[:, None, :].repeat(1, sample_num, 1)  # readjust ref size to match sys
        input_tensor1_ngram = self.form_ngram(ref_tensor, n).float()
        input_tensor2_ngram = self.form_ngram(sys_tensor, n).float()  # batch x sample_num x seq_len-(n-1) x n

        # sim_matrix = torch.matmul(sys_tensor, ref_tensor.transpose(1, 2)).norm(dim=2, keepdim=True)
        # return sim_matrix

        # Calculate similarity matrix
        sim_matrix = (torch.norm(  # Calculate L2 norm to find if N-gram in `sys`` is present in `ref``
            input_tensor2_ngram.unsqueeze(3) - input_tensor1_ngram.unsqueeze(2),
            p=2, dim=-1
        ) == 0.0).to(torch.float)
        # print(sim_matrix.size(), sys_ngram_mask.size(), ref_ngram_mask.size())
        sim_matrix *= sys_ngram_mask.unsqueeze(3) * ref_ngram_mask.unsqueeze(1).unsqueeze(2)
        sim_matrix = torch.sum(torch.max(sim_matrix, dim=-1).values, dim=-1)

        # Brevity penalty 短句+长句惩罚因子
        ref_len = torch.sum(ref_padding, dim=-1, keepdims=True)
        sys_len = torch.sum(sys_padding, dim=-1) + 1

        bp = torch.exp(1 - (ref_len / sys_len))
        bp = torch.where(ref_len >= sys_len, bp, torch.ones_like(bp))


        # bp_exp = torch.exp(1 - (ref_len / sys_len))
        # bp_reciprocal = torch.reciprocal(bp_exp)
        # # bp = torch.where(ref_len >= sys_len, bp, torch.ones_like(bp))
        # bp_exp = torch.where(ref_len >= sys_len, bp_exp, torch.ones_like(bp_exp))
        # bp = torch.where(ref_len < sys_len, bp_reciprocal**2, torch.ones_like(bp_exp))
        # temp = sim_matrix / torch.sum(sys_ngram_mask, dim=-1) * bp



        return sim_matrix / torch.sum(sys_ngram_mask, dim=-1) * bp # batch x sample_num


    def affine_transformation(self, input_features, padding_mask, axis=1):
        length = torch.sum(padding_mask, axis=1) - 1
        padding_mask = seq_len_to_mask(length, max_len=padding_mask.shape[-1])
        trans_tmp = F.relu(self.linear_layer(input_features))  # batch
        trans_tmp = trans_tmp * padding_mask.unsqueeze(-1).float()
        trans_emb = torch.sum(trans_tmp, axis=axis)
        return trans_emb * (1 / length.unsqueeze(-1))

    @torch.no_grad()
    def sample_from_model(self, src_inp, src_pad_mask):
        batch_size = src_inp.size(0)
        candidate_id = self.generator.generate(
            input_ids=src_inp,
            attention_mask=src_pad_mask,
            num_return_sequences=self.args.beam_size,
            num_beam_groups=self.args.beam_size,
            diversity_penalty=self.args.diversity_pen,
            num_beams=self.args.beam_size,
            max_length=self.args.max_length + 2,
            min_length=self.args.min_length + 1,  # +1 from or
            # no_repeat_ngram_size=self.args.no_repeat_ngram,
            length_penalty=self.args.length_pen,
            early_stopping=self.args.early_stop,
        )

        tokenizer = AutoTokenizer.from_pretrained("/media/HD0/CoNT/t5-robust")
        print(tokenizer.batch_decode(candidate_id, skip_special_tokens=True))

        return candidate_id.view(batch_size, self.args.beam_size, -1)

    def pad2max_len(self, input_tensor, max_len):
        pad_size = max_len - input_tensor.shape[-1]
        pad_tensor = torch.full([input_tensor.shape[0], input_tensor.shape[1], pad_size], self.pad_id,
                                device=input_tensor.device).long()
        return torch.cat([input_tensor, pad_tensor], dim=-1)

    def padin2max_len(self, input_tensor, max_len):
        pad_size = max_len - input_tensor.shape[-1]
        pad_tensor = torch.full([input_tensor.shape[0], pad_size], self.pad_id,
                                device=input_tensor.device).long()
        return torch.cat([input_tensor, pad_tensor], dim=-1)

    def ranking_loss(self, cos_distance, bleu_distance):
        # equivalent to initializing TotalLoss to 0
        # here is to avoid that some special samples will not go into the following for loop
        margin = 0.01
        ones = torch.ones(cos_distance.size(), device=cos_distance.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        total_loss = loss_func(cos_distance, cos_distance, ones)

        # candidate loss
        n = cos_distance.size(1)
        for i in range(1, n):
            pos_score = cos_distance[:, :-i]
            neg_score = cos_distance[:, i:]
            same_mask = (torch.abs(bleu_distance[:, :-i] - bleu_distance[:, i:]) > margin).float()
            ones = torch.ones(pos_score.size(), device=cos_distance.device)
            loss_func = torch.nn.MarginRankingLoss(margin * i, reduction='none')  # batch x i
            marginal_loss = loss_func(pos_score, neg_score, ones)
            if same_mask.sum() > 0:
                total_loss += (marginal_loss * same_mask).sum() / same_mask.sum()

        return total_loss

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, args):
        # self.generator = T5ForConditionalGeneration.from_pretrained('/root/autodl-tmp/CoNT/t5-robust')
        # self.generator.to('cuda')
        self.generator.eval()
        ret_dict = self.generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=args.beam_size,
            num_beams=args.beam_size,
            max_length=args.max_length + 2,
            # +2 from original because we start at step=1 and stop before max_length
            min_length=args.min_length + 1,  # +1 from original because we start at step=1
            length_penalty=2.0,
            early_stopping=args.early_stop,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
        cand_ids = ret_dict["sequences"]
        cand_mask = (cand_ids != self.pad_id).long()

        if args.alpha == 0.0:
            cand_ids = cand_ids.view(input_ids.size(0), args.beam_size, -1)
            return cand_ids[:, 0, :]
        cand_len = torch.sum(cand_mask, dim=-1)
        max_len = torch.max(cand_len).item()
        cand_ids = cand_ids[:, :max_len]
        beam_indices = ret_dict['beam_indices']
        beam_indices = torch.where(beam_indices > 0, beam_indices, 0)
        decoder_hidden_states = ret_dict["decoder_hidden_states"]
        # get the hidden_states from the last layer of decoder
        hidden_states_from_output = torch.cat(
            [decoder_hidden_states[i][-1] for i in range(len(decoder_hidden_states))],
            dim=1)  # batch*beam x seq_len x h
        h = hidden_states_from_output.shape[-1]
        decoder_hidden_states = torch.gather(hidden_states_from_output, 0,
                                             beam_indices[:, :-1].unsqueeze(-1).repeat(1, 1, h))

        encoder_hidden_states = ret_dict["encoder_hidden_states"][-1]  # batch x src_len x h
        encoder_feature = self.affine_transformation(encoder_hidden_states, attention_mask)  # batch x h
        decoder_feature = self.affine_transformation(decoder_hidden_states, cand_mask[:, :-1])
        decoder_feature = decoder_feature.view(input_ids.size(0), args.beam_size, -1)  # batch x sample_num x h
        cos_distance = torch.cosine_similarity(encoder_feature.unsqueeze(1), decoder_feature,
                                               dim=-1)  # batch x sample_num
        scores = ret_dict["sequences_scores"].view(input_ids.size(0), -1)
        normalize = torch.sum(0 - scores, keepdim=True, dim=-1)
        score = (1 - args.alpha) * (scores / normalize) + args.alpha * cos_distance
        cand_ids = cand_ids.view(input_ids.size(0), args.beam_size, -1)
        max_indices = torch.argmax(score, dim=-1)[:, None, None]
        dummy = max_indices.repeat(1, 1, cand_ids.size(2))
        return torch.gather(cand_ids, 1, dummy).squeeze(1)  # batch x seq_len

    def evaluate_step(self, src_inp, target_inp, target_outp):
        src_pad_mask = (src_inp != self.pad_id)
        args = copy.deepcopy(self.args)
        if self.args.warmup:
            args.alpha = 0.0
        else:
            args.alpha = self.args.alpha
        args.diversity_pen = 0.0
        args.beam_size = 8
        args.early_stop = True
        candidate_ids = self.generate(src_inp, src_pad_mask, args)              # 无法输出
        return {"score": self.torch_bleu(target_inp, candidate_ids.unsqueeze(1), self.pad_id, 2).mean()}

    def build_mask_matrix(self, seqlen, valid_len_list, prefix_len=0):
        '''
            prefix_len: the length of prefix that we do not want to compute CL loss for.

            (1) if a sequence of length 4 contains zero padding token (i.e., the valid length is 4),
                then the loss padding matrix looks like
                     [0., 1., 1., 1.],
                     [1., 0., 1., 1.],
                     [1., 1., 0., 1.],
                     [1., 1., 1., 0.]

            (2) if a sequence of length 4 contains 1 padding token (i.e., the valid length is 3),
                then the loss padding matrix looks like
                     [0., 1., 1., 0.],
                     [1., 0., 1., 0.],
                     [1., 1., 0., 0.],
                     [0., 0., 0., 0.]
        '''
        res_list = []
        base_mask = torch.ones(seqlen, seqlen) - torch.eye(seqlen, seqlen)
        base_mask = base_mask.type(torch.FloatTensor)
        bsz = len(valid_len_list)
        for i in range(bsz):
            one_base_mask = base_mask.clone()
            one_valid_len = valid_len_list[i]
            one_base_mask[:, one_valid_len:] = 0.
            one_base_mask[one_valid_len:, :] = 0.
            if prefix_len > 0:
                one_base_mask[:prefix_len, :prefix_len] = 0.
            res_list.append(one_base_mask)
        res_mask = torch.stack(res_list, dim=0)  # torch.FloatTensor(res_list)
        # print (res_mask)
        assert res_mask.size() == torch.Size([bsz, seqlen, seqlen])
        return res_mask

    def contrastive_loss(self, score_matrix, input_ids, pad_token_id, prefix_len=0):
        '''
           score_matrix: bsz x seqlen x seqlen
           input_ids: bsz x seqlen
        '''
        bsz, seqlen, _ = score_matrix.size()
        gold_score = torch.diagonal(score_matrix, offset=0, dim1=1, dim2=2)  # bsz x seqlen
        gold_score = torch.unsqueeze(gold_score, -1)
        assert gold_score.size() == torch.Size([bsz, seqlen, 1])
        difference_matrix = gold_score - score_matrix
        assert difference_matrix.size() == torch.Size([bsz, seqlen, seqlen])

        self.margin = 0.5

        loss_matrix = self.margin - difference_matrix  # bsz x seqlen x seqlen
        loss_matrix = torch.nn.functional.relu(loss_matrix)

        ### input mask
        input_mask = torch.ones_like(input_ids).type(torch.FloatTensor)
        if loss_matrix.is_cuda:
            input_mask = input_mask.cuda(loss_matrix.get_device())
        input_mask = input_mask.masked_fill(input_ids.eq(self.pad_id), 0.0)

        if loss_matrix.is_cuda:
            input_mask = input_mask.cuda(loss_matrix.get_device())

        valid_len_list = torch.sum(input_mask, dim=-1).tolist()
        loss_mask = build_mask_matrix(seqlen, [int(item) for item in valid_len_list], prefix_len)
        if score_matrix.is_cuda:
            loss_mask = loss_mask.cuda(score_matrix.get_device())
        masked_loss_matrix = loss_matrix * loss_mask

        loss_matrix = torch.sum(masked_loss_matrix, dim=-1)
        loss_matrix = loss_matrix.unsqueeze(0)
        assert loss_matrix.size() == input_ids.size()
        loss_matrix = loss_matrix * input_mask
        cl_loss = torch.sum(loss_matrix) / torch.sum(loss_mask)
        return cl_loss


    def forward(self, src_inp, target_inp, target_outp):
        """
        cos_score distance of hypothesis to source
        """

        encoder = self.generator.get_encoder()
        decoder = self.generator.get_decoder()

        batch_size = src_inp.size(0)
        src_pad_mask = ~(src_inp == self.pad_id)

        encoder_hidden_states = encoder(src_inp, src_pad_mask)['last_hidden_state']
        if target_inp.size(1) < self.args.max_length:
            target_inp = self.padin2max_len(target_inp, self.args.max_length + 2)
        if target_outp.size(1) < self.args.max_length:
            target_outp = self.padin2max_len(target_outp, self.args.max_length + 2)
        tgt_pad_mask = ~(target_inp == self.pad_id)
        tgt_pad_mask[:, 0] = 1

        decoder_out = decoder(input_ids=target_inp, attention_mask=tgt_pad_mask,
                              encoder_hidden_states=encoder_hidden_states,
                              encoder_attention_mask=src_pad_mask)  # last layer

        decoder_last_layer = decoder_out[0] * (self.generator.model_dim ** -0.5)
        lm_logits = self.generator.lm_head(decoder_last_layer)
        nll_loss = self.loss_fct(lm_logits.view(-1, self.vocab_size), target_outp.view(-1))
        if self.args.warmup:
            return {'loss': nll_loss}

        ###### Contrastive loss ######
        cand_ids = self.sample_from_model(src_inp, src_pad_mask)  # batch x beam_size x seq_len
        # prepare contrastive learning
        samples_from_batch = target_inp[None, :, :].repeat(batch_size, 1, 1)
        cand_len = cand_ids.size(2)
        samples_len = samples_from_batch.size(2)
        if samples_len < cand_len:
            samples_from_batch = self.pad2max_len(samples_from_batch, cand_len)
        else:
            samples_from_batch = samples_from_batch[:, :, :cand_len]
        samples_all = torch.cat([cand_ids, samples_from_batch], dim=1)  # batch x total_sample_num x seq_len

        actual_distance = self.torch_bleu(target_inp, samples_all, self.pad_id, self.args.n_gram)  # batch x total_sample_num
        distance_mask = (actual_distance < 0.99)  # use to mask the gold
        actual_distance_masked = actual_distance * distance_mask.float()
        sample_num = min(self.args.max_sample_num - 1, actual_distance_masked.size(1) - 1)
        actual_distance, actual_indices = torch.sort(actual_distance_masked, dim=-1, descending=True)
        sampled_actual_distance = actual_distance[:, :sample_num]
        sampled_actual_indices = actual_indices[:, :sample_num]
        # concat itself
        self_indices = torch.arange(0, batch_size).reshape(batch_size, 1).to(
            sampled_actual_indices.device) + cand_ids.size(1)  # manually add gold
        sampled_indices = torch.cat([self_indices, sampled_actual_indices], dim=-1)

        self_distance = torch.full([batch_size, 1], 1.0, device=sampled_actual_distance.device)
        sampled_bleu_distance = torch.cat([self_distance, sampled_actual_distance], dim=-1)
        dummy = sampled_indices.unsqueeze(-1).repeat(1, 1, samples_all.size(2))
        sampled_input = torch.gather(samples_all, 1, dummy)  # batch x sample_num x seq_len

        decoder_hidden_states = []
        for sample_idx in range(sampled_indices.size(-1)):
            sampled_input_dec = sampled_input[:, sample_idx, :]

            sample_pad_mask = ~(sampled_input_dec == self.pad_id)
            sample_pad_mask[:, 0] = 1

            decoder_out = decoder(input_ids=sampled_input_dec, attention_mask=sample_pad_mask,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=src_pad_mask)  # last layer
            decoder_feature = decoder_out[0]  # batch x tgt_len x hidden
            decoder_feature = self.affine_transformation(decoder_feature, sample_pad_mask)  # batch x h
            decoder_hidden_states.append(decoder_feature.unsqueeze(1))

        encoder_feature = self.affine_transformation(encoder_hidden_states, src_pad_mask)  # batch x h
        decoder_feature = torch.cat(decoder_hidden_states, dim=1)  # batch x sample_num x h
        cos_distance = torch.cosine_similarity(encoder_feature.unsqueeze(1), decoder_feature,
                                               dim=-1)  # batch x samle_num
        cl_loss = self.ranking_loss(cos_distance, sampled_bleu_distance)
        #
        # norm_rep = decoder_feature / decoder_feature.norm(dim=2, keepdim=True)
        # cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1, 2))
        # cl_loss = self.contrastive_loss(cosine_scores, samples_all, self.pad_id, prefix_len=0)


        print('nll_loss', nll_loss)
        print('cl_loss', cl_loss)

        return {'loss': nll_loss + (1.0 * cl_loss), "cl_loss": cl_loss}
