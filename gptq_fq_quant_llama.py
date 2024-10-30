import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import quant

from ptq.layers import QLinear, QAct, QIntSoftmax, QIntLayerNorm
from gptq import GPTQ, Observer
from utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders, export_quant_table, gen_conditions
from texttable import Texttable


def get_llama(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = 2048
    return model


@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    observer = Observer()
    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')

        layer = layers[i].to(dev)
        full = find_layers(layer)
        if args.true_sequential:
            sequential = [['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'], ['self_attn.o_proj'], ['mlp.up_proj', 'mlp.gate_proj'], ['mlp.down_proj']]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name], observe=args.observe)
                gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

            def add_batch(name):

                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                temp, _, _, _ = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)
                outs[j] = temp[0]
            for h in handles:
                h.remove()

            for name in subset:
                scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

                if args.observe:
                    observer.submit(name=name, layerid=i, gptq=gptq[name], error=error)
                else:
                    gptq[name].free()

        for j in range(args.nsamples):
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            temp, _, _, _ = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)
            outs[j] = temp[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        print('+------------------+--------------+------------+-----------+-------+')
        print('\n')

    if args.observe:
        observer.print()
        conditions = gen_conditions(args.wbits, args.groupsize)
        for item in observer.items():
            name = item[0]
            layerid = item[1]
            gptq = item[2]['gptq']
            error = item[2]['error']
            target = error / 2

            table = Texttable()
            table.header(['wbits', 'groupsize', 'error'])
            table.set_cols_dtype(['i', 'i', 'f'])
            table.add_row([args.wbits, args.groupsize, error])

            print('Optimizing {} {} ..'.format(name, layerid))
            for wbits, groupsize in conditions:

                if error < target:
                    # if error dropped 50%, skip
                    break

                gptq.quantizer.configure(wbits, perchannel=True, sym=args.sym, mse=False)

                scale, zero, g_idx, error = gptq.fasterquant(percdamp=args.percdamp, groupsize=groupsize, actorder=args.act_order, name=name)

                table.add_row([wbits, groupsize, error])
                quantizers['model.layers.%d.%s' % (layerid, name)] = (gptq.quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), wbits, groupsize)

            print(table.draw())
            print('\n')
            gptq.layer.to('cpu')
            gptq.free()

    model.config.use_cache = use_cache

    return quantizers


def llama_multigpu(model, gpus, gpu_dist):
    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    if hasattr(model.model, 'norm') and model.model.norm:
        model.model.norm = model.model.norm.to(gpus[0])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[0])

    class MoveModule(nn.Module):

        def __init__(self, module, invalidate_cache):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
            self.invalidate_cache = invalidate_cache

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)

            for e in kwargs:
                if kwargs[e] is not None and hasattr(kwargs[e], "device"):
                    if kwargs[e].device != self.dev:
                        kwargs[e] = kwargs[e].to(self.dev)

            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.layers
    from math import ceil
    if not gpu_dist:
        pergpu = ceil(len(layers) / len(gpus))
        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(0 if i == 0 or i == len(layers) - 1 else gpus[(i - 1) // pergpu]),
                                   i == 0)
    else:
        assert gpu_dist[0] >= 2, "At least two layers must be on GPU 0."
        assigned_gpus = [0] * (gpu_dist[0] - 1)
        for i in range(1, len(gpu_dist)):
            assigned_gpus = assigned_gpus + [i] * gpu_dist[i]

        remaining_assignments = len(layers) - len(assigned_gpus) - 1
        if remaining_assignments > 0:
            assigned_gpus = assigned_gpus + [-1] * remaining_assignments

        assigned_gpus = assigned_gpus + [0]

        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(gpus[assigned_gpus[i]]), i == 0)

    model.gpus = gpus


# xuan
def quantize_act(model, train_loader, device):
    print('Calibrating ...')
    calib_iter = 10

    # Get calibration set.
    sample_list = []
    for i, (data, target) in enumerate(train_loader):
        if i == calib_iter:
            break
        data = data.to(device)
        sample_list.append(data)

    # open calibrate
    for m in model.modules():
        # if type(m) in [QLinear, QAct, QIntSoftmax]:
        if type(m) == QAct:
            m.calibrate = True

    with torch.no_grad():
        for i, sample in enumerate(sample_list):
            if i == len(sample_list) - 1:
                # This is used for OMSE method to
                # calculate minimum quantization error
                for m in model.modules():
                    # if type(m) in [QLinear, QAct, QIntSoftmax]:
                    if type(m) == QAct:
                        m.last_calibrate = True
            output = model(sample)
            del sample

    # close calibrate
    for m in model.modules():
        # if type(m) in [QLinear, QAct, QIntSoftmax]:
        if type(m) == QAct:
            m.calibrate = False

    # quant
    for m in model.modules():
        # if type(m) in [QLinear, QAct, QIntSoftmax]:
        if type(m) == QAct:
            m.quant = True
        # if model.config.INT_NORM:
        #     if type(m) in [QIntLayerNorm]:
        #         m.mode = 'int'

    # xuan: we need to remove the following tensors in gpu to release more space for the evaluation
    del data
    del sample_list
    model = model.cpu()
    return model


@torch.no_grad()
def llama_eval_new(model, testenc, dev):
    print('Evaluating ...')

    loss_fct = nn.CrossEntropyLoss()

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    nlls = []
    if model.config.token_entropy_computation_mode:
        entropy_samples_record = []
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)

        lm_logits, \
        visualization_act_model, layer_entropy_record, token_prune_idx_all, lower_bit_idx_record = model(batch)

        lower_bit_num_each_layer = []
        for e in lower_bit_idx_record:
            if e is not None:
                lower_bit_num_each_layer.append(e.shape[1])
            else:
                lower_bit_num_each_layer.append(0)

        if model.config.token_entropy_computation_mode:
            entropy_samples_record.append(layer_entropy_record)

        num_tokens_remaining_each_layer = [model.seqlen] * model.config.num_hidden_layers
        # if there is token remove, we need to remove same index in target label here
        if sum(model.config.token_remove_percent_total) > 0:
            shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
            shift_labels = shift_labels.view(-1)

            judge_if_last_token_was_pruned = False

            # getting the remaining tokens
            current_remaining_tokens = model.seqlen
            for idx, e in enumerate(model.config.token_remove_percent_total):
                if idx >= model.config.num_hidden_layers:
                    break
                if e > 0:
                    current_remaining_tokens = current_remaining_tokens - int(current_remaining_tokens * e)
                    num_tokens_remaining_each_layer[idx] = current_remaining_tokens
                else:
                    num_tokens_remaining_each_layer[idx] = current_remaining_tokens

            for i_th_pruned_layer, e in enumerate(token_prune_idx_all):
                if e is not None:
                    remove_idx_here = set(e.view(-1).detach().cpu().numpy())

                    if not judge_if_last_token_was_pruned and shift_labels.shape[0] in remove_idx_here:
                        judge_if_last_token_was_pruned = True

                    all_idx = set(np.arange(shift_labels.shape[0]))
                    remain_token_idx_current = torch.from_numpy(
                        np.array(list(all_idx - remove_idx_here))  # get difference set
                    ).to(shift_labels.device).view(-1)
                    shift_labels = shift_labels[remain_token_idx_current]

            if not judge_if_last_token_was_pruned:
                shift_logits = lm_logits[:, :-1, :].contiguous()
            else:
                # Xuan's important note:
                #   if the last token was pruned by our method, we will not remove it in the shift part
                shift_logits = lm_logits[:, :, :].contiguous()
        else:
            # if there is no token remove, we just go to the normal shift
            shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
            shift_labels = shift_labels.view(-1)
            shift_logits = lm_logits[:, :-1, :].contiguous()

        if i == 0:
            print("Token prune ratio: ", model.config.token_remove_percent_total[:model.config.num_hidden_layers])
            print("If inherit lower bit index from previous layer: ",
                  model.config.lower_bit_inherit_from_previous_layers)
            print("Lower Bit ratio: ", model.config.lower_bit_percent_total[:model.config.num_hidden_layers])
            total_remaining_token_num = 0
            total_lower_bit_token_num = 0
            total_normal_bit_token_num = 0
            for i_th_layer in range(model.config.num_hidden_layers):
                print(
                    "at {} layer, the number of [lower bit] / [remaining] / [original] tokens"
                    "is: [{}]/[{}]/[{}]".format(
                        i_th_layer + 1,
                        lower_bit_num_each_layer[i_th_layer],
                        num_tokens_remaining_each_layer[i_th_layer],
                        model.seqlen,
                    )
                )
                total_remaining_token_num += num_tokens_remaining_each_layer[i_th_layer]
                total_lower_bit_token_num += lower_bit_num_each_layer[i_th_layer]
                total_normal_bit_token_num += \
                    (num_tokens_remaining_each_layer[i_th_layer] - lower_bit_num_each_layer[i_th_layer])
            print("in the end, ")
            print("the total number of remaining tokens is [{}]/[{}]".format(
                total_remaining_token_num, model.seqlen * model.config.num_hidden_layers
            ))
            print("the total number of 4-bit tokens is [{}]/[{}]".format(
                total_lower_bit_token_num, total_remaining_token_num
            ))
            print("the total number of 8-bit tokens is [{}]/[{}]".format(
                total_normal_bit_token_num, total_remaining_token_num
            ))

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    if model.config.token_entropy_computation_mode:
        total_entropy = 0
        total_entropy_layerly = {}
        for e in entropy_samples_record:
            # get entropy for each sample
            for idx, f in enumerate(e):
                # get entropy for each layer in the model with one sample
                total_entropy += f
                if idx in total_entropy_layerly:
                    total_entropy_layerly[idx] += f
                else:
                    total_entropy_layerly[idx] = f
        print("Entropy of all tokens: {}".format(total_entropy))
        print("Entropy of tokens at different layers:")
        for e in total_entropy_layerly:
            print("The entropy at layer {} is {}".format(e+1, total_entropy_layerly[e]))

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print("Perplexity: {:.4f}".format(ppl.item()))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='llama model to load')
    parser.add_argument('--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--nearest', action='store_true', help='Whether to run the RTN baseline.')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
    parser.add_argument('--test-generation', action='store_true', help='test generation.')
    parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
    parser.add_argument('--save_safetensors', type=str, default='', help='Save quantized `.safetensors` checkpoint under this name.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
    parser.add_argument('--new-eval', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument('--layers-dist', type=str, default='', help='Distribution of layers across GPUs. e.g. 2:1:1 for 2 layers on GPU 0, 1 layer on GPU 1, and 1 layer on GPU 2. Any remaining layers will be assigned to your last GPU.')
    parser.add_argument('--observe',
                        action='store_true',
                        help='Auto upgrade layer precision to higher precision, for example int2 to int4, groupsize 128 to 64. \
            When this feature enabled, `--save` or `--save_safetensors` would be disable.')
    parser.add_argument('--quant-directory', type=str, default=None, help='Specify the directory for export quantization parameters to toml format. `None` means no export by default.')

    # xuan
    parser.add_argument('--card', type=int, default=0, help='cuda card')
    parser.add_argument('--max-response-length', type=int, help='response length')
    parser.add_argument('--chat', type=str, help='ask llama')
    parser.add_argument(
        '--save-quant-info', type=str, default="",
        help='save quant info'
    )
    parser.add_argument(
        '--load-quant-info', type=str, default="",
        help='save quant info'
    )

    args = parser.parse_args()

    if args.layers_dist:
        gpu_dist = [int(x) for x in args.layers_dist.split(':')]
    else:
        gpu_dist = []

    model = get_llama(args.model)
    model.eval()

    # first time, we quant the weights with GPTQ method and save it, then we just need to load it
    checkpoint_path = "checkpoints-quantized/{}.pth".format(args.model.replace("/", "-"))
    # ======== quant model weights with gptq method ========
    dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model,
                                         seqlen=model.seqlen)
    if not args.load and args.wbits < 16 and not args.nearest:
        quantizers = llama_sequential(model, dataloader, DEV)

    torch.save(model.state_dict(), checkpoint_path)
    print("Weight quantization finished, model saved at {}, please continue run activation part."
          .format(checkpoint_path))
    exit()
    # ======================================================
    if args.load:
        model.load_state_dict(torch.load(args.load))
    else:
        model.load_state_dict(torch.load(checkpoint_path))

    if args.eval:
        datasets = ['wikitext2', 'ptb', 'c4']
        if args.new_eval:
            datasets = ['wikitext2', 'ptb-new', 'c4-new']

        DEV = torch.device('cuda:{}'.format(args.card))

        for dataset in datasets:
            dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)

            print("Dataset: {}".format(dataset))

            # === quant act here ===================================
            try:
                model = model.to(DEV)
            except:
                gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
                if len(gpus) > 1:
                    llama_multigpu(model, gpus, gpu_dist)
                else:
                    model = model.to(DEV)

            if args.load_quant_info:
                quantization_info = torch.load(args.load_quant_info)
                quantization_info = quantization_info["quantization_info"]
                for idx, m in enumerate(model.modules()):
                    if type(m) == QAct:
                        try:
                            m.quantizer.scale = quantization_info[idx][0]
                            m.quantizer.zero_point = quantization_info[idx][1]
                        except:
                            m.quantizer.softmax_mask = quantization_info[idx]

                del quantization_info

                # open quant
                for m in model.modules():
                    if type(m) == QAct:
                        m.quant = True

            else:

                model = quantize_act(model, dataloader, DEV)  # xuan: calibrate here

                if args.save_quant_info:
                    print("saving quantization information...")
                    quantization_info = {}
                    for idx, m in enumerate(model.modules()):
                        if type(m) == QAct:
                            try:
                                quantization_info[idx] = [m.quantizer.scale, m.quantizer.zero_point]
                            except:
                                quantization_info[idx] = m.quantizer.softmax_mask

                    torch.save(
                        {"quantization_info": quantization_info},
                        "{}/quantization-info-{}-{}.pth".format(args.save_quant_info, args.model.replace("/", "-"),
                                                                dataset)
                    )

            model = model.cpu()
            torch.cuda.empty_cache()
            # ======================================================

            # === evaluation here===================================
            try:  # xuan: if the model is large, use the 'except' part directly   # mark
                model = model.to(DEV)
            except:
                gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
                if len(gpus) > 1:
                    llama_multigpu(model, gpus, gpu_dist)
                else:
                    model = model.to(DEV)

            llama_eval_new(model, testloader, DEV)

            torch.cuda.empty_cache()
            # ======================================================
    
    if args.test_generation:

        # === quant act here ===================================
        try:
            model = model.to(DEV)
        except:
            gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
            if len(gpus) > 1:
                llama_multigpu(model, gpus, gpu_dist)
            else:
                model = model.to(DEV)

        if args.load_quant_info:
            quantization_info = torch.load(args.load_quant_info)
            quantization_info = quantization_info["quantization_info"]
            for idx, m in enumerate(model.modules()):
                if type(m) == QAct:
                    try:
                        m.quantizer.scale = quantization_info[idx][0]
                        m.quantizer.zero_point = quantization_info[idx][1]
                    except:
                        m.quantizer.softmax_mask = quantization_info[idx]

            del quantization_info

            # open quant
            for m in model.modules():
                if type(m) == QAct:
                    m.quant = True

        else:
            # datasets = ['wikitext2', 'ptb', 'c4']
            dataset = ['wikitext2']
            dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)

            model = quantize_act(model, dataloader, DEV)  # xuan: calibrate here

            if args.save_quant_info:
                print("saving quantization information...")
                quantization_info = {}
                for idx, m in enumerate(model.modules()):
                    if type(m) == QAct:
                        try:
                            quantization_info[idx] = [m.quantizer.scale, m.quantizer.zero_point]
                        except:
                            quantization_info[idx] = m.quantizer.softmax_mask

                torch.save(
                    {"quantization_info": quantization_info},
                    "{}/quantization-info-{}-{}.pth".format(args.save_quant_info, args.model.replace("/", "-"),
                                                            dataset)
                )

        model = model.cpu()
        torch.cuda.empty_cache()
        # ======================================================

        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)

        model.config.demo_mode = True

        from transformers import LlamaTokenizer, TextStreamer
        tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)

        chat_input_str = args.chat
        input_ids = tokenizer([chat_input_str], return_tensors="pt").input_ids.to(gpus[0])

        streamer = TextStreamer(tokenizer)
        with torch.no_grad():
            generated_ids = model.generate(input_ids, streamer=streamer, max_length=args.max_response_length)

        torch.cuda.empty_cache()
