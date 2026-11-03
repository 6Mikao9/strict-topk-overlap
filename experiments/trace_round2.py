"""Native-GQA long-context dense-model QKV replay, with critical-path controls."""
import argparse
import gc
import json
import importlib.util
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3 as modeling
from strict_topk import gqa
from benchmarks.critical_path import run_case
from benchmarks.run_benchmark import environment
from experiments.trace_replay import PROMPT


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--model-label',default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--contexts',type=int,nargs='+',default=[2048,16384,32760])
    ap.add_argument('--steps',type=int,default=6)
    ap.add_argument('--topk',type=int,nargs='+',default=[32,64])
    ap.add_argument('--replays',type=int,default=100)
    ap.add_argument('--batches',type=int,default=5)
    ap.add_argument('--output',default='results/round2/model_trace.json')
    args=ap.parse_args()
    torch.manual_seed(20261103)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
    assert max(args.contexts)+args.steps <= model.config.max_position_embeddings
    layers=[0,model.config.num_hidden_layers//2,model.config.num_hidden_layers-1]
    captured,counter={},[0]
    original=modeling.apply_rotary_pos_emb
    def hook(*a,**kw):
        q,k=original(*a,**kw)
        layer=counter[0]%model.config.num_hidden_layers
        if layer in layers:captured[layer]=q[0,:,-1].detach().contiguous()
        counter[0]+=1
        return q,k
    modeling.apply_rotary_pos_emb=hook
    env=environment()
    env['timing']='Batched CUDA Graph replays; quantiles of batch means; single-replay diagnostic also recorded'
    env['selector']='Native shared GQA min/max metadata, fused scores plus torch.topk'
    env['optional_libraries']={x:bool(importlib.util.find_spec(x)) for x in ['flashinfer','flash_attn']}
    env['limitations']=['Dense Qwen generation remains unchanged; no sparse inference task-quality measurement',
        'Repeated authored prompt; three sampled layers; only complete 32-token KV blocks',
        'KV/metadata preparation and previous-step ID construction excluded from measured decode path',
        'Graph inputs fixed per snapshot; stage-only subtraction cannot decompose concurrent execution']
    result={'environment':env,'model':args.model_label,'model_max_positions':model.config.max_position_embeddings,
            'scope':'native GQA long-context dense QKV replay, unchanged dense generation','prompt':PROMPT,
            'contexts':args.contexts,'steps':args.steps,'layers':layers,'experiments':[]}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    def save():
        result['passed']=all(c['passed'] for c in result['experiments'])
        path.write_text(json.dumps(result,indent=2)+'\n')
    try:
        with torch.inference_mode():
            for context in args.contexts:
                previous={}
                ids=tokenizer(PROMPT*(context//30+1),return_tensors='pt').input_ids[:,:context].cuda()
                assert ids.shape[1]==context
                past=None
                # Chunked dense prefill bounds temporary activations and logits.
                for start in range(0,context,1024):
                    counter[0]=0
                    out=model(ids[:,start:start+1024],past_key_values=past,use_cache=True,logits_to_keep=1)
                    past=out.past_key_values
                token=out.logits[:,-1].argmax(-1,keepdim=True)
                print(json.dumps({'prefill_done':context}),flush=True)
                for step in range(args.steps):
                    counter[0]=0
                    out=model(token,past_key_values=past,use_cache=True,logits_to_keep=1)
                    past=out.past_key_values
                    token=out.logits[:,-1].argmax(-1,keepdim=True)
                    for layer in layers:
                        q=captured[layer]
                        ck,cv=past.layers[layer].keys[0],past.layers[layer].values[0]
                        n=ck.shape[1]//32
                        k=ck[:,:n*32].reshape(ck.shape[0],n,32,q.shape[1]).contiguous()
                        v=cv[:,:n*32].reshape_as(k).contiguous()
                        lo,hi=k.amin(2).float(),k.amax(2).float()
                        for budget in args.topk:
                            if budget>n:continue
                            def selector():return gqa.select_blocks(q,lo,hi,budget)
                            selected=selector()
                            key=(layer,budget)
                            if key in previous:
                                meta={'model':args.model_label,'context':context,'step':step,'layer':layer,'prediction':'previous-step selected IDs'}
                                case=run_case(q,k,v,previous[key],selector,meta,replays=args.replays,batches=args.batches)
                                result['experiments'].append(case)
                            previous[key]=selected.clone()
                    save()
                    print(json.dumps({'context':context,'step_done':step,'cases':len(result['experiments']),'passed':result['passed']}),flush=True)
                del out,past,ids,token,k,v,lo,hi,q
                previous.clear();captured.clear();gc.collect();torch.cuda.empty_cache()
    finally:modeling.apply_rotary_pos_emb=original
    save()
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
