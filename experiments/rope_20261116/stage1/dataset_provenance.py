import json,hashlib
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
p=Path(__file__).parent;x=json.load(open(p/'independent_model.json'));d=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test');text='\n\n'.join(d['text']);tok=AutoTokenizer.from_pretrained(x['model'],revision=x['revision']);ids=tok(text)['input_ids'];n=len(ids)//512
out={'dataset':'Salesforce/wikitext','configuration':'wikitext-2-raw-v1','split':'test','dataset_fingerprint':d._fingerprint,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'tokenizer_revision':x['revision'],'token_count':len(ids),'block_length':512,'complete_blocks':n,'predicted_tokens':n*511,'discarded_tail_tokens':len(ids)%512,'document_join':'two newlines','context_reset_each_block':True,'calibration_text_used_by_candidate':False};assert n==563;json.dump(out,open(p/'dataset_provenance.json','w'),indent=2);print(out)
