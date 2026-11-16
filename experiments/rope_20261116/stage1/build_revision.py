"""Update the existing review document using verified independent-model records."""
from pathlib import Path
import json,runpy
from docx import Document
from docx.shared import Inches,Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
p=Path(__file__).parent;root=p.parent
runpy.run_path(str(root/'build_docx.py'),run_name='__main__')
d=Document(root/'SKBQ_RoPE_Experimental_Revision.docx');r=json.load(open(p/'independent_summary.json'));s=json.load(open(p/'budget_summary.json'))
for para in d.paragraphs:
 if para.text=='Verified real model results':para.text='SmolLM2 results'
 if para.text.startswith('Technical replacement sections'):
  para.text='Technical replacement sections and verified experiments for independent review. The original SmolLM2 experiment is followed by an independent Pythia architecture and three-budget evaluation. The stronger joint-optimization control reproduces the proposed choices. Independent novelty and submission readiness are not established.'
 if para.text.startswith('The requested real-RoPE-model comparison'):
  para.text='The SmolLM2 comparison supports a measured gain over a basis frozen at 4/4. Subsequent stronger controls and independent-architecture results appear below and supersede any interpretation of that gain as independent novelty. The full published dynamic quantization pipeline has not been reproduced. Standard dynamic programming and RoPE invariance do not resolve the editor\'s contribution objection.'
def table(headers,rows,widths):
 t=d.add_table(rows=1,cols=len(headers));t.autofit=False
 for col,w in zip(t.columns,widths):col.width=Inches(w)
 for cell,text in zip(t.rows[0].cells,headers):cell.text=text
 for row in rows:
  for cell,text in zip(t.add_row().cells,row):cell.text=str(text)
 for i,row in enumerate(t.rows):
  for j,cell in enumerate(row.cells):
   pr=cell._tc.get_or_add_tcPr();bd=OxmlElement('w:tcBorders')
   for side in ['top','left','bottom','right']:
    e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');bd.append(e)
   pr.append(bd)
   for para in cell.paragraphs:
    para.paragraph_format.space_before=Pt(5);para.paragraph_format.space_after=Pt(5)
    if j:para.alignment=1
   if i==0:
    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E8EDF3');pr.append(sh)
    for run in cell.paragraphs[0].runs:run.bold=True
 repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
d.add_page_break();d.add_heading('Independent architecture evaluation',level=1)
d.add_paragraph('EleutherAI Pythia 70M uses GPT NeoX, fused QKV projections, biases, and partial RoPE. The model revision is a39f36b100fe8a5377810d56c3f4789b9c53ac42. The same angle grid, relative offsets and rowwise 4/8-bit RTN rule were used without test-based retuning. Q/K biases were rotated and retained in FP32. The score uses weights only and omits bias-dependent quantization error terms.')
d.add_paragraph('All 563 complete 512-token WikiText-2 test blocks were evaluated, scoring 287,693 tokens. Each of the three quantized methods uses an actual serialized Q/K payload of 2,408,496 bytes, including scales, biases and the precision map. Other parameters remain FP32; inference uses dequantized tensors.')
labels={'fp32':'FP32 reference','identity':'Identity basis','fixed':'Basis fixed from 4/4','conditioned':'Precision-conditioned basis'}
table(['Method','Perplexity'],[[labels[k],f'{v:.6f}'] for k,v in r['PPL'].items()],[4.8,1.9])
for mode in ['fixed','identity']:
 c=r['comparisons'][mode];lo,hi=c['moving_block_bootstrap_95pct']['8'];d.add_paragraph(f"Conditioned minus {labels[mode].lower()}: {c['delta_ppl']:+.6f} PPL, exploratory 95% circular block-bootstrap interval [{lo:+.6f}, {hi:+.6f}] with eight-block units.")
d.add_paragraph('The interval against the fixed basis includes zero; the identity control is better at this budget. The conditioned and fixed solutions change five bases and no bit choices. Alternating exact bit allocation with basis minimization reaches the same joint solution at 5, 6 and 7 bits on both architectures. Thus the effect is not exclusive to the proposed optimizer.')
d.add_paragraph('A full-precision invariance diagnostic on the first test block gives a maximum FP32 logit difference of 0.03357 against logits reaching magnitude 1083.77, and an NLL difference of 0.0000324. Repeating the eager-attention diagnostic in FP64 gives zero difference at the returned-logit precision. This check does not imply zero numerical error for all contexts. All 96 Q/K projection groups in each serialized patch round-trip exactly.')
d.add_page_break();d.add_heading('Budget sensitivity and decision',level=1)
d.add_paragraph('The following Pythia runs keep the full evaluation corpus and all settings fixed. Bit averages apply only to Q/K weights, not the entire model. Byte counts include the defined patch metadata and retained biases.')
table(['Q K bits','Fixed PPL','Conditioned PPL','Q K bytes'],[[str(b),f"{s[str(b)]['fixed_ppl']:.6f}",f"{s[str(b)]['conditioned_ppl']:.6f}",f"{s[str(b)]['qk_bytes']:,}"] for b in [5,6,7]],[1.05,1.75,2.1,1.8])
for b in [5,6,7]:
 z=s[str(b)];lo,hi=z['bootstrap_95pct_length8'];d.add_paragraph(f"At {b} bits, conditioned minus fixed is {z['delta_ppl']:+.6f}; its within-corpus 95% interval is [{lo:+.6f}, {hi:+.6f}].")
d.add_heading('Claim corrections for the manuscript',level=2)
d.add_paragraph('The exact guarantee concerns the finite, separable operator surrogate. Minimizing over each basis before a multiple-choice knapsack DP is standard variable elimination. It provides no perplexity guarantee and is not an independent contribution by itself. The same limitation applies to the full-precision commuting-rotation identity.')
d.add_paragraph('The real-model and equal-storage implementation gaps have been reduced with two pinned checkpoints, a stronger optimizer control, complete corpus runs, three budgets, numerical checks and serialized patches. The evidence does not establish a substantial independent algorithmic advance. The manuscript must retain these null and adverse controls and must not claim that the novelty objection has been resolved. A full matched reproduction of the closest published pipeline also remains outside this package.')
d.add_paragraph('Artifacts: experiments/rope_20261116/stage1 in https://github.com/ahmadmohammadi80/skb-q-framework. Key records are independent_results.json, independent_summary.json, budget_results.json, budget_summary.json, independent_packed_manifest.json, joint_comparator.json and invariance_diagnostic.json. The initial runtime byte metadata contained a hardcoded arithmetic error; corrected counts are explicitly documented and verified against serialized files.')
d.save(root/'SKBQ_RoPE_Experimental_Revision.docx')
