from pathlib import Path
import json
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
p=Path(__file__).parent
r=json.load(open(p/'summary.json'));v=json.load(open(p/'validation.json'));iv=json.load(open(p/'model_invariance.json'))
d=Document();sec=d.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11);sec.top_margin=sec.bottom_margin=Inches(.75);sec.left_margin=sec.right_margin=Inches(.8)
for nm in ['Normal','Title','Heading 1','Heading 2']:
 st=d.styles[nm];st.font.name='Calibri';st.font.color.rgb=RGBColor(0,0,0)
d.styles['Normal'].font.size=Pt(11);d.styles['Normal'].paragraph_format.space_after=Pt(7)
d.styles['Normal'].paragraph_format.line_spacing=1.08
d.styles['Title'].font.size=Pt(23);d.styles['Heading 1'].font.size=Pt(15)
d.settings.odd_and_even_pages_header_footer=False
sec.different_first_page_header_footer=False
for node in list(d.styles.element.iter(qn('w:pBdr'))):node.getparent().remove(node)
head=sec.header.paragraphs[0];head.text='SKB-Q  |  Real-model evidence and method specification';head.style=d.styles['Normal'];head.runs[0].font.size=Pt(9)
footer=sec.footer.paragraphs[0];footer.alignment=2;footer.add_run('16 November 2026   |   ');fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)
d.add_paragraph('SKB Q RoPE experimental revision','Title');d.add_paragraph('Precision-conditioned basis selection at an equal Q/K storage budget')
d.add_paragraph('Technical replacement sections and verified experimental evidence for independent review. This document does not designate the earlier manuscript as ready for submission. It reports a new real-model experiment, separates a published-rule transplantation from full-method reproduction, and limits the novelty claim to what has actually been investigated.')
d.add_heading('Verified real model results',level=1)
d.add_paragraph(f"The pinned SmolLM2-135M checkpoint was evaluated on {r['blocks']} complete WikiText-2 test blocks, with {r['predicted_tokens']:,} predicted tokens. All quantized rows below use exactly {r['budget_QK_bytes']:,} bytes for the defined Q/K payload, row scales and precision map. Other weights and activations remain float32. Lower perplexity is better.")
t=d.add_table(rows=1,cols=3);t.autofit=False
for c,w in zip(t.columns,[3.65,1.15,1.9]):c.width=Inches(w)
for cell,txt in zip(t.rows[0].cells,['Method','Perplexity','Q/K bytes']):cell.text=txt
labels={'fp32':'FP32 reference','identity':'Identity basis plus exact allocation','fixed':'Basis fixed from 4/4 scores','conditioned':'Precision-conditioned basis','published_rule_transplant':'Published angle rule transplanted to Q/K weights'}
for m,name in labels.items():
 cells=t.add_row().cells;cells[0].text=name;cells[1].text=f"{r['PPL'][m]:.6f}";cells[2].text='53,084,160' if m=='fp32' else f"{r['budget_QK_bytes']:,}"
for ri,row in enumerate(t.rows):
 for ci,cell in enumerate(row.cells):
  pr=cell._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
  for side in ['top','left','bottom','right']:
   e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');b.append(e)
  pr.append(b);mar=OxmlElement('w:tcMar')
  for side in ['top','bottom','left','right']:
   e=OxmlElement('w:'+side);e.set(qn('w:w'),'90');e.set(qn('w:type'),'dxa');mar.append(e)
  pr.append(mar)
  for para in cell.paragraphs:
   para.paragraph_format.space_after=Pt(3);para.paragraph_format.space_before=Pt(3)
   if ci:para.alignment=1
  if ri==0:
   shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E8EDF3');pr.append(shade)
   for run in cell.paragraphs[0].runs:run.bold=True
repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
d.add_paragraph('The FP32 row is a quality anchor, not a matched-budget competitor. Byte counts describe the explicitly defined packed representation; forward passes use dequantized float32 tensors.')
for other in ['fixed','identity','published_rule_transplant']:
 c=r['comparisons'][other];lo,hi=c['circular_moving_block_bootstrap_95pct_ppl_difference']['8']
 d.add_paragraph(f"Against {labels[other].lower()}, the conditioned-minus-comparator PPL difference is {c['delta_ppl_conditioned_minus_comparator']:+.6f}; the exploratory 95% circular block-bootstrap interval is [{lo:+.6f}, {hi:+.6f}] using eight-block resampling units. Its NLL is lower on {c['blocks_lower_nll']} of {r['blocks']} blocks.")
d.add_paragraph(f"The conditioned and fixed controls differ in {r['basis_changes_conditioned_vs_fixed']} selected bases and {r['bit_map_changes_conditioned_vs_fixed']} bit choices. This isolates a basis-selection effect at this budget; it does not demonstrate an allocation-change mechanism. Intervals describe one corpus and checkpoint, with no claim of independent model or calibration replications. Results for resampling lengths one and sixteen are retained in summary.json.")
d.add_heading('Executable method and decision rule',level=1)
text=(p/'METHOD.md').read_text();body=text.split('## Decision rule',1)[1]
for line in body.splitlines():
 if not line.strip():continue
 if line.startswith('## '):d.add_heading(line[3:],level=1)
 else:
  line=line.replace('An orthogonal basis U commuting with R satisfies U-transpose R U = R.','Conjugating a RoPE rotation by an orthogonal commuting basis leaves the rotation unchanged.').replace('Q-transpose R K operators','bilinear query–key operators')
  d.add_paragraph(line.replace('`', ''))
d.add_heading('Checks completed',level=1)
d.add_paragraph(f"An independent dense-matrix check of the Gram formula achieved maximum relative discrepancy {v['gram_relative_error_max']:.3g}. The small allocation check matched exhaustive enumeration. Actual packed Q/K patches were exported for all four quantized methods. Every file measured 10,045,530 bytes, and all 180 projection groups per file round-tripped exactly. Their hashes are retained in packed_qk_manifest.json. On the real model, rotating Q/K without quantization changed logits by at most {iv['max_logit_absolute_difference']:.6g}, with RMS difference {iv['rms_logit_difference']:.6g}; this is a numerical integration check, not an independent accuracy experiment.")
d.add_heading('Submission and novelty boundary',level=1)
d.add_paragraph('The requested real-RoPE-model comparison is now supported by recorded forward passes and equal-budget controls. The published estimator has also been tested directly inside the controlled Q/K weight-only protocol. A complete reproduction of the closest paper’s original dynamic quantization pipeline and comparison against a stronger jointly optimized rotation/allocation competitor are not supplied by these experiments. The editor’s objection cannot be declared resolved from a small or uncertain PPL difference, a standard dynamic program, or full-precision invariance. Independent novelty and a substantial contribution remain unestablished. This document is an evidence-bearing research revision, not a submission-readiness certificate.')
d.add_heading('Primary references and artifacts',level=1)
for s in [
 'Wang, S., Luo, Y., Xu, N., and Cheung, C. W. (2026). When Local Variance Optimality Is Not Enough: RoPE-Aligned Q/K Rotations for Dynamic 4-Bit Quantisation. arXiv:2608.13365. https://arxiv.org/html/2608.13365',
 'SpinQuant: LLM Quantization with Learned Rotations. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html',
 'Bento, M. P., and Seabra, J. (2026). GaugeQuant: Online Learning of Quantization-Optimal Bases from LLM Symmetries. arXiv:2607.20757. https://github.com/MPedraBento/gauge-quant',
 'Ben Basat, R., Mitzenmacher, M., and Vargaftik, S. (2026). TORQUE: Optimizing What (not) to Quantize Before and After Rotation. arXiv:2609.36032.',
 'Model: https://huggingface.co/HuggingFaceTB/SmolLM2-135M; exact revision and all environment versions are recorded with the experiment.',
 'Code and records: https://github.com/ahmadmohammadi80/skb-q-framework/tree/main/experiments/rope_20261116. See verified_results.json, summary.json, validation.json, model_invariance.json and METHOD.md.'
]:d.add_paragraph(s)
d.save(p/'SKBQ_RoPE_Experimental_Revision.docx')
