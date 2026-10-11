"""Render a private local diagnostic viewer; no new service or model execution."""
import argparse
import json
from pathlib import Path

HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pilot corpus · 检索诊断回放</title><style>
body{font:15px/1.55 system-ui,sans-serif;background:#f3f6fa;color:#18283d;margin:0}main{max-width:1440px;margin:auto;padding:28px}h1{margin:0;font-size:28px}h2{font-size:19px}.note{color:#48627e}.status{padding:14px;background:#fff0d9;border:1px solid #dba450;border-radius:9px;font-weight:600}select{padding:10px;width:min(100%,600px);font-size:15px;margin:18px 0}.routes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.route{background:white;padding:18px;border:1px solid #dce3ec;border-radius:12px}.hit{border-top:1px solid #dde5ee;padding:12px 0}code{font-size:10px;overflow-wrap:anywhere}.hit p{margin:7px 0}.context{background:#edf5ff;padding:10px;margin-top:8px;border-left:3px solid #7096cd}a{color:#285ea0}.metric{font-variant-numeric:tabular-nums;color:#3a5776}.empty{color:#854d16}@media(max-width:900px){.routes{grid-template-columns:1fr}}
</style><main><h1>Pilot corpus · 检索诊断回放</h1><p class="note">10篇冻结来源文档 · BM25 / BGE dense / RRF · 本地已记录结果，选择问题不会调用模型</p>
<div class="status">未建立答案支持。当前没有 reader；候选和来源绑定通过均不代表问题已回答。</div>
<select id="cases" aria-label="固定开发诊断"></select><p id="query"></p><p id="timing" class="metric"></p><div id="routes" class="routes"></div>
<p class="note">开发诊断不等于官方 Accuracy / Recall。空表格正文仍未恢复，未使用任意相似度阈值作为校准拒答。原页链接使用已有 8765 演示服务。</p></main>
<script id="data" type="application/json">__DATA__</script><script>
const report=JSON.parse(document.getElementById('data').textContent),sel=document.getElementById('cases');
const node=(tag,value)=>{const e=document.createElement(tag);e.textContent=value;return e;};
report.cases.forEach(c=>{const o=node('option',c.id+' · '+c.kind);o.value=c.id;sel.append(o);});
function region(row,container){
 const box=document.createElement('div');box.className='hit';
 const link=node('a',row.document_name+' · 第'+row.page+'页 · '+row.type);link.href='http://127.0.0.1:8765/page.png?doc='+encodeURIComponent(row.document_name)+'&page='+row.page;link.target='_blank';link.rel='noopener';
 box.append(link,node('p',row.text.slice(0,360)),node('code',row.evidence_id),node('p','bbox ['+(row.source_bbox||[]).join(', ')+'] · 来源绑定已核验'));
 container.append(box);return box;
}
function show(){
 const c=report.cases.find(c=>c.id===sel.value),root=document.getElementById('routes');root.replaceChildren();
 document.getElementById('query').textContent=c.query+' · 范围：'+(c.document||'10篇 pilot corpus');
 document.getElementById('timing').textContent='完整检索 '+c.timings_ms.complete_retrieval+' ms（含 tokenization、编码、三路检索与上下文；不含加载/引用审计）';
 for(const [name,hits] of Object.entries(c.routes)){
  const col=document.createElement('section');col.className='route';col.append(node('h2',name.toUpperCase()+' · '+hits.length+'条候选'),node('p',c.support[name].message));
  if(!hits.length){const empty=node('p','该路没有候选证据。');empty.className='empty';col.append(empty);}
  hits.forEach(h=>{const box=region(h,col);box.append(node('p','路由分数 '+h.score));
   const linked=c.contexts[name].links.filter(l=>l.anchor_evidence_id===h.evidence_id);
   linked.forEach(l=>{const source=c.contexts[name].added_regions.find(r=>r.evidence_id===l.added_evidence_id);const context=document.createElement('div');context.className='context';context.append(node('strong','同页原始顺序上下文'));region(source,context);box.append(context);});
  });root.append(col);
 }
}
sel.value=report.cases.some(c=>c.id==='nonsense')?'nonsense':report.cases[0].id;sel.onchange=show;show();
</script></html>'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=Path('output/pilot-v1/pilot-report.json'))
    parser.add_argument('--output',type=Path,default=Path('output/pilot-v1/pilot-report.html'))
    args=parser.parse_args();report=json.loads(args.report.read_text(encoding='utf-8'))
    if not report.get('complete') or not report.get('all_citations_verified') or report.get('supported_answer'):
        raise ValueError('Only completed source-verified candidate diagnostics can be displayed')
    data=json.dumps(report,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    args.output.write_text(HTML.replace('__DATA__',data),encoding='utf-8')
    print(args.output.resolve())

if __name__=='__main__': main()
