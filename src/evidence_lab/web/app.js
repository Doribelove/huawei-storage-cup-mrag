'use strict';
const el=id=>document.getElementById(id);
const params=new URLSearchParams(location.search);
let currentEvidence=null;
let actionSequence=0,verificationSequence=0;
async function api(path){const response=await fetch(path);const data=await response.json();if(!response.ok)throw new Error(data.error||'本机请求失败');return data;}
function text(tag,value){const node=document.createElement(tag);node.textContent=value;return node;}
const classes={'blank':'目视空白','scan':'扫描页','parse-missing':'表格解析缺失','uncertain':'坐标/内容不确定'};
function clearPage(title){
  currentEvidence=null;verificationSequence++;
  el('verify').hidden=true;el('region-box').hidden=true;el('page-image').hidden=true;
  el('page-title').textContent=title;
  for(const id of ['page-note','page-alert','evidence-text','verification'])el(id).textContent='';
}
async function showPage(doc,page,evidence=null,sequence=++actionSequence){
  if(sequence!==actionSequence)return;
  clearPage('正在读取原始页面…');
  try{
  currentEvidence=null;
  el('verification').textContent='';el('region-box').hidden=true;el('verify').hidden=true;
  const data=await api(`/api/page?doc=${encodeURIComponent(doc)}&page=${page}`);
  if(sequence!==actionSequence)return;
  if(typeof evidence==='string')evidence=data.regions.find(region=>region.evidence_id===evidence)||null;
  currentEvidence=evidence;el('verify').hidden=!evidence;
  el('page-title').textContent=`${doc} · PDF 第 ${page} 页`;
  el('page-note').textContent=`PDF SHA-256：${data.doc_sha256}。${data.geometry_note}`;
  el('page-alert').textContent=data.anomalies.map(a=>`${classes[a.classification]}：${a.note} ${a.items.filter(i=>i.source_bbox).map(i=>`原始 bbox [${i.source_bbox.join(', ')}]`).join('；')}`).join(' / ');
  const image=el('page-image');image.hidden=false;image.src=`/page.png?doc=${encodeURIComponent(doc)}&page=${page}`;
  el('evidence-text').textContent=evidence?evidence.text:'整页回退：没有自动补猜内容。';
  if(evidence&&data.geometry_overlay_supported){
    const [x1,y1,x2,y2]=evidence.bbox_normalized,box=el('region-box');
    box.style.left=`${x1*100}%`;box.style.top=`${y1*100}%`;box.style.width=`${(x2-x1)*100}%`;box.style.height=`${(y2-y1)*100}%`;box.hidden=false;
  }else if(evidence){el('page-alert').textContent+=' 当前页面几何暂不支持区域叠加，请核对整页。';}
  }catch(error){if(sequence===actionSequence)reportError(error);}
}
async function search(){
  const sequence=++actionSequence;
  clearPage('正在检索证据…');
  el('search-status').textContent='正在检索本机证据…';el('hits').replaceChildren();
  const q=el('query').value,doc=el('documents').value;
  try{
    const data=await api(`/api/search?q=${encodeURIComponent(q)}&doc=${encodeURIComponent(doc)}&k=5`);
    if(sequence!==actionSequence)return;
    el('search-status').textContent=`${data.hits.length} 条结果 · ${data.latency_ms} ms · BM25范围：${doc||'全部文档'} · 无模型调用`;
    if(!data.hits.length){
      el('hits').append(text('p','没有匹配证据；不生成答案。'));clearPage('当前查询没有匹配证据');
      return;
    }
    for(const hit of data.hits){
      const card=document.createElement('article');card.className='hit';const button=text('button',`${hit.document_name} · 第${hit.page}页 · ${hit.type}`);
      button.type='button';button.onclick=()=>showPage(hit.document_name,hit.page,hit).catch(reportError);
      card.append(button,text('p',`BM25 ${hit.score} · bbox [${hit.source_bbox.join(', ')}]${hit.anomaly_classes.length?' · '+hit.anomaly_classes.map(c=>classes[c]).join('/') :''}`),text('p',hit.text.slice(0,220)),text('code',hit.evidence_id));
      el('hits').append(card);
    }
    await showPage(data.hits[0].document_name,data.hits[0].page,data.hits[0],sequence);
  }catch(error){if(sequence===actionSequence)reportError(error);}
}
function reportError(error){el('search-status').textContent=`本机操作失败：${error.message}`;}
el('search-form').onsubmit=event=>{event.preventDefault();search();};
el('verify').onclick=async()=>{
  if(!currentEvidence)return;el('verification').textContent='正在核验…';
  const sequence=actionSequence,verification=++verificationSequence,id=currentEvidence.evidence_id;
  const current=()=>sequence===actionSequence&&verification===verificationSequence&&currentEvidence?.evidence_id===id;
  try{const data=await api(`/api/verify?id=${encodeURIComponent(id)}`);if(!current())return;el('verification').textContent=`${data.all_passed?'引用核验通过':'引用核验未通过'}：${Object.entries(data.checks).map(([key,value])=>`${key}=${value?'通过':'失败'}`).join('，')}。${data.meaning}`;}catch(error){if(current())el('verification').textContent=error.message;}
};
async function start(){
  const meta=await api('/api/meta');el('boundary').textContent=meta.boundary;
  el('summary').textContent=`${meta.documents.length} 份 PDF · ${meta.regions} 条证据 · ${meta.reviewed_pages} 页已检查 · ${meta.orphan_chunks} 条孤立 chunk · 原始索引保留`;
  for(const doc of meta.documents){const option=text('option',doc.name);option.value=doc.name;el('documents').append(option);}
  const anomalies=await api('/api/anomalies');
  for(const page of anomalies.pages){const row=document.createElement('div');row.className='anomaly-row';const button=text('button',`${page.document_name} · 第${page.page}页`);button.type='button';button.onclick=()=>showPage(page.document_name,page.page).catch(reportError);row.append(button,text('span',`${classes[page.visual_classification]} — ${page.review_note}`));el('anomalies').append(row);}
  for(const link of anomalies.links){const row=document.createElement('div');row.className='link-row';const source=link.source_links[0];
    if(source){const button=text('button',`${source.document_name} · 第${source.page}页`);button.type='button';button.onclick=()=>showPage(source.document_name,source.page,source.evidence_id).catch(reportError);row.append(button);}
    row.append(text('span',`${link.chunk_id} · 来源 ${link.link_status} · 分析正文 unknown`));el('chunk-links').append(row);
  }
  if(params.has('q'))el('query').value=params.get('q');if(params.has('doc'))el('documents').value=params.get('doc');
  await search();
}
start().catch(reportError);
