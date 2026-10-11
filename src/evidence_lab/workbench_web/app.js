'use strict';
const el=id=>document.getElementById(id),params=new URLSearchParams(location.search);
let currentEvidence=null,actionSequence=0,verificationSequence=0;
async function api(path){const r=await fetch(path),data=await r.json();if(!r.ok)throw new Error(data.error||'本地请求失败');return data;}
function text(tag,value){const n=document.createElement(tag);n.textContent=value;return n;}
function clearPage(title){currentEvidence=null;verificationSequence++;for(const id of ['verify','region-box','page-image'])el(id).hidden=true;el('page-title').textContent=title;for(const id of ['page-note','page-alert','evidence-text','verification'])el(id).textContent='';}
async function showPage(doc,page,evidence=null,sequence=++actionSequence){
 if(sequence!==actionSequence)return;clearPage('正在读取原始PDF页面');
 try{const data=await api(`/api/page?doc=${encodeURIComponent(doc)}&page=${page}`);if(sequence!==actionSequence)return;
 currentEvidence=evidence;el('verify').hidden=!evidence;el('page-title').textContent=`${doc} · PDF第${page}页`;
 el('page-note').textContent=`PDF SHA256 ${data.doc_sha256}`;el('page-alert').textContent=data.geometry_note;
 el('page-image').hidden=false;el('page-image').src=`/page.png?doc=${encodeURIComponent(doc)}&page=${page}`;
 el('evidence-text').textContent=evidence?.text||'当前页面按原样展示。';
 if(evidence&&data.geometry_overlay_supported){const [x1,y1,x2,y2]=evidence.bbox_normalized,b=el('region-box');b.style.left=`${x1*100}%`;b.style.top=`${y1*100}%`;b.style.width=`${(x2-x1)*100}%`;b.style.height=`${(y2-y1)*100}%`;b.hidden=false;}
 }catch(e){if(sequence===actionSequence)el('search-status').textContent=e.message;}
}
async function search(){
 const sequence=++actionSequence;clearPage('正在本机计算');el('hits').replaceChildren();el('search-status').textContent='正在本机计算，重排和迭代检索可能需要数秒';el('trace').textContent='计算中';
 try{const q=el('query').value,doc=el('documents').value,mode=el('mode').value;
 const data=await api(`/api/search?q=${encodeURIComponent(q)}&doc=${encodeURIComponent(doc)}&mode=${mode}&k=5`);if(sequence!==actionSequence)return;
 el('search-status').textContent=`${data.hits.length}条候选 · ${data.latency_ms}ms · ${data.mode} · 真模型前向${data.model_calls}次`;
 el('reader-status').textContent=data.reader_status;el('trace').textContent=JSON.stringify(data.trace||{mode:data.mode,model_calls:data.model_call_details},null,2);
 for(const hit of data.hits){const card=document.createElement('article');card.className='hit';const b=text('button',`${hit.document_name} · 第${hit.page}页 · ${hit.type}`);b.type='button';b.onclick=()=>showPage(hit.document_name,hit.page,hit);card.append(b,text('p',`${hit.route} ${hit.score} · ${hit.extraction?'原PDF原生表格':'原始解析区域'}`),text('p',hit.text.slice(0,220)),text('code',hit.evidence_id));el('hits').append(card);}
 if(data.hits.length)await showPage(data.hits[0].document_name,data.hits[0].page,data.hits[0],sequence);else{el('hits').append(text('p','当前方式没有候选证据。'));clearPage('无候选页面');}
 }catch(e){if(sequence===actionSequence){el('search-status').textContent=e.message;el('trace').textContent='本次请求未完成。';}}
}
el('search-form').onsubmit=e=>{e.preventDefault();search();};
el('verify').onclick=async()=>{if(!currentEvidence)return;const sequence=actionSequence,v=++verificationSequence,id=currentEvidence.evidence_id;el('verification').textContent='正在核验原始来源';
 try{const data=await api(`/api/verify?id=${encodeURIComponent(id)}`);if(sequence!==actionSequence||v!==verificationSequence||currentEvidence?.evidence_id!==id)return;el('verification').textContent=`${data.all_passed?'来源绑定通过':'来源绑定未通过'} · ${Object.entries(data.checks).map(([k,x])=>`${k}=${x}`).join(' · ')} · 来源绑定不代表答案正确。`;}catch(e){if(sequence===actionSequence&&v===verificationSequence)el('verification').textContent=e.message;}};
async function start(){const meta=await api('/api/meta');el('boundary').textContent=meta.boundary;el('summary').textContent=`${meta.documents.length}份PDF · ${meta.regions}源区域 · ${meta.dense_chunks}真BGE分块 · ${meta.native_tables}原生表格`;
 for(const doc of meta.documents){const option=text('option',doc.name);option.value=doc.name;el('documents').append(option);}el('documents').value=params.get('doc')||'crm_4502338';if(params.has('q'))el('query').value=params.get('q');if(params.has('mode'))el('mode').value=params.get('mode');await search();}
start().catch(e=>{el('search-status').textContent=e.message;});
