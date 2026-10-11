// Real loopback UI/source checks. Requires the completed local workbench on8766.
const fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.EVIDENCE_PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const origin=process.env.EVIDENCE_WORKBENCH_ORIGIN||'http://127.0.0.1:8766';
 const target=new URL(origin);
 if(target.protocol!=='http:'||target.hostname!=='127.0.0.1'||target.username||target.password||target.pathname!=='/'||target.search||target.hash)throw new Error('Only a loopback workbench origin is permitted');
 const output=path.resolve(process.env.EVIDENCE_WORKBENCH_OUTPUT||'output/runs/full-workbench-20261005');
 fs.mkdirSync(output,{recursive:true});
 if(fs.existsSync(path.join(output,'browser-workbench.json')))throw new Error('Use a new output directory; completed browser evidence is immutable');
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
 const context=await browser.newContext({viewport:{width:1440,height:1100}}),errors=[],external=[],checks={},runs=[];
 await context.route('**/*',async route=>{const url=new URL(route.request().url());
  if(url.origin===origin)await route.continue();else{external.push(url.origin);await route.abort();}
 });
 const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
 await page.goto(origin);await page.locator('#hits article').first().waitFor({timeout:120000});
 await page.waitForFunction(()=>{const img=document.getElementById('page-image');return !img.hidden&&img.complete&&img.naturalWidth>0;},{},{timeout:60000});
 const meta=await (await context.request.get(origin+'/api/meta')).json();
 checks.full100_documents=meta.documents.length===100;
 checks.genuine12531_chunks=meta.dense_chunks===12531;
 checks.native17_tables=meta.native_tables===17;
 checks.no_reader_or_generated_answer=meta.reader_present===false&&meta.answer_generated===false;
 checks.actual_default_bge_forward=meta.model_call_details.bge_embeddings>=1;
 checks.full_index_binding=meta.index_sha256==='69cf88ab4af7fb2b872b945570f98e279fc8d9010242ca38fd59edafa7a92206';
 checks.five_selectable_modes=await page.locator('#mode option').count()===5;
 async function search(mode,query='Receiving Clerk'){
  await page.locator('#query').fill(query);await page.locator('#mode').selectOption(mode);
  const responsePromise=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/search',{timeout:120000});
  await page.locator('#search-form button').click();const response=await responsePromise,data=await response.json();
  await page.waitForFunction(()=>!document.getElementById('verify').hidden,{},{timeout:60000});
  await page.waitForFunction(()=>{const img=document.getElementById('page-image');return !img.hidden&&img.complete&&img.naturalWidth>0;},{},{timeout:60000});
  runs.push({mode,latency_ms:data.latency_ms,model_calls:data.model_call_details,
   hits:data.hits.map(h=>({evidence_id:h.evidence_id,document_name:h.document_name,page:h.page,native:!!h.extraction})),
   supported_answer:data.supported_answer,answer_generated:data.answer_generated,
   stop_reasons:data.trace?.stop_reasons||[],soft_wall_budget_exceeded:data.trace?.soft_wall_budget_exceeded||false});
  checks[mode+'_real_response']=response.ok()&&data.mode===mode&&data.hits.length>0&&data.hits.length<=5;
  checks[mode+'_does_not_claim_answer']=data.supported_answer===false&&data.answer_generated===false;
  return data;
 }
 const bm25=await search('bm25');
 checks.bm25_zero_model_forward=bm25.model_calls===0;
 const native=bm25.hits.find(h=>h.extraction);
 checks.repaired_native_page8_returned=!!native&&native.page===8;
 if(native){
  const id=bm25.hits.indexOf(native);await page.locator('#hits article button').nth(id).click();
  await page.waitForFunction(()=>document.getElementById('page-title').textContent.includes('8'),{},{timeout:30000});
  const verificationPromise=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/verify');
  await page.locator('#verify').click();const verification=await (await verificationPromise).json();
  checks.native_six_source_checks=verification.all_passed&&Object.keys(verification.checks).length===6&&Object.values(verification.checks).every(x=>x===true);
  checks.coordinate_overlay_visible=await page.locator('#region-box').isVisible();
  const png=await context.request.get(origin+`/page.png?doc=${native.document_name}&page=${native.page}`);
  checks.original_pdf_png=png.ok()&&(await png.body()).subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10]));
  await page.waitForFunction(()=>document.getElementById('verification').textContent.includes('true'));
  await page.evaluate(()=>scrollTo(0,0));
  await page.screenshot({path:path.join(output,'native-table-workbench.png'),fullPage:true});
 }
 const dense=await search('dense');checks.dense_actual_embedding=dense.model_call_details.bge_embeddings===1;
 const hybrid=await search('hybrid');checks.hybrid_actual_embedding=hybrid.model_call_details.bge_embeddings===1;
 const rerank=await search('rerank');checks.real_reranker_pairs=rerank.model_call_details.reranker_pairs>0&&rerank.model_call_details.reranker_pairs<=40;
 const rerankNotes=rerank.trace.source_window_notes;
 checks.reranker_has_verified_dense_matches=rerankNotes.some(n=>n.matched_chunk_verified_against_index_and_source_tokens&&n.matched_chunk);
 checks.reranker_uses_original_dense_windows=rerankNotes.some(n=>n.source_windows.some(w=>w.dense_overlap>0));
 const agent=await search('agent','Receiving Clerk responsibility and purchase document');
 checks.bounded_actual_agent=agent.model_call_details.bge_embeddings>=1&&agent.model_call_details.bge_embeddings<=3&&agent.trace!==null;
 checks.agent_nested_fusion_has_verified_dense_match=agent.trace.rerank_stats.source_window_notes.some(n=>n.matched_chunk_verified_against_index_and_source_tokens&&n.source_windows.some(w=>w.dense_overlap>0));
 await page.locator('details').evaluate(n=>n.open=true);
 await page.screenshot({path:path.join(output,'iterative-workbench.png'),fullPage:false});
 for(const [label,url] of [
  ['unknown_mode','/api/search?q=test&mode=unknown'],['invalid_document','/api/search?q=test&doc=invalid'],
  ['empty_query','/api/search?q=&mode=hybrid'],['over_budget_k','/api/search?q=test&k=6'],
  ['unknown_evidence','/api/verify?id=invalid'],['outside_pdf_page','/api/page?doc=crm_4502338&page=9999']]){
  checks[label+'_rejected']=(await context.request.get(origin+url)).status()===400;
 }
 checks.no_page_errors=errors.length===0;checks.no_external_browser_requests=external.length===0;
 const report={kind:'Actual full-corpus local workbench browser/source checks; not answer accuracy',
  checks,all_passed:Object.values(checks).every(Boolean),runs,page_errors:errors,external_requests:external,
  reader_run:false,source_index_sha256:meta.index_sha256};
 fs.writeFileSync(path.join(output,'browser-workbench.json'),JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report,null,2));await browser.close();if(!report.all_passed)process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1;});
