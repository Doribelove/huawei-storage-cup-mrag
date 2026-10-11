const fs=require('node:fs'),path=require('node:path'),{pathToFileURL}=require('node:url');
const {chromium}=require(process.env.EVIDENCE_PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const output=path.resolve('output/pilot-v1');
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
 const context=await browser.newContext({viewport:{width:1280,height:1000}}),errors=[],external=[];
 await context.route('**/*',async route=>{const u=new URL(route.request().url());
  if(u.protocol==='file:'||u.origin==='http://127.0.0.1:8765')await route.continue();
  else{external.push(u.origin);await route.abort();}
 });
 const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
 await page.goto(pathToFileURL(path.join(output,'pilot-report.html')).href);
 const checks={pilot_corpus_label:await page.locator('h1').textContent()==='Pilot corpus · 检索诊断回放',
  frozen_case_count:await page.locator('#cases option').count()===14,
  three_routes:await page.locator('.route').count()===3,
  unsupported_answer_notice:(await page.locator('.status').textContent()).includes('未建立答案支持'),
  nonsense_keeps_dense_candidates:(await page.locator('.route h2').allTextContents()).join('|')==='BM25 · 0条候选|DENSE · 5条候选|RRF · 5条候选'};
 await page.locator('#cases').selectOption('cross_document_pandemic');
 checks.case_switch_updates_query=(await page.locator('#query').textContent()).includes('COVID-19 financial results and digitalization');
 await page.locator('#cases').selectOption('missing_responsibility_table');
 checks.scoped_case_stays_in_pilot_document=(await page.locator('#query').textContent()).includes('crm_4502338')&&
  (await page.locator('.hit a').allTextContents()).every(t=>t.includes('crm_4502338'));
 await page.locator('#cases').selectOption('nonsense');
 await page.screenshot({path:path.join(output,'pilot-viewer.png'),fullPage:false});
 const metaResponse=await context.request.get('http://127.0.0.1:8765/api/meta');const meta=await metaResponse.json();
 checks.original_demo_still_100_documents=metaResponse.ok()&&meta.documents.length===100&&meta.model_calls===0;
 checks.no_page_errors=errors.length===0;checks.no_external_browser_requests=external.length===0;
 const report={checks,all_passed:Object.values(checks).every(Boolean),page_errors:errors,external_requests:external,
  mode:'private local frozen diagnostic replay; no model call or new service',original_demo:'http://127.0.0.1:8765'};
 fs.writeFileSync(path.join(output,'browser-pilot.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
 await browser.close();if(!report.all_passed)process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1;});
