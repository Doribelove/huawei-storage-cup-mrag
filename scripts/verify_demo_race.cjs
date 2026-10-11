const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {chromium}=require(process.env.EVIDENCE_PLAYWRIGHT_MODULE||'playwright');
const base=process.env.EVIDENCE_DEMO_URL||'http://127.0.0.1:8766';
if(new URL(base).hostname!=='127.0.0.1')throw new Error('Only loopback demo is supported');
(async()=>{
 const output=path.resolve('output/audit/demo-race');fs.mkdirSync(output,{recursive:true});
 const browser=await chromium.launch({headless:true,executablePath:process.env.EVIDENCE_BROWSER_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe'});
 const context=await browser.newContext({viewport:{width:1280,height:1000}});
 const errors=[],external=[];let mode='',oldId='';
 await context.route('**/*',async route=>{
  const u=new URL(route.request().url());
  if(u.origin!==new URL(base).origin){external.push(u.origin);await route.abort();return;}
  const delay=(mode==='search'&&u.pathname==='/api/search'&&u.searchParams.get('doc')==='education_3501298')||
    (mode==='page'&&u.pathname==='/api/page'&&u.searchParams.get('doc')==='education_3501298')||
    (mode==='verify'&&u.pathname==='/api/verify'&&u.searchParams.get('id')===oldId);
  if(delay){const response=await route.fetch();await new Promise(resolve=>setTimeout(resolve,1000));await route.fulfill({response});}
  else await route.continue();
 });
 const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
 await page.goto(base,{waitUntil:'networkidle'});await page.locator('#hits .hit').first().waitFor();
 const query=async(doc,q)=>{await page.locator('#documents').selectOption(doc);await page.locator('#query').fill(q);await page.locator('#search-form button').click();};
 const finance=async()=>{await query('finance_1372325','foreign direct investment');await page.locator('#page-title').filter({hasText:'finance_1372325 · PDF 第 5 页'}).waitFor();};
 const correctFinance=async()=>{
  const labels=await page.locator('#hits .hit button').allTextContents();
  return labels.length===5&&labels.every(s=>s.includes('finance_1372325'))&&
    (await page.locator('#page-title').textContent()).includes('finance_1372325')&&
    (await page.locator('#search-status').textContent()).includes('finance_1372325');
 };
 mode='search';await query('education_3501298','SDCA Cycle');await page.waitForTimeout(150);await finance();await page.waitForTimeout(1300);
 const checks={late_search_cannot_mix_results_or_change_page:await correctFinance()};
 mode='page';await query('education_3501298','SDCA Cycle');await page.locator('#hits .hit button').first().filter({hasText:'education_3501298'}).waitFor();
 await page.waitForTimeout(100);await finance();await page.waitForTimeout(1300);checks.late_page_cannot_replace_latest_selection=await correctFinance();
 mode='verify';oldId=await page.locator('#hits .hit code').first().textContent();await page.locator('#verify').click();
 await page.waitForTimeout(100);await query('education_3501298','SDCA Cycle');
 await page.locator('#page-title').filter({hasText:'education_3501298 · PDF 第 3 页'}).waitFor();await page.waitForTimeout(1300);
 checks.late_verification_cannot_attach_to_new_evidence=(await page.locator('#verification').textContent())==='';
 checks.no_page_errors=errors.length===0;checks.no_external_requests=external.length===0;
 const response=await context.request.get(base+'/app.js'),served=await response.body();
 const local=fs.readFileSync('src/evidence_lab/web/app.js');
 checks.served_script_matches_fixed_worktree=Buffer.compare(served,local)===0;
 await page.screenshot({path:path.join(output,'demo-race-fixed.png'),fullPage:true});
 const report={url:base,delay_ms:1000,response_source:'real local API responses delayed; not fabricated hits',checks,
   all_passed:Object.values(checks).every(Boolean),page_errors:errors,external_requests:external,
   served_app_sha256:crypto.createHash('sha256').update(served).digest('hex')};
 fs.writeFileSync(path.join(output,'browser-race.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
 await browser.close();if(!report.all_passed)process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1;});
