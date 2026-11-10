// Bộ thu thập chạy trong trình duyệt (thay cho bước 01 khi máy không gọi được API trực tiếp).
// Cách dùng: mở https://en.wikipedia.org/wiki/Special:BlankPage, mở DevTools > Console,
// dán toàn bộ file này, Enter. Theo dõi tiến độ bằng: S.status
// Khi S.done === true, gọi: await exportDump()  -> tải về scinet_dump.json.gz
// Rồi: python scripts/00_import_browser_dump.py scinet_dump.json.gz
window.S={status:'start',pages:{},linkTypes:{},aliases:{},count:0,done:false,err:null,reqs:0};
(async()=>{
const S=window.S, API='/w/api.php', H={'Api-User-Agent':'SNA-ScientistNetwork/1.0 (student course project)'};
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function api(params){ params=Object.assign({format:'json',formatversion:2},params);
  for(let a=0;a<6;a++){ try{ const r=await fetch(API,{method:'POST',body:new URLSearchParams(params),headers:H}); S.reqs++; if(r.status==429||r.status>=500){await sleep(3000*(a+1));continue;} return await r.json(); }catch(e){await sleep(2000*(a+1));} }
  throw new Error('api fail'); }
const SKIP=['file:','image:','category:','wikt:','wiktionary:','s:','commons:','help:','wikipedia:','template:','portal:',':category:','w:','d:','q:','b:','n:','v:','meta:','species:'];
function norm(t){ t=(t||'').replace(/_/g,' ').replace(/\s+/g,' ').trim(); if(!t) return null; const l=t.toLowerCase(); if(SKIP.some(s=>l.startsWith(s))) return null; t=t.split('#')[0].trim(); if(!t) return null; return t[0].toUpperCase()+t.slice(1); }
function links(s){ const out=[]; const re=/\[\[([^\]\|]*)(?:\|[^\]]*)?\]\]/g; let m; while((m=re.exec(s))){ const t=norm(m[1]); if(t) out.push(t);} return out; }
const SCI=['Infobox scientist','Infobox academic','Infobox mathematician','Infobox physicist','Infobox chemist','Infobox economist','Infobox engineer','Infobox medical person','Infobox biologist','Infobox astronomer','Infobox computer scientist','Infobox philosopher'].map(x=>'Template:'+x);
const INST=['Infobox university','Infobox organization','Infobox laboratory','Infobox research institute','Infobox company','Infobox government agency','Infobox school','Infobox college','Infobox academy','Infobox institute','Infobox scientific organization','Infobox learned society'].map(x=>'Template:'+x);
const TL=SCI.concat(INST).join('|');
async function classify(titles){ const res={};
  for(let i=0;i<titles.length;i+=50){ const chunk=titles.slice(i,i+50); let cont={}; const alias={}, kinds={};
    while(true){ const d=await api(Object.assign({action:'query',prop:'templates',redirects:1,tllimit:'max',tltemplates:TL,titles:chunk.join('|')},cont)); const q=d.query||{};
      (q.normalized||[]).forEach(n=>alias[n.from]=n.to); (q.redirects||[]).forEach(n=>alias[n.from]=n.to);
      (q.pages||[]).forEach(p=>{ if(p.missing||p.invalid) return; const ts=(p.templates||[]).map(t=>t.title); const k=ts.some(t=>SCI.includes(t))?'Scientist':(ts.some(t=>INST.includes(t))?'Institution':'Other'); if(!kinds[p.title]||k!=='Other') kinds[p.title]=k; });
      if(d.continue){cont=d.continue;} else break; }
    for(const t of chunk){ let c=t; for(let k=0;k<3;k++) c=alias[c]||c; if(kinds[c]) res[t]=[c,kinds[c]]; } }
  return res; }
async function fetchWT(titles){ const res={};
  for(let i=0;i<titles.length;i+=20){ const chunk=titles.slice(i,i+20);
    const d=await api({action:'query',prop:'revisions',rvprop:'content',rvslots:'main',redirects:1,titles:chunk.join('|')}); const q=d.query||{}; const alias={};
    (q.normalized||[]).forEach(n=>alias[n.from]=n.to); (q.redirects||[]).forEach(n=>alias[n.from]=n.to);
    const pages={}; (q.pages||[]).forEach(p=>{ if(!p.missing&&p.revisions) pages[p.title]={title:p.title,pageid:p.pageid,wt:p.revisions[0].slots.main.content}; });
    for(const t of chunk){ let c=t; for(let k=0;k<3;k++) c=alias[c]||c; if(pages[c]) res[t]=pages[c]; } }
  return res; }
function infobox(wt){ const m=/\{\{\s*infobox/i.exec(wt); if(!m) return ''; let i=m.index, depth=0; for(let j=i;j<wt.length-1;j++){ if(wt[j]=='{'&&wt[j+1]=='{'){depth++;j++;} else if(wt[j]=='}'&&wt[j+1]=='}'){depth--;j++; if(depth==0) return wt.slice(i,j+1);} } return wt.slice(i,i+8000); }
const SEC=/(career|research|scientific|work|contribution|controvers|criticism|dispute|debate|legacy|later life|academic|influence|reception)/i;
const SKIPSEC=['references','notes','external links','see also','bibliography','further reading','publications','selected publications','honours and awards','awards'];
// Bộ lọc đoạn văn rộng hơn từ điển trigger của bước 02 (chỉ để giảm dung lượng, không quyết định quan hệ)
const TRIG=/(critic|challeng|object|attack|skeptic|sceptic|question|oppos|clash|denounc|refut|disprov|rebut|reject|debunk|contradict|overturn|falsif|discredit|wrong|incorrect|flawed|priority|plagiar|credit|collaborat|together|jointly|co-?author|co-?discover|co-?found|partner|along with|with (his|her)|work(ed|ing|s)? (together )?with|build|built|based on|extend|generali[sz]|elaborat|draw|drew|inspire|follow|develop|refin|expand|advis|consult|council)/i;
function clean(wt){ return wt.replace(/<!--[\s\S]*?-->/g,'').replace(/<ref[^>\/]*\/>/gi,'').replace(/<ref[^>]*>[\s\S]*?<\/ref>/gi,'').replace(/\{\{\s*(sfn|sfnp|efn|cite[^|}]*|harvnb|harv|refn|citation needed|cn|rp|dubious|clarify)[^{}]*\}\}/gi,''); }
const PERSON=/\|\s*(doctoral_advisors?|doctoral_students|academic_advisors?|notable_students|other_students|influences|influenced)\s*=([\s\S]*?)(?=\n\s*\||\n\}\})/g;
function compact(wt){ wt=clean(wt); const ib=infobox(wt); const rest=ib?wt.replace(ib,''):wt;
  const parts=rest.split(/^(==+)\s*(.*?)\s*\1\s*$/m); let out=ib+'\n'; const secLinks=[];
  const keepPars=(txt)=>txt.split(/\n\s*\n/).filter(p=>{const s=p.trim(); return s.includes('[[')&&TRIG.test(s)&&!s.startsWith('{|')&&!/^\[\[(file|image):/i.test(s);});
  let lead=keepPars(parts[0]); out+=lead.join('\n\n')+'\n'; lead.forEach(p=>secLinks.push(...links(p)));
  for(let i=1;i<parts.length;i+=3){ const h=parts[i+1]||''; const body=parts[i+2]||''; const hl=h.toLowerCase().replace(/\[\[|\]\]|'''|''/g,'');
    if(SKIPSEC.includes(hl.trim())||!SEC.test(hl)) continue; const ps=keepPars(body); if(!ps.length) continue; out+='\n== '+h+' ==\n'+ps.join('\n\n')+'\n'; ps.forEach(p=>secLinks.push(...links(p))); }
  const person=[]; let m; PERSON.lastIndex=0; while((m=PERSON.exec(ib))) person.push(...links(m[2]));
  return {text:out, ibLinks:links(ib), secLinks, person}; }
try{
  const cats=["Nobel laureates in Physics","Nobel laureates in Chemistry","Nobel laureates in Physiology or Medicine","Nobel laureates in Economics","Fields Medalists","Turing Award laureates","Wolf Prize in Physics laureates","Wolf Prize in Mathematics laureates","Abel Prize laureates","Recipients of the Copley Medal","Royal Medal winners","Kyoto laureates in Basic Sciences","Kyoto laureates in Advanced Technology"];
  let seeds=[]; for(const c of cats){ const d=await api({action:'query',list:'categorymembers',cmtype:'page',cmnamespace:0,cmlimit:500,cmtitle:'Category:'+c}); seeds.push(...d.query.categorymembers.map(x=>x.title)); }
  seeds=[...new Set(seeds)]; S.seeds=seeds.length; S.status='seeds '+seeds.length;
  const TARGET=3200; const seen=new Set(seeds); let queue=seeds.map(t=>[t,0]); const allLinks=new Set();
  while(queue.length&&S.count<TARGET){ const batch=queue.splice(0,50); const depth={}; batch.forEach(([t,d])=>depth[t]=d);
    const kinds=await classify(batch.map(b=>b[0])); const keep=Object.keys(kinds).filter(t=>kinds[t][1]==='Scientist'||depth[t]===0);
    keep.forEach(t=>{S.aliases[t]=kinds[t][0]; S.linkTypes[kinds[t][0]]=kinds[t][1];});
    const pages=await fetchWT(keep);
    for(const t of Object.keys(pages)){ const p=pages[t]; if(S.pages[p.title]) continue; const c=compact(p.wt);
      S.pages[p.title]={title:p.title,pageid:p.pageid,depth:depth[t],wikitext:c.text}; S.count++;
      c.ibLinks.forEach(x=>allLinks.add(x)); c.secLinks.forEach(x=>allLinks.add(x));
      if(depth[t]<2){ const nb=[...new Set(c.person.concat(c.secLinks.slice(0,60)))]; nb.forEach(n=>{ if(!seen.has(n)){seen.add(n); queue.push([n,depth[t]+1]);} }); } }
    S.status='crawl '+S.count+' queue '+queue.length; }
  const todo=[...allLinks].filter(t=>!(t in S.aliases)&&!S.pages[t]); S.status='classify links '+todo.length;
  const W=4, per=Math.ceil(todo.length/W); let doneN=0;
  await Promise.all([...Array(W).keys()].map(async w=>{ const part=todo.slice(w*per,(w+1)*per); for(let i=0;i<part.length;i+=200){ const r=await classify(part.slice(i,i+200)); for(const t in r){S.aliases[t]=r[t][0]; S.linkTypes[r[t][0]]=r[t][1];} doneN+=Math.min(200,part.length-i); S.status='classify links '+doneN+'/'+todo.length; } }));
  Object.values(S.pages).forEach(p=>S.linkTypes[p.title]='Scientist');
  S.status='done'; S.done=true;
}catch(e){ S.err=String(e); S.status='error'; }
})();
window.exportDump=async()=>{ const S=window.S;
  const obj={pages:Object.values(S.pages),link_types:S.linkTypes,aliases:S.aliases};
  const blob=new Blob([JSON.stringify(obj)],{type:'application/json'});
  const gz=await new Response(blob.stream().pipeThrough(new CompressionStream('gzip'))).blob();
  const a=document.createElement('a'); a.href=URL.createObjectURL(gz); a.download='scinet_dump.json.gz'; document.body.appendChild(a); a.click(); a.remove();
  return (gz.size/1e6).toFixed(2)+' MB'; };
