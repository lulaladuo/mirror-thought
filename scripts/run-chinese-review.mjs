import fs from 'node:fs/promises';
const base=process.env.BASE_URL||'http://127.0.0.1:5174';
const cases=[
 {id:'family-expectation',label:'父母期待与感恩',turns:['父母总说是为我好。我知道他们关心我，但不按他们的安排做，就觉得自己不懂事。我又不想一直按他们的意思生活。','我觉得他们为我付出了很多，所以不听他们的，好像就是不感恩。但我不确定这个联系有没有道理。']},
 {id:'study-choice',label:'考研与现实期待',turns:['大家都说考研更有出路，可我其实不确定自己想不想继续读。我怕不考以后后悔，也怕只是因为别人都在考。','我现在还分不清，是我想要更好的机会，还是只是害怕落后。']},
 {id:'work-expression',label:'职场表达与讨好',turns:['开会时我经常先猜别人想听什么，再决定要不要说自己的想法。我知道这可能让我错过表达机会，但直接说又怕显得不合群。','我不确定这是在顾及合作，还是我太在意别人怎么看我。']}
];
async function call(payload){const r=await fetch(`${base}/api/chat`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});const j=await r.json();if(!r.ok)throw Error(JSON.stringify(j));return j}
const out=[];
for(const c of cases){let history=[],beliefs=[],tensions=[],assumptions=[],unclear=[];const rows=[];for(const text of c.turns){const r=await call({text,history,detectedBeliefs:beliefs,detectedTensions:tensions,detectedAssumptions:assumptions,unclearConcepts:unclear,turnCount:history.filter(m=>m.role==='user').length+1});rows.push({input:text,response:r});history.push({role:'user',content:text,responseType:undefined},{role:'assistant',content:r.message,responseType:r.response_type,mapping:r.mapping,question:r.question});beliefs=r.detected_beliefs;tensions=r.detected_tensions;assumptions=r.detected_assumptions;unclear=r.unclear_concepts;}out.push({id:c.id,label:c.label,turns:rows});}
await fs.writeFile('eval/chinese-context-real-review.json',JSON.stringify({generated_at:new Date().toISOString(),cases:out},null,2));
await fs.writeFile('eval/chinese-context-real-review.md',out.map(c=>`## ${c.label}\n\n${c.turns.map((x,i)=>`### 第${i+1}轮\n输入：${x.input}\n\n回应：${x.response.message}\n\n追问：${x.response.question??'无'}\n`).join('\n')}`).join('\n'));
console.log(`完成 ${out.length} 个场景测试，结果已保存到 eval/chinese-context-real-review.json 和 .md`);
