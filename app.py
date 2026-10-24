"""Dependency-free local web demo for TreeQA saved results and live inference."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from uuid import uuid4
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from demo_data import (
    DEFAULT_RESULTS_PATH,
    PROJECT_ROOT,
    answer_matches,
    calculate_metrics,
    live_configuration_issues,
    load_jsonl,
    record_status,
)
from live_evaluation import (
    MIN_LIVE_QUESTIONS,
    MAX_LIVE_QUESTIONS,
    append_run_event,
    load_live_runs,
    select_benchmark_sample,
    utc_now,
)


active_live_run: dict[str, Any] | None = None
active_live_run_lock = threading.Lock()
TWO_WIKI_RESULTS_PATH = PROJECT_ROOT / "result" / "2wiki_eval_seed42_n20.jsonl"
TWO_WIKI_SAMPLE_PATH = PROJECT_ROOT / "result" / "2wiki_sample_seed42_n20.json"
TWO_WIKI_DISPLAY_QUESTION_LIMIT = 3


def load_2wiki_evaluation() -> dict[str, Any]:
    import eval_2wiki

    records_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if TWO_WIKI_RESULTS_PATH.is_file():
        with TWO_WIKI_RESULTS_PATH.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if (
                    not isinstance(record, dict)
                    or not record.get("id")
                    or record.get("config") not in eval_2wiki.CONFIGS
                ):
                    raise ValueError(f"Invalid 2Wiki result at line {line_number}.")
                records_by_key[(str(record["id"]), record["config"])] = record

    records = list(records_by_key.values())
    sample_size = 20
    if TWO_WIKI_SAMPLE_PATH.is_file():
        sample = json.loads(TWO_WIKI_SAMPLE_PATH.read_text(encoding="utf-8"))
        if not isinstance(sample, list):
            raise ValueError("Saved 2Wiki sample must contain a JSON list.")
        sample_size = min(TWO_WIKI_DISPLAY_QUESTION_LIMIT, len(sample))
        selected_ids = {str(row["id"]) for row in sample[:sample_size]}
        records = [record for record in records if str(record["id"]) in selected_ids]
    else:
        sample_size = TWO_WIKI_DISPLAY_QUESTION_LIMIT
    successful = [record for record in records if not record.get("error")]

    summary = {}
    for config in eval_2wiki.CONFIGS:
        rows = [row for row in successful if row.get("config") == config]
        summary[config] = {
            "n": len(rows),
            "hits_at_1": eval_2wiki.format_mean_ci(
                [float(row["hits_at_1"]) for row in rows], binary=True
            ),
            "f1": eval_2wiki.format_mean_ci([float(row["f1"]) for row in rows]),
            "containment": eval_2wiki.format_mean_ci(
                [float(row["containment"]) for row in rows], binary=True
            ),
            "tokens": eval_2wiki.format_mean_ci(
                [float(row["total_tokens"]) for row in rows]
            ),
            "seconds": eval_2wiki.format_mean_ci(
                [float(row["wall_time_seconds"]) for row in rows]
            ),
            "tree_updates": eval_2wiki.format_mean_ci(
                [float(row["tree_updates"]) for row in rows]
            ),
        }

    paired_difference = {}
    for metric in ("hits_at_1", "f1"):
        mean, low, high = eval_2wiki.paired_bootstrap_difference(successful, metric)
        paired_difference[metric] = f"{mean:.3f} [{low:.3f}, {high:.3f}]"
    paired_n = len({
        row["id"] for row in successful if row.get("config") == "dr"
    } & {
        row["id"] for row in successful if row.get("config") == "treeqa_full"
    })

    complete_questions = 0
    ids_by_config = {
        config: {row["id"] for row in successful if row.get("config") == config}
        for config in eval_2wiki.CONFIGS
    }
    if ids_by_config:
        complete_questions = len(set.intersection(*ids_by_config.values()))

    target_tasks = sample_size * len(eval_2wiki.CONFIGS)
    return {
        "records": records,
        "summary": summary,
        "paired_difference": paired_difference,
        "paired_n": paired_n,
        "sample_size": sample_size,
        "display_limit": TWO_WIKI_DISPLAY_QUESTION_LIMIT,
        "seed": 42,
        "target_tasks": target_tasks,
        "completed_tasks": len(successful),
        "complete_questions": complete_questions,
        "failed_attempts": sum(bool(record.get("error")) for record in records),
    }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TreeQA: Logic Tree Decomposition Framework for Multi-Hop Question Answering LLM-RAG System</title>
<style>
:root{--navy:#132d4b;--blue:#176b87;--cyan:#dff4f7;--ink:#20334b;--muted:#63748a;--paper:#f3f6fa;--line:#dce4ed;--white:#fff;--green:#18794e;--red:#b42318;--gold:#986900}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 "Segoe UI",Arial,sans-serif}
header{background:linear-gradient(115deg,var(--navy),var(--blue));color:white;padding:25px max(24px,calc((100% - 1250px)/2))}
header h1{font-size:30px;margin:0 0 4px;letter-spacing:.2px}header p{margin:0;color:#dceefa}
main{max-width:1250px;margin:22px auto;padding:0 20px 48px}
.nav{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}.nav button{border:1px solid var(--line);border-radius:9px;background:white;padding:10px 15px;color:var(--navy);font-weight:650;cursor:pointer}.nav button.active{color:white;background:var(--navy);border-color:var(--navy)}
.panel{background:white;border:1px solid var(--line);border-radius:13px;padding:20px;margin-bottom:16px;box-shadow:0 2px 8px #15304b0a}
h2{font-size:21px;color:var(--navy);margin:0 0 6px}h3{font-size:17px;color:var(--navy);margin:20px 0 8px}h4{font-size:15px;color:var(--navy);margin:16px 0 6px}.muted,.subtle{color:var(--muted)}.subtle{font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:11px;margin:16px 0}.kpi{padding:13px 15px;border:1px solid var(--line);background:#fff;border-radius:11px;min-height:82px}.kpi .label{font-size:12px;color:var(--muted)}.kpi .value{font-size:22px;font-weight:700;color:var(--navy);margin-top:3px}
.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin:14px 0}.field{display:flex;flex-direction:column;gap:4px;min-width:170px;flex:1}label{font-weight:600;color:#42546a}input,select,textarea{border:1px solid #cbd5e1;border-radius:8px;padding:9px 10px;font:inherit;background:white;color:var(--ink)}textarea{width:100%;resize:vertical}
button.primary,.button{border:0;border-radius:8px;padding:10px 15px;background:var(--blue);color:white;font:inherit;font-weight:650;cursor:pointer;text-decoration:none;display:inline-block}button.primary:hover,.button:hover{background:var(--navy)}button:disabled{opacity:.48;cursor:not-allowed}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:9px}table{width:100%;border-collapse:collapse;font-size:13px;min-width:750px}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #e9eef4;vertical-align:top}th{background:#eef3f8;color:var(--navy);position:sticky;top:0}tr:last-child td{border-bottom:0}.row-select{border:0;background:transparent;color:#135f80;text-align:left;text-decoration:underline;cursor:pointer;font:inherit;padding:0}
.badge{display:inline-block;border-radius:20px;padding:3px 9px;font-size:12px;font-weight:700;background:#eaf0f6;color:#324861}.badge.good{background:#e4f5ec;color:var(--green)}.badge.bad{background:#fde9e7;color:var(--red)}.badge.missing{background:#fff4d6;color:var(--gold)}
.answer{padding:14px 16px;border-left:4px solid var(--blue);border-radius:7px;background:#f0f8fc;white-space:pre-wrap}.tree details{border:1px solid var(--line);border-radius:9px;padding:10px 12px;margin:9px 0;background:#fbfdff}.tree details details{margin:9px 0 0 16px}.tree summary{cursor:pointer;color:var(--navy);font-weight:650}.evidence{padding:11px 12px;background:#f6f8fb;border-radius:8px;margin:8px 0}.evidence p{white-space:pre-wrap;margin:7px 0}.evidence a{color:#12668a}
pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:500px;overflow:auto;background:#f5f7fa;padding:14px;border-radius:9px;font-size:12px}.chart{display:flex;align-items:end;gap:7px;height:190px;border-bottom:1px solid var(--line);padding:8px 4px 0;overflow-x:auto}.bar-col{min-width:46px;flex:1;display:flex;flex-direction:column;align-items:center;justify-content:end;height:100%;gap:5px}.bar{width:min(36px,70%);min-height:2px;background:linear-gradient(#43a5bd,#176b87);border-radius:5px 5px 0 0}.bar-count{font-size:11px;color:var(--muted)}.bar-label{font-size:10px;color:var(--muted);white-space:nowrap;max-width:85px;overflow:hidden;text-overflow:ellipsis}
.pipeline{display:grid;gap:9px}.stage{padding:13px 15px;border:1px solid var(--line);border-radius:10px;background:#fbfdff}.stage strong{color:var(--navy)}.stage:not(:last-child):after{content:"↓";display:block;text-align:center;color:var(--blue);font-size:20px;font-weight:700;height:19px;margin-bottom:-24px}
.alert{border-radius:9px;padding:12px 14px;margin:12px 0}.alert.info{background:#eaf4fb;color:#174b70}.alert.warn{background:#fff4d6;color:#775600}.alert.error{background:#fde9e7;color:#8d2118}.alert.ok{background:#e4f5ec;color:#176344}
.split{display:grid;grid-template-columns:1fr 1fr;gap:15px}@media(max-width:760px){.split{grid-template-columns:1fr}main{padding:0 12px 32px}.panel{padding:15px}}
.hidden{display:none!important}.footer{color:var(--muted);font-size:12px;margin:20px 2px}
</style>
</head>
<body>
<header><h1>TreeQA: Logic Tree Decomposition Framework for Multi-Hop Question Answering LLM-RAG System</h1></header>
<main>
<div class="nav" role="tablist">
 <button class="active" data-tab="live-evaluation">Test Results</button>
 
 <button data-tab="live">Live Inference</button>
</div>
<section id="live-evaluation" class="page hidden"></section>
<section id="earlier-results" class="page hidden"></section>
<section id="live" class="page hidden"></section>
<p class="footer">Test Results are generated at run time. Earlier Results contain previously recorded benchmark outputs.</p>
</main>
<script id="earlier-records-data" type="application/json">__EARLIER_RECORDS__</script>
<script id="earlier-metrics-data" type="application/json">__EARLIER_METRICS__</script>
<script>
"use strict";
const earlierRecords=JSON.parse(document.getElementById("earlier-records-data").textContent);
const earlierMetrics=JSON.parse(document.getElementById("earlier-metrics-data").textContent);
const $=(selector,root=document)=>root.querySelector(selector);
const make=(tag,text="",className="")=>{const node=document.createElement(tag);if(text!==undefined&&text!==null)node.textContent=String(text);if(className)node.className=className;return node};
function add(parent,tag,text="",className=""){const node=make(tag,text,className);parent.appendChild(node);return node}
function fmt(value,digits=1){return typeof value==="number"&&Number.isFinite(value)?value.toLocaleString(undefined,{maximumFractionDigits:digits}):"Unavailable"}
function statusBadge(status){return make("span",status,"badge "+(status==="Successful"?"good":status==="Failed"?"bad":status==="Missing answer"?"missing":""))}
function kpi(parent,label,value){const card=add(parent,"div","","kpi");add(card,"div",label,"label");add(card,"div",value,"value")}
function panel(parent,title,subtitle=""){const box=add(parent,"div","","panel");add(box,"h2",title);if(subtitle)add(box,"p",subtitle,"subtle");return box}
function question(record){return String(record.question||"Question unavailable in this record.")}
function answer(record){if(record.final_answer!==undefined&&record.final_answer!==null)return String(record.final_answer);const data=payload(record);return String(data&&data.answer||"")}
function payload(record){let value=record.processed_answer;if(typeof value==="string"){try{value=JSON.parse(value)}catch{return null}}return value&&typeof value==="object"?value:null}
function gold(record){const value=record.original_answer;if(value===undefined||value===null)return[];return(Array.isArray(value)?value:[value]).filter(v=>v!==null&&String(v).trim()!=="").map(String)}
function normalize(value){return String(value??"").toLowerCase().replace(/[\u0021-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e]/g,"").replace(/\b(a|an|the)\b/g," ").trim().split(/\s+/).filter(Boolean).join(" ")}
function match(record,exact=false){if(sourceStatus(record)!=="Successful")return false;const prediction=normalize(answer(record)),answers=gold(record).map(normalize).filter(Boolean);if(!prediction||!answers.length)return false;if(exact)return answers.includes(prediction);return answers.some(value=>prediction.includes(value))}
function sourceStatus(record){if(record.fix_count===-1||(payload(record)||{}).status==="failed")return"Failed";return answer(record).trim()?"Successful":"Missing answer"}
function metric(record,key){return typeof record[key]==="number"?record[key]:null}
function evidence(container,refs,path){if(!refs||typeof refs!=="object"||!Object.keys(refs).length){add(container,"p","No evidence references were stored for this node.","subtle");return}for(const[source,values]of Object.entries(refs)){add(container,"strong",source[0].toUpperCase()+source.slice(1));for(const[index,raw]of(Array.isArray(values)?values:[values]).entries()){const text=String(raw),box=add(container,"div","","evidence");if(source.toLowerCase()==="wikipedia"){const split=text.split("||"),title=split.length>1?split[0]:text,passage=split.length>1?split.slice(1).join("||"):text;add(box,"p",passage);const article=title.trim().replace(/_\d+_\d+$/,"").replaceAll(" ","_");if(article){const link=add(box,"a","Open Wikipedia source");link.href="https://en.wikipedia.org/wiki/"+encodeURIComponent(article).replace(/%2F/g,"/");link.target="_blank";link.rel="noopener noreferrer"}}else{add(box,"p",text);const qid=text.match(/\bQ\d+\b/),pid=text.match(/\bP\d+\b/),url=qid?"https://www.wikidata.org/wiki/"+qid[0]:pid?"https://www.wikidata.org/wiki/Property:"+pid[0]:"https://www.wikidata.org/w/index.php?search="+encodeURIComponent(text);const link=add(box,"a",qid?"Open Wikidata entity "+qid[0]:pid?"Open Wikidata property "+pid[0]:"Search Wikidata for this reference");link.href=url;link.target="_blank";link.rel="noopener noreferrer"}box.dataset.key=path+"-"+index}}}
function treeNode(parent,node,path,depth=0){if(!node||typeof node!=="object")return;const details=add(parent,"details","","");if(depth===0)details.open=true;const title=node.sub_question||"Reasoning step "+path;add(details,"summary","Step "+path+": "+title);if(node.sub_question){add(details,"p","Sub-question: "+node.sub_question)}if(node.hypothesis_answer)add(details,"p","Stored hypothesis: "+node.hypothesis_answer);const verification=node.verification_status||node.status;add(details,"p",verification?"Verification status: "+String(verification):"Verification status was not stored.","subtle");if(node.correction_reason)add(details,"p","Correction reason: "+node.correction_reason);if(node.verification_reason)add(details,"p","Verification note: "+node.verification_reason);evidence(details,node.ref,path);if(Array.isArray(node.children)){node.children.forEach((child,index)=>treeNode(details,child,path+"."+(index+1),depth+1))}}
function downloadFile(name,text,type){const blob=new Blob([text],{type}),url=URL.createObjectURL(blob),link=make("a");link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function renderModuleTrace(parent,tree){
 const trace=tree&&tree.execution_trace||{},logic=tree&&tree.logic_tree||{};
 const module1=add(parent,"details");module1.open=true;add(module1,"summary","Module 1 · Logic Tree Construction");add(module1,"p","The initial LLM-proposed tree is captured before evidence checking or corrections.","subtle");
 const initialChildren=trace.initial_tree&&Array.isArray(trace.initial_tree.children)?trace.initial_tree.children:[];
 if(initialChildren.length){const initial=add(module1,"div","","tree");initialChildren.forEach((node,index)=>treeNode(initial,node,String(index+1)))}
 else add(module1,"p","The initial tree was not included in this run.","subtle");
 const events=Array.isArray(trace.events)?trace.events:[];
 const retrievals=events.filter(event=>event.type==="retrieval"),checks=events.filter(event=>event.type==="verification"),corrections=events.filter(event=>event.type==="correction");
 const module2=add(parent,"details");add(module2,"summary","Module 2 · Knowledge Retrieval ("+retrievals.length+" retrieval step(s))");
 if(!retrievals.length)add(module2,"p","No retrieval trace was captured.","subtle");
 retrievals.forEach((event,index)=>{
  const section=add(module2,"details");add(section,"summary","Retrieval "+(index+1)+(event.path?" · node "+event.path.join("."):""));
  if(event.query)add(section,"p","Retrieval query: "+event.query);
  add(section,"p","Extracted entity mentions: "+((event.extracted_entities||[]).join(", ")||"None recorded."));
  add(section,"h4","Wikidata candidates");
  const candidates=event.wikidata_candidates||[];
  if(candidates.length){candidates.forEach(item=>add(section,"p",[item.text,item.wikidata,item.definition].filter(Boolean).join(" · ")))}
  else add(section,"p","No Wikidata candidates were returned.","subtle");
  add(section,"p","Selected entity IDs: "+((event.selected_entity_ids||[]).join(", ")||"None."));
  if((event.graph_facts||[]).length){add(section,"h4","Retrieved graph facts");add(section,"pre",JSON.stringify(event.graph_facts,null,2))}
  const passages=event.wikipedia_passages||[];
  if(passages.length){add(section,"h4","Retrieved Wikipedia passages");passages.forEach((passage,passageIndex)=>{const item=add(section,"div","","evidence");add(item,"strong",passage.title||passage.id||"Wikipedia passage "+(passageIndex+1));if(typeof passage.similarity==="number")add(item,"p","Local lexical similarity: "+passage.similarity.toFixed(3),"subtle");add(item,"p",passage.content||"")})}
 });
 const module3=add(parent,"details");module3.open=true;add(module3,"summary","Module 3 · Self Correction and Reasoning");
 add(module3,"p","Verification decisions are shown per node. Corrected hypotheses are labelled pending until a later check verifies them; corrections are not automatically treated as golden.","subtle");
 const timeline=add(module3,"ol");
 events.filter(event=>event.type==="verification"||event.type==="correction").forEach(event=>{
  const item=add(timeline,"li");
  const path=event.path&&event.path.length?event.path.join("."):"root";
  if(event.type==="verification")add(item,"p","Node "+path+" · "+event.status+(event.reason?" · "+event.reason:""));
  else{
   add(item,"p","Node "+path+" · corrected after conflicting evidence: "+(event.reason||"No reason recorded."));
   const before=add(item,"details");add(before,"summary","Before correction");add(before,"pre",JSON.stringify(event.before,null,2));
   const after=add(item,"details");add(after,"summary","Proposed corrected subtree (awaiting verification)");add(after,"pre",JSON.stringify(event.after,null,2));
  }
 });
 if(!timeline.children.length)add(module3,"p","No verification or correction events were recorded.","subtle");
 const verifiedPaths=[];
 function collectVerified(nodes,path,verified){
  (nodes||[]).forEach((node,index)=>{
   const nextPath=path.concat(index+1),currentVerified=verified&&node.verification_status==="verified";
   const children=Array.isArray(node.children)?node.children:[];
   if(!children.length){if(currentVerified)verifiedPaths.push(nextPath.join("."));return}
   collectVerified(children,nextPath,currentVerified);
  });
 }
 collectVerified(logic.children,[],true);
 add(module3,"h4","Golden tree concept · fully verified paths only");
 add(module3,"p",verifiedPaths.length?verifiedPaths.join(", "):"No complete leaf-to-root path has explicit verified status in this run. The generated tree and final answer are not themselves proof of correctness.","subtle");
 add(module3,"h4","Final annotated reasoning tree");
 if(Array.isArray(logic.children)&&logic.children.length){const finalTree=add(module3,"div","","tree");logic.children.forEach((node,index)=>treeNode(finalTree,node,String(index+1)))}
 else add(module3,"p","No final reasoning tree was returned.","subtle");
}
let liveEvalPoll=null,liveEvalSelectedRun=null,liveEvalRuns=[];
function render2WikiEvaluation(parent,data){
 const section=panel(parent,"2Wiki comparative pilot","Paired comparison of the first three questions in the fixed 20-question seed-42 sample. This is separate from the live QALD test below.");
 const controls=add(section,"div","","controls"),refresh=add(controls,"button","Refresh 2Wiki results","primary");
 refresh.onclick=refreshLiveEvaluation;
 if(data.error){add(section,"div",data.error,"alert error");return}
 const records=data.records||[],summary=data.summary||{},cards=add(section,"div","","grid");
 kpi(cards,"Progress",data.completed_tasks+" / "+data.target_tasks+" runs");
 kpi(cards,"Questions with all 3 configs",data.complete_questions+" / "+data.sample_size);
 kpi(cards,"Sample / seed",String(data.sample_size)+" / "+String(data.seed));
 if(data.target_tasks&&data.completed_tasks<data.target_tasks){
  add(section,"div","Pilot is incomplete. Resume only these three questions with: python eval_2wiki.py --run --n 20 --limit-questions 3 --seed 42","alert info");
 }else if(data.target_tasks){
  add(section,"div","All configured question/model runs are present. Review failed attempts and quota errors before treating this as final.","alert ok");
 }else{
  add(section,"p","No 2Wiki comparative evaluation file is present yet. Start or resume the pilot using the command shown above.","subtle");
 }
 add(section,"h3","Configuration summary · successful records only");
 const summaryWrap=add(section,"div","","table-wrap"),summaryTable=add(summaryWrap,"table"),summaryHead=add(summaryTable,"thead"),summaryHeader=add(summaryHead,"tr");
 ["Configuration","N","Hits@1 (95% CI)","F1 (95% CI)","Containment (95% CI)","Mean tokens (95% CI)","Mean seconds (95% CI)","Mean tree updates (95% CI)"].forEach(label=>add(summaryHeader,"th",label));
 const summaryBody=add(summaryTable,"tbody");
 ["dr","treeqa_no_correction","treeqa_full"].forEach(config=>{
  const item=summary[config]||{},row=add(summaryBody,"tr");
  [config,String(item.n||0),item.hits_at_1||"N/A",item.f1||"N/A",item.containment||"N/A",item.tokens||"N/A",item.seconds||"N/A",item.tree_updates||"N/A"].forEach(value=>add(row,"td",value));
 });
 const paired=data.paired_difference||{};
 add(section,"p","Paired full TreeQA − DR differences (N="+String(data.paired_n||0)+"): Hits@1 "+(paired.hits_at_1||"N/A")+" · F1 "+(paired.f1||"N/A"),"subtle");
 add(section,"p","Hits@1 is normalized exact match for the single generated answer. These pilot estimates are small-sample diagnostics, not paper-scale benchmark claims.","subtle");
 if(records.length){
  add(section,"h3","Question-level results (latest attempt per configuration)");
  const wrap=add(section,"div","","table-wrap"),table=add(wrap,"table"),head=add(table,"thead"),header=add(head,"tr");
  ["ID","Configuration","Question","Gold","Prediction","Hits@1","F1","Tokens","Seconds","Updates","Status"].forEach(label=>add(header,"th",label));
  const body=add(table,"tbody");
  records.forEach(record=>{
   const row=add(body,"tr");
   [record.id,record.config,record.question,record.gold,record.short_answer,record.error?"N/A":(record.hits_at_1?"Yes":"No"),record.error?"N/A":fmt(record.f1,3),fmt(record.total_tokens,0),fmt(record.wall_time_seconds,1),String(record.tree_updates??0),record.error||"OK"].forEach(value=>add(row,"td",value));
  });
  add(section,"h3","Per-question trace");
  const traceSelect=add(section,"select");
  records.forEach((record,index)=>{const option=add(traceSelect,"option",record.config+" · "+record.id+" · "+String(record.question||"").slice(0,90));option.value=String(index)});
  const trace=add(section,"pre");
  const showTrace=()=>{trace.textContent=JSON.stringify(records[Number(traceSelect.value)],null,2)};
  traceSelect.onchange=showTrace;showTrace();
  const download=add(section,"button","Download 2Wiki results as JSON","primary");
  download.onclick=()=>downloadFile("treeqa-2wiki-comparative-pilot.json",JSON.stringify(data,null,2),"application/json");
 }
}
async function refreshLiveEvaluation(){
 try{
  const [response,twoWikiResponse]=await Promise.all([
   fetch("/api/live-evaluations",{cache:"no-store"}),
   fetch("/api/2wiki-evaluation",{cache:"no-store"})
  ]);
  const data=await response.json(),twoWiki=await twoWikiResponse.json();
  if(!response.ok)throw new Error(data.error||"Could not load live evaluation runs.");
  if(!twoWikiResponse.ok)twoWiki.error=twoWiki.error||"Could not load the 2Wiki evaluation.";
  liveEvalRuns=(data.runs||[]).filter(run=>run.run_type==="sample"&&run.status!=="cancelled");
  renderLiveEvaluation(data,twoWiki);
  if(data.active_run&&data.active_run.status==="running"){
   if(!liveEvalPoll)liveEvalPoll=setInterval(refreshLiveEvaluation,2500);
  }else if(liveEvalPoll){clearInterval(liveEvalPoll);liveEvalPoll=null}
 }catch(error){
  const root=$("#live-evaluation");root.replaceChildren();
  add(root,"div",error.message,"alert error");
 }
}
function renderLive(){
 const root=$("#live");root.replaceChildren();
 const box=panel(root,"Live Inference","Enter a question to generate a fresh answer and inspect its reasoning trace.");
 const area=add(box,"textarea");area.rows=4;area.placeholder="Enter a multi-hop question…";
 const run=add(box,"button","Run live inference","primary"),result=add(box,"div");
 run.onclick=async()=>{
  if(!area.value.trim())return;
  run.disabled=true;run.textContent="Running…";result.replaceChildren();
  try{
   const response=await fetch("/api/live",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:area.value.trim()})});
   const data=await response.json();result.replaceChildren();
   if(!response.ok||!data.success){add(result,"div",data.error||"Live inference failed.","alert error");return}
   add(result,"div","Completed · "+fmt(data.metrics.total_processing_time,2)+" s · "+fmt(data.metrics.total_tokens,0)+" tokens · "+data.fix_count+" correction(s)","alert ok");
   add(result,"h3","Live answer");add(result,"div",data.metrics.final_answer||"No final answer was returned.","answer");
   if(data.tree)renderModuleTrace(result,data.tree);
  }catch(error){
   result.replaceChildren();add(result,"div","Live request failed: "+error.message,"alert error");
  }finally{
   run.disabled=false;run.textContent="Run live inference";
  }
 };
}
function renderEarlierResults(){
 const root=$("#earlier-results");root.replaceChildren();
 const box=panel(root,"Earlier Results","Previously recorded benchmark outputs and evaluation summary.");
 const cards=add(box,"div","","grid"),metrics=earlierMetrics;
 kpi(cards,"Benchmark questions",metrics.record_count);
 kpi(cards,"Successful",metrics.successful_count);
 kpi(cards,"Failed / missing",metrics.failed_count+" / "+metrics.missing_answer_count);
 kpi(cards,"Answer match",metrics.answer_match_rate===null?"N/A":(100*metrics.answer_match_rate).toFixed(1)+"%");
 kpi(cards,"Exact answer match",metrics.exact_match_rate===null?"N/A":(100*metrics.exact_match_rate).toFixed(1)+"%");
 kpi(cards,"Mean / median runtime",metrics.latency_mean===null?"N/A":fmt(metrics.latency_mean,1)+" / "+fmt(metrics.latency_median,1)+" s");
 kpi(cards,"Mean / median tokens",metrics.token_mean===null?"N/A":fmt(metrics.token_mean,0)+" / "+fmt(metrics.token_median,0));
 kpi(cards,"Mean corrections",metrics.correction_mean===null?"N/A":fmt(metrics.correction_mean,2));
 add(box,"h3","Question-level results");
 const wrap=add(box,"div","","table-wrap"),table=add(wrap,"table"),thead=add(table,"thead"),header=add(thead,"tr");
 ["ID","Status","Question","Reference answer","Prediction","Corrections","Runtime","Tokens","Match","Exact"].forEach(label=>add(header,"th",label));
 const body=add(table,"tbody");
 earlierRecords.forEach(record=>{
  const row=add(body,"tr");
  [record.id,record.status,record.question,record.gold_answers.join("; "),record.final_answer||"",String(record.fix_count??"N/A"),record.runtime===null?"N/A":fmt(record.runtime,1)+" s",record.tokens===null?"N/A":fmt(record.tokens,0),record.containment_match?"Yes":"No",record.exact_match?"Yes":"No"].forEach(value=>add(row,"td",value));
 });
}
function renderLiveEvaluation(data,twoWiki){
 const root=$("#live-evaluation");root.replaceChildren();
 const box=panel(root,"Test Results","Fresh model predictions are generated for each benchmark question in this run.");
 add(box,"p","Benchmark reference answers are used to calculate answer-match metrics.","subtle");
 render2WikiEvaluation(box,twoWiki||{error:"Could not load 2Wiki results."});
 const controls=add(box,"div","","controls"),countField=add(controls,"div","","field");
 add(countField,"label","Questions per run ("+String(data.min_questions||1)+"-"+String(data.max_questions||25)+")");
 const count=add(countField,"input");count.type="number";count.min=String(data.min_questions||1);count.max=String(data.max_questions||25);count.step="1";count.value="5";
 const start=add(controls,"button","Run test","primary");
 const active=data.active_run;
 if(active&&active.status==="running"){start.disabled=true;add(box,"div","Evaluation in progress: "+active.completed_count+" / "+active.target_count+" questions completed.","alert info")}
 start.onclick=async()=>{
  start.disabled=true;start.textContent="Starting…";
  try{
   const response=await fetch("/api/live-evaluations",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({count:Number(count.value)})});
   const result=await response.json();
   if(!response.ok||!result.success)throw new Error(result.error||"Could not start evaluation.");
   liveEvalSelectedRun=result.run_id;
   await refreshLiveEvaluation();
  }catch(error){add(box,"div",error.message,"alert error");start.disabled=false;start.textContent="Run test"}
 };
 if(data.issues&&data.issues.length){const warning=add(box,"details");add(warning,"summary","Evaluation log issues");data.issues.forEach(issue=>add(warning,"p",issue))}
 if(!liveEvalRuns.length){add(box,"p","No test runs yet. Start a run to populate these results.","subtle");return}
 add(box,"h3","Test runs");
 const runSelect=add(box,"select");
 liveEvalRuns.forEach(run=>{const option=add(runSelect,"option",run.started_at+" · "+run.status+" · "+run.completed_count+"/"+run.target_count);option.value=run.run_id;if(run.run_id===(liveEvalSelectedRun||liveEvalRuns[0].run_id))option.selected=true});
 liveEvalSelectedRun=runSelect.value;runSelect.onchange=()=>{liveEvalSelectedRun=runSelect.value;renderLiveEvaluation(data)};
 const selected=liveEvalRuns.find(run=>run.run_id===liveEvalSelectedRun)||liveEvalRuns[0];
 const metrics=selected.metrics||{},cards=add(box,"div","","grid");
 kpi(cards,"Questions completed",selected.completed_count+" / "+selected.target_count);
 kpi(cards,"Successful / failed",metrics.successful_count+" / "+metrics.failed_count);
 kpi(cards,"Containment match",metrics.answer_match_rate===null?"N/A":(100*metrics.answer_match_rate).toFixed(1)+"%");
 kpi(cards,"Hits@1 (normalized exact)",metrics.exact_match_rate===null?"N/A":(100*metrics.exact_match_rate).toFixed(1)+"%");
 kpi(cards,"Mean / median runtime",metrics.latency_mean===null?"N/A":fmt(metrics.latency_mean,1)+" / "+fmt(metrics.latency_median,1)+" s");
 kpi(cards,"Mean / median tokens",metrics.token_mean===null?"N/A":fmt(metrics.token_mean,0)+" / "+fmt(metrics.token_median,0));
 kpi(cards,"Mean corrections",metrics.correction_mean===null?"N/A":fmt(metrics.correction_mean,2));
 add(box,"p","Hits@1 treats the generated answer as the single top-ranked answer and checks normalized exact match against the reference answers. Precision@3 is not available because this run does not produce a ranked list of three answers with matching gold labels.","subtle");
 const rows=selected.results||[];
 if(rows.length){
  add(box,"h3","Question-level results");
  const wrap=add(box,"div","","table-wrap"),table=add(wrap,"table"),thead=add(table,"thead"),tr=add(thead,"tr");
  ["Benchmark ID","Status","Question","Gold answer","Live answer","Corrections","Runtime","Tokens"].forEach(label=>add(tr,"th",label));
  const body=add(table,"tbody");
  rows.forEach(record=>{const row=add(body,"tr");[String(record.id||""),sourceStatus(record),question(record),gold(record).join("; "),answer(record),String(record.fix_count),fmt(record.total_processing_time,1)+" s",fmt(record.total_tokens,0)].forEach(value=>add(row,"td",value))});
  add(box,"h3","Live module outputs");
  const selectedQuestion=add(box,"select");
  rows.forEach((record,index)=>{const option=add(selectedQuestion,"option",record.id+" · "+question(record).slice(0,110));option.value=String(index)});
  const tracePanel=add(box,"div");
  const showTrace=()=>{tracePanel.replaceChildren();const record=rows[Number(selectedQuestion.value)],tree=payload(record)||{};add(tracePanel,"h4",question(record));add(tracePanel,"div",answer(record)||"No final answer.","answer");renderModuleTrace(tracePanel,tree)};
  selectedQuestion.onchange=showTrace;showTrace();
 }
 const exportButton=add(box,"button","Download run as JSON","primary");
 exportButton.onclick=()=>downloadFile("treeqa-live-run-"+selected.run_id+".json",JSON.stringify(selected,null,2),"application/json");
 const csvButton=add(box,"button","Download metrics as CSV","primary");csvButton.style.marginLeft="8px";
 csvButton.onclick=()=>{const fields=["id","status","question","gold_answers","prediction","containment_match","exact_match","fix_count","total_processing_time","total_tokens"],quote=value=>'"'+String(value??"").replaceAll('"','""')+'"',lines=[fields.join(",")];rows.forEach(record=>lines.push([record.id,sourceStatus(record),question(record),gold(record).join("; "),answer(record),match(record),match(record,true),record.fix_count,record.total_processing_time,record.total_tokens].map(quote).join(",")));downloadFile("treeqa-live-run-"+selected.run_id+".csv",lines.join("\r\n"),"text/csv")};
}
function switchTab(name){document.querySelectorAll(".page").forEach(page=>page.classList.toggle("hidden",page.id!==name));document.querySelectorAll(".nav button").forEach(button=>button.classList.toggle("active",button.dataset.tab===name));if(name==="live-evaluation")refreshLiveEvaluation();if(name==="earlier-results")renderEarlierResults();if(name==="live")renderLive()}
document.querySelectorAll(".nav button").forEach(button=>button.addEventListener("click",()=>switchTab(button.dataset.tab)));
switchTab("live-evaluation");
</script>
</body>
</html>"""


def render_page() -> bytes:
    records, _ = load_jsonl(DEFAULT_RESULTS_PATH)
    metrics = calculate_metrics(records)
    earlier_records = []
    for record in records:
        payload = record.get("processed_answer")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}
        if not isinstance(payload, dict):
            payload = {}
        final_answer = record.get("final_answer")
        if final_answer is None:
            final_answer = payload.get("answer")
        raw_gold = record.get("original_answer")
        if raw_gold is None:
            gold_answers = []
        elif isinstance(raw_gold, list):
            gold_answers = raw_gold
        else:
            gold_answers = [raw_gold]
        status = record_status(record)
        earlier_records.append({
            "id": str(record.get("id", "")),
            "status": status,
            "question": str(record.get("question", "")),
            "gold_answers": [str(value) for value in gold_answers if value is not None],
            "final_answer": str(final_answer or ""),
            "fix_count": record.get("fix_count"),
            "runtime": record.get("total_processing_time"),
            "tokens": record.get("total_tokens"),
            "containment_match": status == "Successful" and answer_matches(record),
            "exact_match": status == "Successful" and answer_matches(record, exact=True),
        })
    page = PAGE.replace("__EARLIER_RECORDS__", _safe_script_json(earlier_records), 1)
    page = page.replace("__EARLIER_METRICS__", _safe_script_json(metrics), 1)
    return page.encode("utf-8")


def _safe_script_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _execute_live_question(
    sample: dict[str, Any],
    run_id: str,
    run_index: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from inference import answerQuestion

        tree, fix_count, metrics = answerQuestion(sample["question"])
        if isinstance(tree, str):
            try:
                tree = json.loads(tree)
            except json.JSONDecodeError:
                tree = {"status": "failed", "error": "Inference returned invalid tree JSON."}
        if not isinstance(tree, dict):
            tree = {"status": "failed", "error": "Inference returned no reasoning tree."}
        if not isinstance(metrics, dict):
            metrics = {}
        answer = metrics.get("final_answer") or tree.get("answer")
        failed = tree.get("status") == "failed" or not answer
        if failed and not tree.get("error"):
            tree["status"] = "failed"
            tree["error"] = "Inference returned no final answer."
        return {
            "id": sample["id"],
            "question": sample["question"],
            "original_answer": sample.get("original_answer", []),
            "final_answer": answer,
            "fix_count": -1 if failed else fix_count,
            "total_processing_time": metrics.get(
                "total_processing_time", time.perf_counter() - started
            ),
            "total_tokens": metrics.get("total_tokens", 0),
            "logic_init_time": metrics.get("logic_init_time"),
            "self_adaptive_time": metrics.get("self_adaptive_time"),
            "final_reasoning_time": metrics.get("final_reasoning_time"),
            "logic_init_tokens": metrics.get("logic_init_tokens"),
            "self_adaptive_tokens": metrics.get("self_adaptive_tokens"),
            "final_reasoning_tokens": metrics.get("final_reasoning_tokens"),
            "processed_answer": tree,
            "run_id": run_id,
            "run_index": run_index,
            "run_type": "sample",
        }
    except Exception as error:
        return {
            "id": sample["id"],
            "question": sample["question"],
            "original_answer": sample.get("original_answer", []),
            "final_answer": None,
            "fix_count": -1,
            "total_processing_time": time.perf_counter() - started,
            "total_tokens": 0,
            "processed_answer": {
                "status": "failed",
                "error": str(error),
                "input_question": sample["question"],
            },
            "run_id": run_id,
            "run_index": run_index,
            "run_type": "sample",
        }


def _run_live_sample(run_id: str, samples: list[dict[str, Any]]) -> None:
    error = None
    try:
        for index, sample in enumerate(samples):
            record = _execute_live_question(sample, run_id, index)
            append_run_event({
                "event": "question_result",
                "run_id": run_id,
                "record": record,
            })
            with active_live_run_lock:
                if active_live_run and active_live_run.get("run_id") == run_id:
                    active_live_run["completed_count"] = index + 1
    except Exception as exception:
        error = str(exception)
    finally:
        status = "failed" if error else "complete"
        try:
            append_run_event({
                "event": "run_finished",
                "run_id": run_id,
                "status": status,
                "finished_at": utc_now(),
                "error": error,
            })
        finally:
            with active_live_run_lock:
                if active_live_run and active_live_run.get("run_id") == run_id:
                    active_live_run["status"] = status
                    active_live_run["error"] = error
                    active_live_run["finished_at"] = utc_now()


class DemoRequestHandler(BaseHTTPRequestHandler):
    server_version = "TreeQADemo/1.0"
    max_live_request_bytes = 20_000

    def log_message(self, format_string, *args):
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def _send_json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if path == "/api/live-evaluations":
            runs, issues = load_live_runs()
            with active_live_run_lock:
                active = dict(active_live_run) if active_live_run else None
            self._send_json(200, {
                "runs": runs,
                "issues": issues,
                "active_run": active,
                "min_questions": MIN_LIVE_QUESTIONS,
                "max_questions": MAX_LIVE_QUESTIONS,
            })
            return
        if path == "/api/2wiki-evaluation":
            try:
                self._send_json(200, load_2wiki_evaluation())
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self._send_json(500, {"error": f"Could not load 2Wiki results: {error}"})
            return
        if path not in {"/", "/index.html"}:
            self.send_error(404, "Not found")
            return
        body = render_page()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        global active_live_run
        path = urlparse(self.path).path
        if path not in {"/api/live", "/api/live-evaluations"}:
            self.send_error(404, "Not found")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"success": False, "error": "Invalid request length."})
            return
        if content_length <= 0 or content_length > self.max_live_request_bytes:
            self._send_json(413, {"success": False, "error": "Question request is empty or too large."})
            return
        try:
            request_data = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"success": False, "error": "Request body must be valid JSON."})
            return
        configuration_issues = live_configuration_issues()
        if configuration_issues:
            self._send_json(503, {
                "success": False,
                "error": "Live inference is not configured.",
                "issues": configuration_issues,
            })
            return

        if path == "/api/live-evaluations":
            count = request_data.get("count") if isinstance(request_data, dict) else None
            try:
                samples = select_benchmark_sample(count)
            except ValueError as error:
                self._send_json(400, {"success": False, "error": str(error)})
                return
            with active_live_run_lock:
                if active_live_run and active_live_run.get("status") == "running":
                    self._send_json(409, {
                        "success": False,
                        "error": "A live evaluation is already running.",
                    })
                    return
                run_id = str(uuid4())
                active_live_run = {
                    "run_id": run_id,
                    "status": "running",
                    "target_count": len(samples),
                    "completed_count": 0,
                    "started_at": utc_now(),
                }
                try:
                    append_run_event({
                        "event": "run_started",
                        "run_id": run_id,
                        "run_type": "sample",
                        "started_at": active_live_run["started_at"],
                        "target_count": len(samples),
                        "sample_ids": [sample["id"] for sample in samples],
                    })
                except OSError as error:
                    active_live_run = None
                    self._send_json(500, {
                        "success": False,
                        "error": f"Could not persist the live evaluation: {error}",
                    })
                    return
            worker = threading.Thread(
                target=_run_live_sample,
                args=(run_id, samples),
                daemon=True,
                name=f"treeqa-live-{run_id[:8]}",
            )
            worker.start()
            self._send_json(202, {
                "success": True,
                "run_id": run_id,
                "target_count": len(samples),
                "sample_ids": [sample["id"] for sample in samples],
            })
            return

        question = request_data.get("question") if isinstance(request_data, dict) else None
        if not isinstance(question, str) or not question.strip():
            self._send_json(400, {"success": False, "error": "Enter a non-empty question."})
            return

        try:
            single_run_id = str(uuid4())
            append_run_event({
                "event": "run_started",
                "run_id": single_run_id,
                "run_type": "single",
                "started_at": utc_now(),
                "target_count": 1,
                "sample_ids": [],
            })
            record = _execute_live_question(
                {
                    "id": "live_question",
                    "question": question.strip(),
                    "original_answer": [],
                },
                single_run_id,
                0,
            )
            append_run_event({
                "event": "question_result",
                "run_id": single_run_id,
                "record": record,
            })
            append_run_event({
                "event": "run_finished",
                "run_id": single_run_id,
                "status": "complete" if record.get("final_answer") else "failed",
                "finished_at": utc_now(),
                "error": None if record.get("final_answer") else record["processed_answer"].get("error"),
            })
        except OSError as error:
            self._send_json(502, {
                "success": False, "error": f"Could not persist the live result: {error}",
            })
            return
        tree = record.get("processed_answer")
        metrics = {
            key: record.get(key)
            for key in (
                "final_answer",
                "total_processing_time",
                "total_tokens",
                "logic_init_time",
                "self_adaptive_time",
                "final_reasoning_time",
                "logic_init_tokens",
                "self_adaptive_tokens",
                "final_reasoning_tokens",
            )
        }
        if not record.get("final_answer"):
            self._send_json(502, {
                "success": False,
                "error": str(tree.get("error", "Live inference failed.")),
            })
            return
        self._send_json(200, {
            "success": True,
            "question": question.strip(),
            "tree": tree if isinstance(tree, dict) else None,
            "fix_count": record["fix_count"],
            "metrics": metrics,
        })


def configure_console_encoding(stdout=None, stderr=None) -> None:
    for stream in (stdout or sys.stdout, stderr or sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def main() -> None:
    configure_console_encoding()
    parser = argparse.ArgumentParser(description="Run the local TreeQA research demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only).")
    parser.add_argument("--port", type=int, default=8501, help="Port to serve the app on.")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DemoRequestHandler)
    print(f"TreeQA demo is running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping TreeQA demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
