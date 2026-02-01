async function jget(url){
  const r = await fetch(url);
  return await r.json();
}
async function jpost(url, body){
  const r = await fetch(url, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(body)
  });
  return await r.json();
}

const el = (id)=>document.getElementById(id);

async function startPipeline(){
  const keyword = el("kw").value.trim();
  const amount = parseInt(el("amt").value || "3", 10);
  if(!keyword) return;

  el("pipeOut").textContent = "starting...";
  const data = await jpost("/pipeline/external/scrape", {keyword, amount});
  el("pipeOut").textContent = JSON.stringify(data, null, 2);

  if(data.job_id){
    alert("job_id: " + data.job_id + " (เปิด Workflow UI แล้วใส่ job_id)");
  }
}

async function ask(){
  const question = el("q").value.trim();
  const top_k = parseInt(el("topk").value || "8", 10);
  if(!question) return;

  el("answer").textContent = "thinking...";
  const data = await jpost("/ask", {question, top_k});
  if(data.status !== "ok"){
    el("answer").textContent = JSON.stringify(data, null, 2);
    return;
  }
  el("answer").textContent = data.answer || "";
  el("chunks").textContent = JSON.stringify(data.chunks || [], null, 2);
  el("tavily").textContent = JSON.stringify(data.tavily_results || [], null, 2);
}

async function preview(){
  const namespace = el("ns").value;
  const source_type = el("stype").value.trim();
  const limit = parseInt(el("limit").value || "20", 10);

  el("preview").textContent = "loading...";
  const data = await jget(`/rag/preview?namespace=${encodeURIComponent(namespace)}&source_type=${encodeURIComponent(source_type)}&limit=${limit}`);
  el("preview").textContent = JSON.stringify(data, null, 2);
}

el("btnStart").addEventListener("click", startPipeline);
el("btnAsk").addEventListener("click", ask);
el("btnPrev").addEventListener("click", preview);
