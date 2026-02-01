let timer = null;

async function jget(url){
  const r = await fetch(url);
  return await r.json();
}
const el = (id)=>document.getElementById(id);

function renderSteps(steps, currentStage){
  const root = el("steps");
  root.innerHTML = "";
  for(const s of steps){
    const div = document.createElement("div");
    div.className = "step";
    div.textContent = s.label;
    if(s.key === currentStage) div.classList.add("active");
    if(currentStage === "ready" && s.key === "ready") div.classList.add("done");
    if(currentStage === "error") {
      if(s.key === "error") div.classList.add("error");
    }
    root.appendChild(div);
  }
}

async function watch(){
  const jobId = el("jobId").value.trim();
  if(!jobId) return;

  const stepsRes = await jget("/workflow/steps");
  const steps = stepsRes.steps || [];

  async function tick(){
    const st = await jget(`/jobs/${encodeURIComponent(jobId)}/status`);
    const lg = await jget(`/jobs/${encodeURIComponent(jobId)}/logs?tail=200`);
    renderSteps(steps, st.stage);
    el("status").textContent = JSON.stringify(st, null, 2);
    el("logs").textContent = JSON.stringify(lg.logs || [], null, 2);

    if(st.stage === "ready" || st.stage === "error" || st.stage === "not_found"){
      stop();
    }
  }

  stop();
  await tick();
  timer = setInterval(tick, 2500);
}

function stop(){
  if(timer){
    clearInterval(timer);
    timer = null;
  }
}

el("btnWatch").addEventListener("click", watch);
el("btnStop").addEventListener("click", stop);
