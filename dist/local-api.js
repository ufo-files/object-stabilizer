export async function request(path,options={}){
 const response=await fetch(path,options);
 if(!response.ok){let message=`Local service error (${response.status})`;try{const data=await response.json();message=typeof data.detail==='string'?data.detail:message;}catch{}throw Error(message);}
 return response.json();
}
export async function waitForJob(id,onProgress,shouldCancel=()=>false){
 for(;;){
  if(shouldCancel()){await request(`/api/jobs/${id}/cancel`,{method:'POST'});throw Error('Cancelled.');}
  const job=await request(`/api/jobs/${id}`);onProgress(job.progress,job.message);
  if(job.status==='complete')return job.result;
  if(job.status==='failed')throw Error(job.error||'Processing failed.');
  if(job.status==='cancelled')throw Error('Cancelled.');
  await new Promise(resolve=>setTimeout(resolve,250));
 }
}
