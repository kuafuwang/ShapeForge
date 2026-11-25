import hashlib, json
from pathlib import Path
from tcenv.evaluation import EvaluationRequest
from tcenv.remote.grpc_client import GrpcExpected, grpc_evaluator_factory_from_config

ROOT=Path('/tmp/shapeforge-remote-extended-client')
TLS=ROOT/'tls'
CASES=json.loads((ROOT/'task-fingerprints.json').read_text())
META={
 'msl-bundle': {'track_id':'msl-bundle','track_version':1},
 'host-program': {'track_id':'msl-bundle','track_version':1},
 'tilelang': {'track_id':'tilelang-pipeline','track_version':3},
}
factory=grpc_evaluator_factory_from_config(
 endpoint='172.16.145.1:7443',
 token_provider=lambda _run,_session:(TLS/'session.token').read_text().strip(),
 run_id='remote-extended-run',target_id='apple-m4-max',
 ca_file=TLS/'ca.pem',certificate_file=TLS/'client.pem',private_key_file=TLS/'client.key',
 keyring_file=TLS/'keyring.json',
 expected=GrpcExpected(site_id='extended-staging',device_id='apple-m4-max',target_id='apple-m4-max',worker_build='v2-extended-envelope-staging-r2',task_pack_digest='051d0d65a03797ae7305a48382fdc2e16a48d19f57d40ae884fd6bcab1da7b1e'),
 server_name_override='evaluator.local',rpc_timeout_s=60.0,watch_slice_s=10.0,job_margin_s=60.0,
)
rows=[]
try:
 epoch=factory.ensure_connected()
 for task in CASES:
  name={'affine_ramp':'msl-bundle','varlen_kda':'host-program','relu':'tilelang'}[task['task']]
  source=(ROOT/'candidates'/(name+'.taic')).read_text()
  request=EvaluationRequest(
   kind='formal_eval',request_id='extended-'+name+'-2',run_id='remote-extended-run',session_id='session-1',
   task=task['task'],dtype=task['dtype'],size=task['size'],target_id='apple-m4-max',
   task_fingerprint=task['fingerprint'],source=source,mode='quick',seed=7,reps=1,
   queue_timeout_s=60.0,candidate_timeout_s=300.0,cache_policy='disabled',
  )
  first=factory.formal(request)
  replay=factory.formal(request)
  assert first.report['status']=='ok', first.report
  assert first.report['correctness']['ok'] and first.report['safety']['ok']
  assert first.report['build']['candidate_sha256']==request.candidate_sha256
  assert first.report['build']['provenance']['source_sha256']==request.candidate_sha256
  assert first.report==replay.report
  for key in ('attempt_id','receipt_id','lease_fence'):
   assert first.receipt[key]==replay.receipt[key], (name,key,first.receipt,replay.receipt)
  assert first.provenance.epoch==replay.provenance.epoch==epoch
  descriptor=(first.report.get('build') or {}).get('native',{}).get('descriptor',{})
  if name=='host-program':
   host=descriptor.get('host_program') or {}
   assert host.get('entry_path')=='host/tiling.c' and host.get('execution_backend')=='native-c99'
   assert host.get('native_sha256') and host.get('assembled_host_sha256')
   assert first.report['pipeline']['stages']==4
  elif name=='tilelang':
   tilelang=descriptor.get('tilelang') or {}
   assert tilelang.get('track')=={'track_id':'tilelang-pipeline','track_version':3}
   assert first.report['build']['provenance']['builder_version']=='3'
  else:
   assert first.report['pipeline']['kind']=='v2' and 'host_program' not in descriptor
  result_dir=ROOT/'remote-results'; result_dir.mkdir(exist_ok=True)
  (result_dir/(name+'.json')).write_text(json.dumps({
   'report':first.report,'receipt':first.receipt,'evidence':first.evidence,
   'provenance':first.provenance.__dict__,'replay_report_equal':True,
  },indent=2,sort_keys=True,default=str)+'\n')
  row={
   'name':name,**META[name],
   'task':task['task'],'dtype':task['dtype'],'size':task['size'],
   'task_fingerprint':task['fingerprint'],
   'candidate_sha256':request.candidate_sha256,
   'envelope_sha256':hashlib.sha256(source.encode()).hexdigest(),
   'request_digest':request.request_digest,
   'status':first.report['status'],'correctness_ok':first.report['correctness']['ok'],'safety_ok':first.report['safety']['ok'],
   'attempt_id':first.receipt['attempt_id'],'receipt_id':first.receipt['receipt_id'],'lease_fence':first.receipt['lease_fence'],
   'evaluator_epoch':first.provenance.epoch,'replay_identity_match':True,
   'track_report':first.report.get('track'),
   'build_track':(first.report.get('build') or {}).get('provenance',{}).get('build_profile'),
  }
  rows.append(row)
  print(json.dumps(row,sort_keys=True),flush=True)
finally:
 factory.close()
out={
 'schema_id':'local.v2-remote-extended-staging.1','publication_allowed':False,
 'client':'real Ubuntu VM via SSH and mTLS','server':'real Apple M4 Max process-worker development staging',
 'cache_policy':'disabled','measurement_kind':'measured','rows':rows,
}
(ROOT/'remote-summary.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
