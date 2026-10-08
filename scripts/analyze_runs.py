"""읽기 전용 공통 SAC/MBPO 분석. simulator/learner를 생성하지 않는다."""
import argparse
from array import array
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICY = ['actor_loss','critic_loss','policy_q_min_mean','alpha','alpha_loss','entropy_estimate']


def stream(path):
    if path.exists():
        with path.open() as f:
            for number,line in enumerate(f,1):
                try:
                    yield json.loads(line)
                except ValueError as e:
                    raise ValueError(f'{path}:{number}: {e}') from e


def flat(obj, prefix=''):
    if isinstance(obj,dict):
        for k,v in obj.items(): yield from flat(v,f'{prefix}/{k}' if prefix else k)
    elif isinstance(obj,(list,tuple)):
        for i,v in enumerate(obj): yield from flat(v,f'{prefix}/{i}')
    elif isinstance(obj,(int,float)):
        yield prefix, obj


def csvfile(path,rows):
    rows=list(rows)
    keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,keys); writer.writeheader(); writer.writerows(rows)


def stats(values):
    a=np.asarray(values,dtype=float); good=a[np.isfinite(a)]
    if not len(good): return dict(count=len(a),nonfinite=len(a))
    return dict(count=len(a),nonfinite=int(len(a)-len(good)),mean=float(good.mean()),
        median=float(np.median(good)),p95=float(np.quantile(good,.95)),p99=float(np.quantile(good,.99)),
        minimum=float(good.min()),maximum=float(good.max()),abs_maximum=float(np.abs(good).max()),last=float(good[-1]))


def generation_for(step, generations):
    before=[g for g in generations if g['real_env_steps']<=step]
    return before[-1]['refit'] if before else None


def analyze(run,out):
    meta=json.loads((run/'metadata.json').read_text()); label=run.parts[-4]+'_'+run.name
    task=meta.get('training_environment_contract',{}).get('env_id',label)
    algo=meta.get('algorithm','unknown'); common=dict(run=label,task=task,algorithm=algo)
    directory=out/label; directory.mkdir()
    available={str(p.relative_to(run)):p.stat().st_size for p in run.rglob('*')
               if p.is_file() and 'wandb' not in p.parts}
    (directory/'files.json').write_text(json.dumps(available,indent=2))
    generations=list(stream(run/'metrics/synthetic_generations.jsonl'))
    model=list(stream(run/'metrics/model_refits.jsonl'))
    inventory={**common,'path':str(run),'git_commit':meta.get('git_commit'), 'git_dirty':meta.get('git_dirty'),
        **{k:meta.get(k) for k in ['real_env_steps','vector_steps','policy_gradient_steps','dynamics_model_train_steps',
        'dynamics_model_refit_count','synthetic_transition_count','real_policy_samples','synthetic_policy_samples']},
        'resolved_config_exists':(run/'resolved_config.yaml').exists(),
        'tracking_errors_exists':(run/'tracking_errors.jsonl').exists(),
        'replay_artifacts':[p for p in available if 'replay' in p.lower()],
        'video_count':sum(p.endswith('.mp4') for p in available)}
    for name in ('metadata.json','resolved_config.yaml','config.yaml','tracking_summary.json'):
        if (run/name).exists(): (directory/name).write_bytes((run/name).read_bytes())
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
    proc=subprocess.run([sys.executable,str(ROOT/'scripts/check_tracking.py'),'--run-dir',str(run)],
                         capture_output=True,text=True,env=env)
    (directory/'tracking_audit.txt').write_text(proc.stdout+proc.stderr)
    inventory['audit_exit_code']=proc.returncode
    # Exact quantiles: compact numeric arrays; never retain full policy JSON objects.
    values=defaultdict(lambda:array('d')); summaries=[]; anomalies=[]; synthetic=[]; evaluations=[]
    sample_points=[]; latest_refit=None; nonfinite=0; events=0; refit_events={}
    anomaly_buckets=defaultdict(list)
    with ((run/'tracking_metrics.jsonl').open() if (run/'tracking_metrics.jsonl').exists() else io.StringIO('')) as f:
        for line in f:
            row=json.loads(line); m=row['metrics']; step=m['real_env_steps']; event=m.get('metric_event_id'); events+=1
            if 'model/refit' in m:
                latest_refit=m['model/refit']; refit_events[latest_refit]=event
            if 'policy_update/actor_loss' in m:
                update=m['policy_gradient_steps']
                for metric in POLICY:
                    key='policy_update/'+metric
                    if key not in m: continue
                    v=m[key]
                    for axis,bin_id in [('all',0),('real_env_steps',(int(step)-1)//10000),('policy_gradient_steps',(int(update)-1)//5000)]:
                        values[(axis,bin_id,metric)].append(v)
                    if not math.isfinite(v): nonfinite+=1
                if update%256==0 or update==1:
                    sample_points.append(dict(step=step,update=update,**{k:m.get('policy_update/'+k) for k in POLICY}))
                # Explicit descriptive thresholds, not causal or success gates.
                for metric,threshold in [('critic_loss',1e4),('policy_q_min_mean',100),('alpha',10)]:
                    v=m.get('policy_update/'+metric,0)
                    if abs(v)>threshold:
                        anomaly_buckets[(metric,(step-1)//10000)].append((step,update,v,event,latest_refit))
            if 'eval/mean_return' in m:
                evaluations.append(dict(**common,source='periodic_ledger',step=step,
                    **{k:m.get('eval/'+k) for k in ['success_once','success_at_end','mean_return','mean_episode_length','episodes']}))
            for layer,prefix in [('B_pool','model_pool/'),('C_sampled_batch','model_sampled_batch/')]:
                selected={k[len(prefix):]:v for k,v in m.items() if k.startswith(prefix)}
                if selected:
                    synthetic.extend(dict(**common,layer=layer,step=step,event=event,refit=selected.get("refit",latest_refit),metric=k,value=v)
                                     for k,v in selected.items())
    for (axis,bin_id,metric),a in values.items():
        width=10000 if axis=='real_env_steps' else 5000
        summaries.append(dict(**common,axis=axis,bin_start=bin_id*width,bin_end=(bin_id+1)*width if axis!='all' else '',
                              metric=metric,aggregation='all_update_events_exact_quantiles',**stats(a)))
    for (metric,bin_id),items in anomaly_buckets.items():
        first=items[0]; peak=max(items,key=lambda x:abs(x[2])); last=items[-1]
        anomalies.append(dict(**common,kind='policy_threshold_window',metric=metric,step=first[0],update=first[1],
            event=first[3],refit=first[4],value=first[2],peak_step=peak[0],peak_value=peak[2],
            last_step=last[0],count=len(items),window_start=bin_id*10000))
    blocks=[]
    for r in model:
        step=r['real_env_steps']; refit=r['refit']
        for k,v in flat(r):
            blocks.append(dict(**common,step=step,refit=refit,event=refit_events.get(refit),metric=k,value=v))
        for i,d in enumerate(r.get('synthetic_state_diagnostics',[])):
            synthetic.extend(dict(**common,layer='A_generation',step=step,refit=refit,event=refit_events.get(refit),member='TS1_mixture',metric=k,value=v)
                             for k,v in flat(d))
            if d.get('reward_min',0)<-1 or d.get('reward_max',0)>2:
                retained=[g for g in generations if any(x['refit']==refit for x in g['retained_generations'])]
                later=[g for g in generations if g['refit']>refit and not any(x['refit']==refit for x in g['retained_generations'])]
                anomalies.append(dict(**common,kind='generation_reward_extreme',step=step,refit=refit,event=refit_events.get(refit),
                    value=d['reward_min'],maximum=d['reward_max'],elite_mean_min=d.get('elite_mean_reward_min'),
                    elite_mean_max=d.get('elite_mean_reward_max'),elites=r.get('elites'),
                    last_retained_refit_step=retained[-1]['real_env_steps'] if retained else '',
                    fully_evicted_step=later[0]['real_env_steps'] if later else 'not_observed'))
        if any(r.get('restored_fit_start',[])):
            anomalies.append(dict(**common,kind='fit_start_restored',step=step,refit=refit,event=refit_events.get(refit),
                members=r['restored_fit_start'],train_nll=r['model_train_loss'],holdout_nll=r['model_validation_loss'],elites=r['elites']))
    # Raw per-environment episodes, never conflate means with per-update observations.
    ep=defaultdict(lambda:defaultdict(list))
    for r in stream(run/'metrics/episodes.jsonl'):
        b=(r['real_env_steps']-1)//10000
        for k in ('episode_return','episode_length','success_once','success_at_end'):
            ep[b][k].append(float(r[k]))
    for b,group in ep.items():
        for k,v in group.items():summaries.append(dict(**common,axis='real_env_steps',bin_start=b*10000,bin_end=(b+1)*10000,
            metric='train_episode/'+k,aggregation='raw_finished_episodes',**stats(v)))
    frozen=[]
    for p in run.glob('frozen*/**/summary.json'):
        d=json.loads(p.read_text()); frozen.append(d)
        evaluations.append(dict(**common,source=str(p),step=d['counters']['real_env_steps'],
            **{k:d.get(k) for k in ['success_once','success_at_end','mean_return','mean_episode_length','episodes']},
            contract=json.dumps(d['environment_contract'],sort_keys=True),seeds=json.dumps(d.get('episode_seeds')),
            learner_updates=d.get('learner_updates_during_evaluation'),probe=d.get('learner_restore_probe')))
    if not evaluations:
        for p in run.glob('metrics/evaluation*/summary.json'):
            d=json.loads(p.read_text())
            evaluations.append(dict(**common,source='periodic_file',step=d.get('counters',{}).get('real_env_steps',0),
                **{k:d.get(k) for k in ['success_once','success_at_end','mean_return','mean_episode_length','episodes']}))
        evaluations.sort(key=lambda e:e['step'])
    inventory['policy_coverage'] = 'all_update_ledger' if events else 'UNVERIFIED: update ledger unavailable; no substitute of last-update logs'
    inventory.update(ledger_events=events,nonfinite_policy_values=nonfinite,frozen_evaluations=len(frozen))
    csvfile(directory/'learning_summary.csv',summaries); csvfile(directory/'anomaly_timeline.csv',anomalies)
    csvfile(directory/'model_diagnostics.csv',blocks); csvfile(directory/'synthetic_diagnostics.csv',synthetic)
    return dict(common=common,inventory=inventory,summaries=summaries,anomalies=anomalies,model=model,
                synthetic=synthetic,evaluations=evaluations,frozen=frozen,points=sample_points,generations=generations,run=run)


def checkpoints(result,out):
    import torch
    rows=[]; manifests=[]; probes=[]; common=result['common']; run=result['run']; previous=None; norms_equal=True
    for path in sorted((run/'checkpoints').glob('*.pt')):
        if path.name=='latest.pt': continue
        try:
            state=torch.load(path,map_location='cpu',weights_only=True)
            manifests.append(dict(file=str(path),keys=list(state),replay_persisted=state.get('replay_persisted'),
                                  simulator_persisted=state.get('simulator_persisted')))
            if 'model' in state:
                m=state['model']; norm=m.get('normalization')
                if not norm: continue
                arrays={k:v.cpu().numpy().reshape(-1) for k,v in norm.items()}
                if previous is not None: norms_equal &= all(np.array_equal(previous[k],v) for k,v in arrays.items())
                previous=arrays
                layout=(m.get('observation_codec') or m.get('geometry') or {}).get('layout',{})
                for kind in ('input','target'):
                    mean,std=arrays[kind+'_mean'],arrays[kind+'_std']
                    for i,(mu,sigma) in enumerate(zip(mean,std)):
                        block=next((k for k,(a,b) in layout.items() if a<=i<b), 'reward' if kind=='target' else 'action')
                        rows.append(dict(**common,checkpoint=path.name,kind=kind,index=i,block=block,mean=float(mu),std=float(sigma),
                            at_floor=bool(sigma<=1.000001e-6),below_1e_3=bool(sigma<1e-3),unit_change_z=1/float(sigma)))
            elif 'learner_probe' in state:
                probe=state['learner_probe']; probes.append((path.name,np.asarray(probe['observation'])))
        except Exception as e: manifests.append(dict(file=str(path),error=str(e)))
    result['normalizers_equal']=norms_equal if previous is not None else None
    result['normalizer_rows']=rows
    # Only saved probe observations, not the missing replay or training inputs.
    if previous is not None:
        codec=(result['frozen'][0]['environment_contract'].get('observation_codec',{}) if result['frozen'] else {})
        for name,obs in probes:
            obs=obs.astype(float).copy()
            for key in codec.get('quaternion_poses',[]):
                a,b=codec['layout'][key]; q=obs[a+3:b]; q=q/np.linalg.norm(q)
                obs[a+3:b]=q*(-1 if q[np.argmax(np.abs(q))]<0 else 1)
            z=np.abs((obs-previous['input_mean'][:len(obs)])/previous['input_std'][:len(obs)])
            manifests.append(dict(probe_checkpoint=name,coverage='one saved env observation; not historical rollout batch',
                                  normalized_abs_max=float(z.max()),p95=float(np.quantile(z,.95)),argmax=int(z.argmax())))
    (out/result['common']['run']/'checkpoint_inventory.json').write_text(json.dumps(manifests,indent=2))
    return rows


def plots(results,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9})
    for task in sorted(set(r['common']['task'] for r in results)):
        runs=[r for r in results if r['common']['task']==task]
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for r in runs:
            label=r['common']['algorithm']+' '+r['run'].name[-8:]
            ev=[e for e in r['evaluations'] if e['source'].startswith('periodic_')]
            axes[0,0].plot([e['step'] for e in ev],[e['success_once'] for e in ev],'.-',label=label)
            for ax,key in zip(axes.flat[1:],['critic_loss','policy_q_min_mean','alpha','entropy_estimate','actor_loss']):
                bins=[x for x in r['summaries'] if x['axis']=='real_env_steps' and x['metric']==key]
                x=[b['bin_end'] for b in bins]; ax.plot(x,[b['median'] for b in bins],label=label)
                ax.fill_between(x,[b['minimum'] for b in bins],[b['p99'] for b in bins],alpha=.12)
                ax.set_yscale('symlog',linthresh=.01)
        for ax,title in zip(axes.flat,['Periodic eval success','Critic loss median / min-p99','Policy Q (signed)','Alpha','Entropy estimate','Actor loss']):
            ax.set_title(title);ax.set_xlabel('real transitions');ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle(task+' | training seed 0');fig.tight_layout();fig.savefig(out/(task+'_learning.png'),dpi=150);plt.close(fig)
        for r in runs:
            if not r['model']:continue
            fig,axes=plt.subplots(2,3,figsize=(14,8))
            model=r['model'];x=[m['real_env_steps'] for m in model]
            for key in model[0]['prediction_errors']['holdout']['state_blocks']:
                if '.quaternion_components' in key:continue
                values=[]
                for m in model:
                    b=m['prediction_errors']['holdout']['state_blocks'][key];den=b['zero_delta_baseline_rmse']
                    values.append(np.median(b['member_rmse'])/den if den>0 else np.nan)
                axes[0,0].plot(x,values,label=key)
            axes[0,0].axhline(1,color='black',ls='--');axes[0,0].set_yscale('log');axes[0,0].legend(fontsize=5)
            for layer in ('A_generation','B_pool','C_sampled_batch'):
                for suffix,ax in [('reward_min',axes[0,1]),('reward_max',axes[0,2])]:
                    data=[d for d in r['synthetic'] if d['layer']==layer and d['metric'].endswith('/'+suffix) or d['layer']==layer and d['metric']==suffix]
                    ax.plot([d['step'] for d in data],[d['value'] for d in data],'.',ms=2,label=layer);ax.set_yscale('symlog',linthresh=1);ax.legend(fontsize=7)
            for part in ('model_train_loss','model_validation_loss'):
                axes[1,0].plot(x,[max(m[part]) for m in model],label=part)
            axes[1,0].set_yscale('symlog',linthresh=1);axes[1,0].legend(fontsize=7)
            gs=r['generations'];axes[1,1].plot([g['real_env_steps'] for g in gs],[g['oldest_generation_age_transitions'] for g in gs])
            for part in ('extra.tcp_to_obj_pos_consistency_rmse','extra.tcp_to_cubeA_pos_consistency_rmse'):
                for layer in ('A_generation','B_pool','C_sampled_batch'):
                    data=[d for d in r['synthetic'] if d['layer']==layer and d['metric'].endswith(part)]
                    if data:axes[1,2].plot([d['step'] for d in data],[d['value'] for d in data],'.',ms=2,label=layer)
            axes[1,2].set_yscale('log');axes[1,2].legend(fontsize=7)
            for ax,title in zip(axes.flat,['Holdout RMSE / unchanged baseline','Synthetic reward min','Synthetic reward max','Worst member NLL','Oldest retained generation age','Relative-position inconsistency']):
                ax.set_title(title);ax.set_xlabel('real transitions');ax.grid(alpha=.2)
            fig.suptitle(task+' MBPO (diagnostic coverage differs by layer)');fig.tight_layout();fig.savefig(out/(task+'_model.png'),dpi=150);plt.close(fig)


def checkpoint_sensitivity(result,out):
    """CPU deterministic arithmetic only; no simulator, optimizer or learner update."""
    import torch
    from torch import nn
    rows=[];run=result['run']
    paths=sorted((run/'checkpoints').glob('step_*_model.pt'))
    paths=([min(paths,key=lambda p:int(p.name.split('_')[1]))] if paths else [])
    if (run/'checkpoints/final_model.pt').exists():paths.append(run/'checkpoints/final_model.pt')
    for path in paths:
        saved=torch.load(path,map_location='cpu',weights_only=True)['model']; codec=saved.get('observation_codec') or {}
        fields=codec.get('boolean_fields',[])
        if not fields:continue
        policy_path=path.with_name(path.name.replace('_model.pt','.pt'))
        payload=torch.load(policy_path,map_location='cpu',weights_only=True)
        probe=payload.get('learner_probe')
        if probe is None:continue
        observation=np.asarray(probe['observation'],dtype=np.float32).copy()
        for key in codec['quaternion_poses']:
            a,b=codec['layout'][key];q=observation[a+3:b];q=q/np.linalg.norm(q)
            observation[a+3:b]=q*(-1 if q[np.argmax(np.abs(q))]<0 else 1)
        norm=saved['normalization']; od=saved['obs_dim']; ad=saved['action_dim']
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            for member in range(saved['config']['ensemble_size']):
                layers=[];widths=[od+ad,*saved['config']['model_hidden_dims']]
                for a,b in zip(widths,widths[1:]):layers.extend([nn.Linear(a,b),nn.SiLU()])
                layers.append(nn.Linear(widths[-1],2*(od+1)));network=nn.Sequential(*layers)
                prefix=f'{member}.net.'
                network.load_state_dict({k[len(prefix):]:v for k,v in saved['members'].items() if k.startswith(prefix)})
                for field in fields:
                    index=codec['layout'][field][0]
                    for flag in (0.,1.):
                        obs=observation.copy();obs[index]=flag
                        inputs=torch.tensor(np.r_[obs,probe['action']],dtype=torch.float32)
                        z=(inputs-norm['input_mean'])/norm['input_std']
                        mean,raw=network(z).chunk(2,-1)
                        lv=.5-torch.nn.functional.softplus(.5-raw)
                        lv=-10+torch.nn.functional.softplus(lv+10)
                        physical=mean*norm['target_std']+norm['target_mean']
                        rows.append(dict(**result['common'],checkpoint=path.name,member=member,elite=member in saved['elites'],
                            field=field,value=flag,normalized_field=float(z[index]),normalized_abs_max=float(z.abs().max()),
                            normalized_mean_reward=float(mean[-1]),physical_mean_reward=float(physical[-1]),
                            reward_noise_std=float((lv[-1].exp().sqrt()*norm['target_std'][-1])),
                            coverage='counterfactual fixed probe/action; physical consistency NOT established; not past rollout'))
    return rows


def video_contact_sheets(results,out,ffmpeg):
    from PIL import Image, ImageDraw
    import io
    rows=[]
    for task in sorted(set(r['common']['task'] for r in results)):
        for seed in (21000,21001):
            entries=[]
            for r in results:
                if r['common']['task']!=task:continue
                files=sorted(r['run'].glob(f'videos/step_500000*/step_500000_seed_{seed}.mp4'))
                if not files:continue
                p=files[0]; summary=json.loads((p.parent/'summary.json').read_text())
                episodes=list(csv.DictReader((p.parent/'episodes.csv').open()))
                row=next(x for x in episodes if int(x['seed'])==seed)
                probe=subprocess.run([str(Path(ffmpeg).with_name('ffprobe')),'-v','quiet','-show_entries','format=duration','-of','json',str(p)],capture_output=True,text=True,check=True)
                duration=float(json.loads(probe.stdout)['format']['duration'])
                frames=[]
                for t in np.linspace(0,max(0,duration-.05),5):
                    image=subprocess.run([ffmpeg,'-v','error','-ss',str(t),'-i',str(p),'-frames:v','1','-f','image2pipe','-vcodec','png','-'],capture_output=True,check=True)
                    frames.append(Image.open(io.BytesIO(image.stdout)).convert('RGB').resize((240,240)))
                entries.append((r['common']['algorithm'],frames,duration,row))
                rows.append(dict(**r['common'],file=str(p),duration=duration,**row,
                                 contract=json.dumps(summary['environment_contract'],sort_keys=True)))
            if entries:
                sheet=Image.new('RGB',(1200,270*len(entries)),'white');draw=ImageDraw.Draw(sheet)
                for i,(algo,frames,duration,row) in enumerate(entries):
                    draw.text((5,i*270+4),f'{task} {algo} seed {seed} | 0..{duration:.2f}s | success={row["success_once"]} length={row["episode_length"]}',fill='black')
                    for j,frame in enumerate(frames):sheet.paste(frame,(j*240,i*270+25))
                sheet.save(out/f'{task}_seed_{seed}_video.png')
    csvfile(out/'video_inventory.csv',rows)


def paired_evaluations(results,out):
    pairs=[]
    for task in sorted(set(r['common']['task'] for r in results)):
        group=[r for r in results if r['common']['task']==task and r['frozen']]
        if len(group)!=2:continue
        left=next((r for r in group if r['common']['algorithm']=='sac'),None)
        right=next((r for r in group if r['common']['algorithm']=='mbpo'),None)
        if left is None or right is None:continue
        if len(left['frozen']) != 1 or len(right['frozen']) != 1:
            pairs.append(dict(task=task,status='UNVERIFIED: multiple protocols; select one explicitly'));continue
        a,b=left['frozen'][0],right['frozen'][0]
        same=a['environment_contract']==b['environment_contract'] and a['episode_seeds']==b['episode_seeds']
        if not same:
            pairs.append(dict(task=task,status='NOT_COMPARABLE: contract/seed mismatch'));continue
        def load(r):
            p=next(r['run'].glob('frozen*/**/episodes.csv')); rows=list(csv.DictReader(p.open()))
            ids=[int(x['seed']) for x in rows]
            return {int(x['seed']):x for x in rows},len(ids)-len(set(ids))
        aa,ad=load(left);bb,bd=load(right)
        for name,flags in [('both',(True,True)),('sac_only',(True,False)),('mbpo_only',(False,True)),('neither',(False,False))]:
            seeds=[s for s in aa.keys()&bb.keys() if (aa[s]['success_once'].lower() in ('true','1'),bb[s]['success_once'].lower() in ('true','1'))==flags]
            d=dict(task=task,group=name,count=len(seeds),contract_and_seeds_match=same,duplicate_sac=ad,duplicate_mbpo=bd,
                   missing_sac=len(bb.keys()-aa.keys()),missing_mbpo=len(aa.keys()-bb.keys()))
            for key in ('episode_length','per_episode_return'):
                d[key+'_mbpo_minus_sac_mean']=float(np.mean([float(bb[s][key])-float(aa[s][key]) for s in seeds])) if seeds else ''
            pairs.append(d)
    csvfile(out/'paired_evaluation.csv',pairs)


def extra_plots(results,out):
    import matplotlib.pyplot as plt
    for task in sorted(set(r['common']['task'] for r in results)):
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for r in results:
            if r['common']['task']!=task:continue
            label=r['common']['algorithm']
            for ax,key in zip(axes[0],['critic_loss','policy_q_min_mean','alpha']):
                bins=[s for s in r['summaries'] if s['axis']=='policy_gradient_steps' and s['metric']==key]
                ax.plot([s['bin_end'] for s in bins],[s['median'] for s in bins],label=label);ax.set_yscale('symlog',linthresh=.01)
            for ax,key in zip(axes[1,:2],['mean_return','mean_episode_length']):
                ev=[e for e in r['evaluations'] if e['source'].startswith('periodic_')]
                ax.plot([e['step'] for e in ev],[e[key] for e in ev],'.-',label=label)
            bins=[s for s in r['summaries'] if s['metric']=='train_episode/success_once']
            axes[1,2].plot([s['bin_end'] for s in bins],[s['mean'] for s in bins],label=label)
        for i,(ax,title) in enumerate(zip(axes.flat,['Critic median','Signed policy Q median','Alpha median','Eval return (termination-biased)','Eval episode length','Training episode success'])):
            ax.set_title(title);ax.set_xlabel('policy updates' if i<3 else 'real transitions');ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/f'{task}_updates_and_episodes.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'outputs/nominal_tasks')
    parser.add_argument('--run',type=Path,action='append')
    parser.add_argument('--aux-root',type=Path)
    parser.add_argument('--interpretation',type=Path,help='Optional Korean interpretation specific to the selected run IDs')
    parser.add_argument('--ffmpeg',help='Existing ffmpeg executable; no downloads')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    runs=args.run or sorted(p.parent for p in args.root.glob('*/*/seed_0/*/metadata.json'))
    if args.aux_root: runs += sorted(p.parent for p in args.aux_root.glob('*/seed_0/*/metadata.json'))
    results=[];normalizers=[]
    for run in runs:
        print('분석:',run,flush=True);r=analyze(run.resolve(),out);results.append(r)
        normalizers.extend(checkpoints(r,out))
    for filename,key in [('run_inventory.csv','inventory'),('learning_summary.csv','summaries'),('anomaly_timeline.csv','anomalies'),('synthetic_diagnostics.csv','synthetic'),('evaluation_summary.csv','evaluations')]:
        csvfile(out/filename,[row for r in results for row in ([r[key]] if isinstance(r[key],dict) else r[key])])
    csvfile(out/'normalizers.csv',normalizers)
    # Merge long model tables without retaining their many scalar rows globally.
    with (out/'model_diagnostics.csv').open('x') as dest:
        written=False
        for i,r in enumerate(results):
            with (out/r['common']['run']/'model_diagnostics.csv').open() as source:
                header=source.readline()
                if header.strip() == 'status': continue
                if not written: dest.write(header); written=True
                for line in source:dest.write(line)
    plots(results,out)
    extra_plots(results,out)
    paired_evaluations(results,out)
    csvfile(out/"checkpoint_sensitivity.csv",[row for r in results for row in checkpoint_sensitivity(r,out)])
    if args.ffmpeg:
        try: video_contact_sheets(results,out,args.ffmpeg)
        except Exception as e: (out/'video_errors.txt').write_text(str(e))
    sources={}
    for p in [*ROOT.glob('src/dynamics_shift/**/*.py'),Path(__file__)]:
        sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'source_fingerprints.json').write_text(json.dumps(sources,indent=2))
    (out/'git_state.txt').write_text(subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout + subprocess.run(['git','status','--short'],cwd=ROOT,capture_output=True,text=True).stdout)
    nll_rows=[]
    for r in results:
        if not r['model']:continue
        last=r['model'][-1]
        for n in r['normalizer_rows']:
            if n['checkpoint']!='final_model.pt' or n['kind']!='target' or not n['at_floor']:continue
            block=last['prediction_errors']['holdout']['state_blocks'].get(n['block'])
            if not block or n['block']!='extra.is_grasped':continue
            for member,rmse in enumerate(block['member_rmse']):
                width=next(x['index'] for x in reversed(r['normalizer_rows']) if x['checkpoint']=='final_model.pt' and x['kind']=='target')+1
                bound=(rmse/n['std'])**2/(2*width*math.exp(-10+math.log1p(math.exp(10.5))))-5
                nll_rows.append(dict(**r['common'],member=member,component=n['block'],rmse=rmse,std=n['std'],
                    nll_lower_bound=bound,observed_holdout_nll=last['model_validation_loss'][member],
                    lower_bound_fraction=bound/last['model_validation_loss'][member]))
    csvfile(out/'normalizer_nll_bound.csv',nll_rows)
    summary=[]
    for r in results:
        summary.append(dict(**r['inventory'],normalizers_equal=r['normalizers_equal'],
            last_update={s['metric']:s['last'] for s in r['summaries'] if s['axis']=='all'},
            frozen=[{k:d.get(k) for k in ['success_once','mean_return','mean_episode_length','checkpoint','checkpoint_sha256']} for d in r['frozen']]))
    (out/'analysis_summary.json').write_text(json.dumps(summary,indent=2))
    report=['# SAC·MBPO nominal 분석','', '자동 집계: 전체 update event의 정확한 분위수; 10k real-transition 및 5k update 구간. 원본은 수정하지 않음.','']
    for r in summary:
        report.extend([f"## {r['task']} / {r['algorithm']}",f"경로: `{r['path']}`",f"실행 commit: {r['git_commit']}, dirty={r['git_dirty']}. 당시 미커밋 소스 동일성 미검증.",
            f"500k 카운터 및 audit: 업데이트 {r['policy_gradient_steps']}, audit exit {r['audit_exit_code']}.",f"최종 update: `{r['last_update']}`",f"Frozen 평가: `{r['frozen']}`",''])
    report.extend(['## 해석 제한','동일 task/seed/contract만 비교한다. 성공 즉시 종료하므로 raw return은 성공률과 함께 해석한다.',
        '고정 normalizer의 초기 데이터/replay와 과거 batch는 저장되지 않았다. checkpoint probe는 한 시점의 한 환경이며 분포를 대표하지 않는다.',
        '생성/보존 pool/실제 batch 진단 coverage가 다르다. ledger audit PASS는 학습 안정성이나 synthetic utility를 입증하지 않는다.',
        '단일 training seed의 nominal 결과이며 dynamics shift·일반적 우월성 증거가 아니다.'])
    if args.interpretation: report.append(args.interpretation.read_text())
    (out/'analysis_report.md').write_text('\n'.join(report)+'\n')
    print(out)

if __name__=='__main__':main()
