# 公共训练调度：通过适配器调用数据、模型、损失及验证策略，业务差异留在子类中。
# 训练使用 train，选权重和置信度使用 val，测试集不参与这一循环。
"""Config-driven training template shared through the task adapter contract."""
import random
import time
import math
from pathlib import Path
import numpy as np
import torch
from src.core.common import save_json,digest,image_size
from src.core.config import resolve_path
from src.deployment.weights import load_weights
from src.engine.experiment import ExperimentStore
from src.optim.training import build_optimizer,ModelEMA
from src.metrics.small_target import select_operating_point

# 无梯度验证 EMA 模型：保留低分预测，让任务子类在验证集上选择置信度。
def _uses_amp(cfg, device):
    precision = cfg['train'].get('precision', 'fp32')
    if precision not in ('fp32', 'amp'):
        raise NotImplementedError(f'Unsupported training precision: {precision}')
    if precision == 'amp' and not str(device).startswith('cuda'):
        raise ValueError('AMP training requires a CUDA device')
    return precision == 'amp'


def _better_lexicographic(current, previous):
    """Return true only for a strict source-style lexicographic improvement."""
    if previous is None:
        return True
    for value, old in zip(current, previous):
        if math.isclose(float(value), float(old), rel_tol=0.0, abs_tol=1e-12):
            continue
        return float(value) > float(old)
    return False


@torch.inference_mode()
def validate(model,adapter,cfg,loader,device,limit=None):
    model.eval(); records=[]
    amp = _uses_amp(cfg, device)
    for images,targets,meta in loader:
        with torch.cuda.amp.autocast(enabled=amp):
            output=model(images.to(device))
        for i,m in enumerate(meta):
            raw={k:v[i:i+1] for k,v in output.items() if torch.is_tensor(v)}
            # 验证时传入 0 阈值收集候选；业务 NMS 等规则仍由适配器按配置执行。
            pred=adapter.postprocess(raw,m['original_size'],0.)
            records.append(dict(predictions=pred,ground_truth=dict(boxes=m['boxes'],labels=m['labels'])))
        if limit and len(records)>=limit: records=records[:limit]; break
    return adapter.select_operating_point(records,cfg)

# 公共监督检测配方支持 FP32；红外识别任务通过配置启用 CUDA AMP。
def train(cfg,adapter,args):
    adapter.bind(cfg)
    if cfg['train'].get('trainer')!='supervised_detection':
        raise NotImplementedError('This recipe needs its own training strategy; visible DEIM handover is retained in tools/visible_train.py')
    amp = _uses_amp(cfg, args.device)
    seed=cfg['train']['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if str(args.device).startswith('cuda'): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6); torch.backends.cudnn.benchmark=False
    model=adapter.build_model(cfg).to(args.device); criterion=adapter.build_criterion(cfg).to(args.device)
    # 冒烟检查使用真实标注图片计算损失和梯度，提前验证数据、网络、损失的衔接。
    if args.dry_run:
        # One actual annotated image through the same dataset/criterion, without writing weights.
        x,targets,_=next(iter(adapter.build_dataloader(cfg,'train')))
        targets=[{k:v.to(args.device) for k,v in t.items()} for t in targets]
        model.train()
        with torch.cuda.amp.autocast(enabled=amp):
            losses=criterion(model(x.to(args.device),targets),targets); total=sum(losses.values())
        if not torch.isfinite(total): raise FloatingPointError('Non-finite training loss')
        total.backward()
        out=dict(task_id=cfg['task']['id'],loss=float(total),parameters=sum(p.numel() for p in model.parameters()),
                 input_size=list(x.shape),gradient_tensors=sum(p.grad is not None for p in model.parameters()),
                 note='criterion already applies loss weights; no second weighting')
        print(__import__('json').dumps(out,ensure_ascii=False)); return out
    store=ExperimentStore(cfg,args.name,bool(args.resume))
    optimizer=build_optimizer(model,cfg); sched=cfg['scheduler']
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=sched['milestones'],gamma=sched['gamma'])
    warmup_steps=int(cfg['train'].get('warmup_steps',0) or 0)
    warmup=dict(step=0,end_lrs=[group['lr'] for group in optimizer.param_groups])
    if warmup_steps:
        for group,end_lr in zip(optimizer.param_groups,warmup['end_lrs']): group['lr']=end_lr/warmup_steps
    if not args.resume:
        init=cfg['train'].get('initialize')
        if init: load_weights(model,resolve_path(cfg,init),cfg,initialize=True)
    # 先加载初始化权重，再创建 EMA，确保平均模型从相同参数起步。
    ema=ModelEMA(model,**cfg['train']['ema']); start=0; best=None
    scaler=torch.cuda.amp.GradScaler(enabled=amp)
    best_ap40=None
    if args.resume:
        # 完整恢复需同时核对任务、类别、网络、训练配方和标注指纹。
        ck=torch.load(args.resume,map_location='cpu')
        if ck['task_id']!=cfg['task']['id'] or ck['class_names']!=cfg['classes']['names'] or ck['model_config']!=cfg['model']:
            raise ValueError('Resume task/class/model configuration mismatch')
        if ck['training_config']!={k:cfg[k] for k in ('train','optimizer','scheduler')}:
            raise ValueError('Resume training recipe differs from checkpoint')
        model.load_state_dict(ck['model'],strict=True); ema.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        if amp and ck.get('scaler') is not None: scaler.load_state_dict(ck['scaler'])
        if ck.get('warmup') is not None: warmup=ck['warmup']
        # 恢复随机状态以延续数据打乱和增强序列，EMA、优化器及学习率调度也一起恢复。
        random.setstate(ck['rng']['python']); np.random.set_state(ck['rng']['numpy']); torch.set_rng_state(ck['rng']['torch'])
        if str(args.device).startswith('cuda') and ck['rng'].get('cuda') is not None: torch.cuda.set_rng_state_all(ck['rng']['cuda'])
        start=ck['epoch']; best=ck.get('best'); best_ap40=ck.get('best_ap40')
    train_loader=adapter.build_dataloader(cfg,'train'); val_loader=adapter.build_dataloader(cfg,'val')
    provenance={s:digest(adapter.build_dataloader(cfg,s).dataset.annotation_path) for s in ('train','val')}
    if args.resume and provenance!=ck['annotation_sha256']: raise ValueError('Resume dataset differs')
    epochs=args.epochs or cfg['train']['epoches']
    for epoch in range(start,epochs):
        if hasattr(train_loader.dataset,'set_epoch'): train_loader.dataset.set_epoch(epoch)
        model.train(); running=[]; begin=time.perf_counter()
        progress_interval=int(cfg['experiment'].get('progress_interval',0) or 0)
        total_steps=min(len(train_loader),args.max_steps) if args.max_steps else len(train_loader)
        if progress_interval:
            print(f'Epoch {epoch+1}/{epochs}: training {total_steps} batches',flush=True)
        for step,(images,targets,meta) in enumerate(train_loader):
            targets=[{k:v.to(args.device) for k,v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            # criterion 已对各项及辅助层损失施加权重，下一行只求和，避免重复加权。
            with torch.cuda.amp.autocast(enabled=amp):
                losses=criterion(model(images.to(args.device),targets),targets)
                total=sum(losses.values())  # criterion returns weighted losses, including auxiliary losses
            if not torch.isfinite(total): raise FloatingPointError('Non-finite loss')
            if amp:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['clip_max_norm'])
                scaler.step(optimizer); scaler.update()
            else:
                total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['clip_max_norm'])
                optimizer.step()
            if warmup_steps and warmup['step'] < warmup_steps:
                warmup['step'] += 1
                if warmup['step'] < warmup_steps:
                    factor=(warmup['step']+1)/warmup_steps
                    for group,end_lr in zip(optimizer.param_groups,warmup['end_lrs']): group['lr']=factor*end_lr
            ema.update(model); running.append(float(total.detach()))
            if progress_interval and ((step+1)%progress_interval==0 or step+1==total_steps):
                elapsed=time.perf_counter()-begin
                remaining=elapsed/(step+1)*(total_steps-step-1)
                print(f'Epoch {epoch+1}/{epochs} | batch {step+1}/{total_steps} | '
                      f'loss {running[-1]:.4f} | elapsed {elapsed:.0f}s | '
                      f'estimated training remaining {remaining:.0f}s',flush=True)
            if args.max_steps and step+1>=args.max_steps: break
        # Source LinearWarmup controls the first 16 optimizer steps; its epoch
        # scheduler starts only after those steps have completed.
        if not warmup_steps or warmup['step'] >= warmup_steps: scheduler.step()
        if progress_interval:
            print(f'Epoch {epoch+1}/{epochs}: validating',flush=True)
        selected=validate(ema.module,adapter,cfg,val_loader,args.device,args.val_limit)
        if selected.get('selection_key') is not None:
            # Imported R18VD recipe: R40 -> R50 -> R60 -> AP40.
            score=tuple(float(x) for x in selected['selection_key'])
            is_best=_better_lexicographic(score,None if best is None else best['score'])
            ap_score=float(selected['ap_score'])
            is_best_ap40=best_ap40 is None or ap_score>float(best_ap40['score'])
        else:
            pd=selected['metrics']['pd'] or 0.
            fa=selected['metrics']['fa'] if selected['metrics']['fa'] is not None else 1.
            # 已达到 Pd 约束的模型优先，再比较较低 Fa；尚未达标时优先提高 Pd。
            score=(1,-fa,pd) if selected['target_met'] else (0,pd,-fa)
            is_best=best is None or tuple(best['score'])<score
            is_best_ap40=False
        if is_best: best=dict(epoch=epoch+1,score=score,operating_point=selected)
        if is_best_ap40: best_ap40=dict(epoch=epoch+1,score=ap_score,operating_point=selected)
        event=dict(epoch=epoch+1,loss=float(np.mean(running)),seconds=time.perf_counter()-begin,validation=selected,
                   steps=len(running),smoke=bool(args.max_steps or args.val_limit),
                   selection=dict(best_recall=bool(is_best),best_ap40=bool(is_best_ap40)))
        store.event(event)
        # last.pth 保存继续训练所需的全部状态；smoke_test 标识短检查权重，避免当作正式实验结果。
        payload=dict(task_id=cfg['task']['id'],class_names=cfg['classes']['names'],model_config=cfg['model'],
            training_config={k:cfg[k] for k in ('train','optimizer','scheduler')},smoke_test=bool(args.max_steps or args.val_limit),
            model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict() if amp else None,warmup=warmup,epoch=epoch+1,best=best,best_ap40=best_ap40,
            operating_point=selected,annotation_sha256=provenance,
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else None))
        store.checkpoint('last.pth',payload)
        inference=dict(task_id=cfg['task']['id'],class_names=cfg['classes']['names'],model=ema.module.state_dict(),
            epoch=epoch+1,operating_point=selected,model_config=cfg['model'],smoke_test=bool(args.max_steps or args.val_limit))
        if is_best:
            # Inference checkpoint contains selected EMA only. Keep complete restart state in last.pth.
            # best.pth 仅保存选中的 EMA 推理模型及工作点；继续训练应使用完整的 last.pth。
            store.checkpoint('best.pth',inference)
            save_json(store.path/'reports/operating_point.json',selected)
        if is_best_ap40:
            store.checkpoint('best_AP40.pth',inference)
        frequency=int(cfg['train'].get('checkpoint_frequency',0) or 0)
        if frequency and (epoch+1)%frequency==0:
            # Source naming is zero-based: epoch 10 is checkpoint0009.pth.
            store.checkpoint(f'checkpoint{epoch:04d}.pth',inference)
        if best_ap40 is not None:
            save_json(store.path/'reports/best_metrics.json',dict(
                selection_order=['Recall_IoU40','Recall_IoU50','Recall_IoU60','AP40'],
                best_recall=best,best_AP40=best_ap40,
            ))
        print('EPOCH '+__import__('json').dumps(event,ensure_ascii=False),flush=True)
    result=dict(path=str(store.path),completed_epoch=epochs,best=best,best_AP40=best_ap40)
    # TensorRT selection is an infrared-recipe option. Smoke runs never build
    # engines or replace weights, so other tasks and quick checks are unchanged.
    if (cfg.get('tensorrt_selection') or {}).get('enabled') and not (args.max_steps or args.val_limit):
        from src.engine.infrared_trt_selection import select_checkpoints
        result['tensorrt_selection']=select_checkpoints(cfg,adapter,store.path,args.device)
    return result
