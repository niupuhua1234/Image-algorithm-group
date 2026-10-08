'''
by lyuwenyu
'''
import time 
import json
import datetime

import torch 
from torch.utils.tensorboard import SummaryWriter

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }
        history_path = self.output_dir / 'log.txt' if self.output_dir else None
        if self.last_epoch >= 0 and history_path is not None and history_path.exists():
            for line in history_path.read_text().splitlines():
                try:
                    history = json.loads(line)
                    bbox_stats = history.get('test_coco_eval_bbox', [])
                    history_ap = bbox_stats[0] if bbox_stats else -1
                    if history_ap > best_stat.get('coco_eval_bbox', -1):
                        best_stat['coco_eval_bbox'] = history_ap
                        best_stat['epoch'] = history.get('epoch', -1)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
            print('Restored historical best_stat: ', best_stat)
        writer = SummaryWriter(self.output_dir / 'tensorboard') if dist.is_main_process() else None
        if writer is not None:
            print(f'TensorBoard log directory: {self.output_dir / "tensorboard"}')

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()
            
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            current_ap = test_stats.get('coco_eval_bbox', [-1])[0]
            is_best = current_ap > best_stat.get('coco_eval_bbox', -1)
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat[k] = test_stats[k][0]
            if is_best:
                best_stat['epoch'] = epoch
            print('best_stat: ', best_stat)

            if is_best and self.output_dir:
                dist.save_on_master(self.state_dict(epoch), self.output_dir / 'best.pth')


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            bbox_stats = test_stats.get('coco_eval_bbox', [])
            if dist.is_main_process():
                map_50_95 = bbox_stats[0] if len(bbox_stats) > 0 else float('nan')
                map_50 = bbox_stats[1] if len(bbox_stats) > 1 else float('nan')
                print(
                    f'Epoch summary [{epoch + 1}/{args.epoches}] '
                    f'loss={train_stats.get("loss", float("nan")):.6f} '
                    f'mAP50:95={map_50_95:.6f} mAP50={map_50:.6f} '
                    f'best_epoch={best_stat["epoch"] + 1} '
                    f'best_mAP50:95={best_stat.get("coco_eval_bbox", float("nan")):.6f}'
                )
                for name, value in train_stats.items():
                    writer.add_scalar(f'train/{name}', value, epoch + 1)
                coco_names = (
                    'mAP_50_95', 'mAP_50', 'mAP_75', 'mAP_small', 'mAP_medium', 'mAP_large',
                    'mAR_1', 'mAR_10', 'mAR_100', 'mAR_small', 'mAR_medium', 'mAR_large',
                )
                for name, value in zip(coco_names, bbox_stats):
                    writer.add_scalar(f'val/{name}', value, epoch + 1)
                writer.add_scalar('val/best_mAP_50_95', best_stat.get('coco_eval_bbox', -1), epoch + 1)
                writer.flush()

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        if writer is not None:
            writer.close()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
