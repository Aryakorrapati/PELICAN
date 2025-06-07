import torch
from .utils import all_gather, synchronize, get_world_size
import numpy as np
from itertools import islice
#from .scheduler import GradualWarmupScheduler, GradualCooldownScheduler
from torch.optim.lr_scheduler import OneCycleLR  #Dyanmic learning rate implementation
import matplotlib.pyplot as plt #Plot learning rate
from IPython.display import display, clear_output
import os
from datetime import datetime
from math import inf, ceil
import logging
logger = logging.getLogger(__name__)
from src.trainer.utils import apply_mixup_cutmix

class Trainer:
    """
    Class to train network. Includes checkpoints, optimizer, scheduler,
    """
    def __init__(self, args, dataloaders, model, loss_fn, metrics_fn, minibatch_metrics_fn, minibatch_metrics_string_fn, 
                 optimizer, scheduler, restart_epochs, device_id, device, dtype, trial_number=None):
        np.set_printoptions(precision=5)
        self.args = args
        self.dataloaders = dataloaders
        self.model = model
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn
        self.minibatch_metrics_fn = minibatch_metrics_fn
        self.minibatch_metrics_string_fn = minibatch_metrics_string_fn
        self.optimizer = optimizer
        self.scheduler = scheduler #For dyanmic scheduler
        self.restart_epochs = restart_epochs
        self.trial_number = trial_number

        self.lr_history = [] #Store learning rates for plotting
        self.loss_history = []

        self.summarize_csv =args.summarize_csv
        self.summarize = args.summarize
        if args.summarize:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=args.logdir+args.prefix)

        self.epoch = 1
        self.minibatch = 0
        self.minibatch_metrics = {}
        self.best_epoch = 0
        self.best_metrics = {'loss': inf}

        self.device_id = device_id
        self.device = device
        self.dtype = dtype
        self.best_val_acc = -float('inf')
        self.best_acc_epoch = 0
        self.best_acc_metrics = None

    def _wrap_scheduler(self):
        return OneCycleLR(
        self.optimizer, 
        max_lr=self.args.max_lr,  # Adjust max learning rate
        steps_per_epoch=len(self.dataloaders['train']), 
        epochs=self.args.num_epoch,
        pct_start=self.args.pct_start,  # 20% of training is warmup
        anneal_strategy='cos',  # Cosine decay
        final_div_factor=100  # Reduce learning rate at the end
    )

    def _save_checkpoint(self, valid_metrics=None):
        if not self.args.save:
            return

        save_dict = {'args': self.args,
                     'model_state': self.model.state_dict(),
                     'optimizer_state': self.optimizer.state_dict(),
                     'scheduler_state': self.scheduler.state_dict(),
                     'epoch': self.epoch,
                     'minibatch': self.minibatch,
                     'minibatch_metrics': self.minibatch_metrics,
                     'best_epoch': self.best_epoch,
                     'best_metrics': self.best_metrics}

        if valid_metrics is None:
            logger.info('Saving model to checkpoint file: {}'.format(self.args.checkfile))
            torch.save(save_dict, self.args.checkfile)
        elif valid_metrics['loss'] < self.best_metrics['loss']: # and self.epoch/self.args.num_epoch >= 0.5:
            self.best_epoch = self.epoch
            self.best_metrics = save_dict['best_metrics'] = valid_metrics
            logger.info('Lowest loss achieved! Saving best model to file: {}'.format(self.args.bestfile))
            torch.save(save_dict, self.args.bestfile)


        if valid_metrics and 'accuracy' in valid_metrics and valid_metrics['accuracy'] > self.best_val_acc:
            self.best_val_acc = valid_metrics['accuracy']
            self.best_acc_epoch = self.epoch
            self.best_acc_metrics = valid_metrics.copy()
            logger.info(f'New best accuracy: {self.best_val_acc:.4f} at epoch {self.best_acc_epoch}, saving to {self.args.bestaccfile}')
            torch.save({
                'args': self.args,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'epoch': self.epoch,
                'minibatch': self.minibatch,
                'minibatch_metrics': self.minibatch_metrics,
                'best_acc_epoch': self.best_acc_epoch,
                'best_acc_metrics': self.best_acc_metrics,
            }, self.args.bestaccfile)


    def load_checkpoint(self):
        """
        Load checkpoint from previous file.
        """
        if not self.args.load:
            return
        elif os.path.exists(self.args.checkfile):
            logger.info('Loading previous model from checkpoint!')
            self.load_state(self.args.checkfile)
            if self.minibatch + 1  < len(self.dataloaders['train']):
                self.minibatch += 1
            else:
                self.minibatch = 0
                self.epoch += 1
        else:
            logger.info('No checkpoint included! Starting fresh training program.')
            return

    def load_state(self, checkfile):
        logger.info('Loading from checkpoint!')

        checkpoint = torch.load(checkfile, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.epoch = checkpoint['epoch']
        self.minibatch = checkpoint['minibatch']        
        self.minibatch_metrics = checkpoint['minibatch_metrics']
        self.best_epoch = checkpoint['best_epoch']
        self.best_metrics = checkpoint['best_metrics']
        del checkpoint

        logger.info(f'Loaded checkpoint at epoch {self.epoch}.\nBest metrics from checkpoint are at epoch {self.best_epoch}:\n{self.best_metrics}')

    def save_best_acc_metrics_csv(self):
        """Save best accuracy metrics to classifier.best.accuracy.csv"""
        if self.best_acc_metrics is not None:
            filename = os.path.join(self.args.workdir, self.args.logdir, "classifier.best.accuracy.csv")
            header_written = os.path.exists(filename)
            with open(filename, 'a' if header_written else 'w') as f:
                if not header_written:
                    f.write("epoch," + ",".join(self.best_acc_metrics.keys()) + "\n")
                f.write(f"{self.best_acc_epoch}," + ",".join(str(v) for v in self.best_acc_metrics.values()) + "\n")
            logger.info(f"Best accuracy metrics saved to {filename}")


    def evaluate(self, splits=['train', 'valid', 'test'], distributed=False, best=True, final=True, ir_data=None, c_data=None, expand_data=None):
        """
        Evaluate model on splits (in practice used only for final testing).

        :splits: List of splits to include. Only valid splits are: 'train', 'valid', 'test'
        :distributed: Evaluate using multiple GPUs. Off by default so that the order of datapoints is preserved.
        :best: Evaluate best model as determined by minimum validation loss over evolution
        :final: Evaluate final model at end of training phase
        """
        if not self.args.save:
            logger.info('No model saved! Cannot give final status.')
            return
        
        # make sure main process has finished writing to disk before proceeding
        synchronize()
        if distributed:
            logger.warning('Using distributed evaluation for the testing dataset (args.distribute_eval). Order of samples may not be preserved in the predict file.')
        elif self.device_id > 0:
            logger.info(f'Evaluating only on device 0. Quitting on device {self.device_id}\n')
            return

        # Evaluate final model
        if final:
            # Load checkpoint model to make predictions
            checkpoint = torch.load(self.args.checkfile, map_location=self.device)
            final_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state'])
            logger.info(f'Getting predictions for final model {self.args.checkfile} (epoch {final_epoch}).')

            # Loop over splits, predict, and output/log predictions
            for split in splits:
                predict, targets = self.predict(set=split, distributed=distributed, ir_data=ir_data, c_data=c_data, expand_data=expand_data)
                if self.device_id <= 0:
                    best_metrics, logstring = self.log_predict(predict, targets, split, description='Final')
                synchronize()
        # Evaluate best model as determined by validation error
        if best:
            # Load best model to make predictions
            checkpoint = torch.load(self.args.bestfile, map_location=self.device)
            best_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state'])
            if (not final) or (final and not best_epoch == final_epoch):
                logger.info(f'Getting predictions for best model {self.args.bestfile} (epoch {best_epoch}, best validation metrics were {checkpoint["best_metrics"]}).')
                # Loop over splits, predict, and output/log predictions
                for split in splits:
                    predict, targets = self.predict(split, distributed=distributed, ir_data=ir_data, c_data=c_data, expand_data=expand_data)
                    if self.device_id <= 0:
                        best_metrics, logstring = self.log_predict(predict, targets, split, description='Best')
                    synchronize()

            elif best_epoch == final_epoch:
                logger.info('BEST MODEL IS SAME AS FINAL')
                if self.device_id <= 0:
                    self.log_predict(predict, targets, split, description='Best', repeat=[best_metrics, logstring])
        logger.info('Inference phase complete!\n')

        # Evaluate best accuracy model
        if hasattr(self.args, "bestaccfile") and os.path.exists(self.args.bestaccfile):
            checkpoint = torch.load(self.args.bestaccfile, map_location=self.device)
            best_acc_epoch = checkpoint.get('best_acc_epoch', -1)
            self.model.load_state_dict(checkpoint['model_state'])
            logger.info(f"Getting predictions for best-accuracy model {self.args.bestaccfile} (epoch {best_acc_epoch}, best accuracy metrics were {checkpoint.get('best_acc_metrics', {})}).")
            for split in splits:
                predict, targets = self.predict(split, distributed=distributed, ir_data=ir_data, c_data=c_data, expand_data=expand_data)
                if self.device_id <= 0:
                    best_acc_metrics, logstring = self.log_predict(predict, targets, split, description='BestAcc')
                synchronize()
            # After best-accuracy evaluation:
            self.save_best_acc_metrics_csv()



    def _warm_restart(self, epoch):
        restart_epochs = self.restart_epochs

        if epoch in restart_epochs:
            logger.info('Warm learning rate restart at epoch {}!'.format(epoch))
            self.scheduler.last_epoch = 1
            idx = restart_epochs.index(epoch)
            self.scheduler.T_max = restart_epochs[idx + 1] - restart_epochs[idx]
            if self.args.lr_minibatch:
                self.scheduler.T_max *= ceil(self.args.num_train / (self.args.back_batch_size if self.args.back_batch_size is not None else self.args.batch_size))
            self.scheduler.step(0)

    def _log_minibatch(self, batch_idx, loss, targets, predict, batch_t, fwd_t, bwd_t, epoch_t):
        mini_batch_loss = loss.item()
        minibatch_metrics = self.minibatch_metrics_fn(predict, targets, mini_batch_loss)

        if batch_idx == 0:
            self.minibatch_metrics = minibatch_metrics
        else:
            """ 
            Exponential average of recent Loss/AltLoss on training set for more convenient logging.
            alpha must be a positive real number. 
            alpha = 0 corresponds to no smoothing ("average over 0 previous minibatchas")
            alpha > 0 will produce smoothing where the weight of the k-to-last minibatch is proportional to exp(-gamma * k)
                    where gamma = - log(alpha/(1 + alpha)). At large alpha the weight is approx. exp(- k/alpha).
                    The number of minibatches that contribute significantly at large alpha scales like alpha.
            """
            alpha = self.args.alpha
            assert alpha >= 0, "alpha must be a nonnegative real number"
            alpha = alpha / (1 + alpha)
            for i, metric in enumerate(minibatch_metrics):
                self.minibatch_metrics[i] = alpha * self.minibatch_metrics[i] + (1 - alpha) * metric

        dt_batch = (datetime.now() - batch_t).total_seconds()
        dt_fwd = (fwd_t - batch_t).total_seconds()
        dt_bwd = (bwd_t - fwd_t).total_seconds()
        tepoch = (datetime.now() - epoch_t).total_seconds()
        self.batch_time += dt_batch
        tcollate = tepoch - self.batch_time

        if self.args.textlog:
            logstring = self.args.prefix + ' E:{:3}/{}, B: {:5}/{}'.format(self.epoch, self.args.num_epoch, batch_idx + 1, len(self.dataloaders['train']))
            logstring += self.minibatch_metrics_string_fn(self.minibatch_metrics)
            logstring += '  dt:({:> 4.3f}+{:> 4.3f}={:> 4.3f}){:> 8.2f}{:> 8.2f}'.format(dt_fwd, dt_bwd, dt_batch, tepoch, tcollate)
            logstring += '  {:.2E}'.format(self.scheduler.get_last_lr()[0])
            if self.args.verbose:
                logger.info(logstring)
            else:
                print(logstring)


    #def _step_lr_batch(self):
        #self.scheduler.step()
        #self.scheduler.step()
        #self.lr_history.append(self.scheduler.get_last_lr()[0]) #Track learning rate for plotting

    def _step_lr_epoch(self):
        #if not self.args.lr_minibatch:
            #self.scheduler.step()
        pass

    #Plots the learning rate schedule over training
    def plot_lr(self):
        plt.figure(figsize=(8,5))
        plt.plot(self.lr_history, label="Learning Rate", color='blue')
        plt.xlabel("Batch Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.legend()
        plt.grid(True)
        plt.show()

    def train(self, trial=None, metric_to_report='loss'):
        start_epoch = self.epoch
        start_minibatch = self.minibatch
        patience = 10  # Stop if validation loss doesn't improve for 5 epochs
        best_val_loss = float('inf')  # Track best validation loss
        patience_counter = 0  # Counter for epochs with no improvement
        for epoch in range(start_epoch, self.args.num_epoch + 1):
            self.epoch = epoch
            if epoch > start_epoch:
                start_minibatch = 0
            if get_world_size() > 1:
                self.dataloaders['train'].batch_sampler.sampler.set_epoch(epoch)
            logger.info(f'STARTING Epoch {epoch} from minibatch {start_minibatch+1}')

            self._warm_restart(epoch)
            self._step_lr_epoch()
            train_predict, train_targets, epoch_t = self.train_epoch(start_minibatch)
            if self.device_id <= 0:
                self._save_checkpoint()
                train_metrics,_ = self.log_predict(train_predict, train_targets, 'train', epoch=epoch, epoch_t=epoch_t)

            valid_predict, valid_targets = self.predict(set='valid', distributed=True)

            if self.device_id <= 0:
                self._save_checkpoint()
                valid_metrics, _ = self.log_predict(valid_predict, valid_targets, 'valid', epoch=epoch)
                self._save_checkpoint(valid_metrics)

                # Extract validation loss
                current_val_loss = valid_metrics['loss']

                # Check if validation loss improved
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0  # Reset counter
                    logger.info(f"Validation loss improved to {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"Validation loss did not improve for {patience_counter}/{patience} epochs")

                # Stop training if validation loss hasn't improved for 'patience' epochs
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}. Stopping training.")
                    break
            synchronize()
            
            if trial:
                trial.set_user_attr("best_epoch", self.best_epoch)
                trial.set_user_attr("best_metrics", self.best_metrics)
                trial.report(min(valid_metrics[metric_to_report], 1), epoch - 1)
                if trial.should_prune():
                    import optuna
                    raise optuna.exceptions.TrialPruned()

            logger.info(f'FINISHED Epoch {epoch}\n_________________________\n')
            if hasattr(self.loss_fn, "update_epoch"):
                self.loss_fn.update_epoch(epoch)
            
        if self.summarize: self.writer.close()

        # Plot Training Loss and Learning Rate Schedule
        if self.device_id <= 0: 
            plt.figure(figsize=(12, 5))

        # Plot Training Loss

            plt.subplot(1, 2, 1)
            plt.plot(self.loss_history, label="Training Loss", color="blue")
            plt.xlabel("Batch")
            plt.ylabel("Loss")
            plt.title(f"Training Loss Over Time (Trial {self.trial_number})")
            plt.legend()

        # Plot Learning Rate Schedule
            plt.subplot(1, 2, 2)
            plt.plot(self.lr_history, label="Learning Rate", color="red")
            plt.xlabel("Batch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Over Time")
            plt.legend()

        # Show & Save the Figure
            plt.savefig("training_progress.png") 
            plt.show()

        return self.best_epoch, self.best_metrics

    def _get_target(self, data):
        """
        Get the learning target.
        If a stats dictionary is included, return a normalized learning target.
        """        
        target_type = torch.long if self.args.target=='is_signal' else self.dtype
        targets = data[self.args.target].to(self.device, target_type)
        return targets

    def train_epoch(self, start_minibatch=0):
        dataloader = self.dataloaders['train']

        self.loss_val, self.alt_loss_val, self.batch_time = 0, 0, 0
        all_predict, all_targets = {}, []

        self.model.train()
        epoch_t = datetime.now()
        if start_minibatch > 0:
            dataloader.dataset.fast_skip = True
            data_slice = islice(enumerate(dataloader), start_minibatch - 1, None)
            logger.info(f'Iterating dataloader to get to minibatch {start_minibatch + 1}')
            _, _ = next(data_slice)
            logger.info('Starting the training loop')
            dataloader.dataset.fast_skip = False
        else:
            data_slice = enumerate(dataloader)
        for batch_idx, data in data_slice:
            self.minibatch = batch_idx
            batch_t = datetime.now()

            inputs = data["Pmu"].to(self.device)
            targets = self._get_target(data)

            # Apply Mixup or CutMix if enabled
            if getattr(self.args, 'aug_mixcut', False):
                alpha = getattr(self.args, 'aug_alpha', 0.5)
                targets_a, targets_b, lam = targets, targets, 1.0
                data["inputs"] = inputs  # update input batch with augmented one
                mixcut = True
            else:
                mixcut = False

            predict = self.model(data)
            fwd_t = datetime.now()

            # Compute loss
            loss = self.loss_fn(predict['predict'], targets)

            self.optimizer.zero_grad()
            loss.backward()
            bwd_t = datetime.now()

            # Calculate loss and backprop
            #self.optimizer.zero_grad()
            #loss.backward()
            #bwd_t = datetime.now()

             # Update optimizer and scheduler (OneCycleLR will dynamically adjust LR)
            self.optimizer.step()
            self.scheduler.step()

            # Store loss and learning rate for plotting
            self.loss_history.append(loss.item())
            self.lr_history.append(self.scheduler.get_last_lr()[0])

            if not self.args.quiet and not all(param.grad is not None for param in dict(self.model.named_parameters()).values()):
                logger.warning("The following params have missing gradients at backward pass (they are probably not being used in output):\n", {key: '' for key, param in self.model.named_parameters() if param.grad is None})
            # Step optimizer and learning rate
            #self.optimizer.step()
            # self.model.apply(_max_norm)
            #self._step_lr_batch()

            targets = all_gather(targets).detach().cpu()
            predict = {key: all_gather(val).detach().cpu() for key, val in predict.items()}
            if self.device_id <= 0:
                all_targets.append(targets)
                for key, val in predict.items(): all_predict.setdefault(key,[]).append(val)
                if (self.args.save_every > 0) and (batch_idx + 1) % self.args.save_every == 0:
                    self._save_checkpoint()
                if (self.args.log_every > 0) and batch_idx % self.args.log_every == 0:
                    self._log_minibatch(batch_idx, loss, targets, predict['predict'], batch_t, fwd_t, bwd_t, epoch_t)
        
        if self.device_id > 0:
            return None, None, epoch_t

        all_predict = {key: torch.cat(val) for key, val in all_predict.items()}
        all_targets = torch.cat(all_targets)

        return all_predict, all_targets, epoch_t

    def predict(self, set='valid', distributed=True, ir_data=None, c_data=None, expand_data=None):
        dataloader = self.dataloaders[set]

        self.model.eval()
        all_predict, all_targets = {}, []
        start_time = datetime.now()
        logger.info('Starting testing on {} set: '.format(set))

        if not distributed:
            if self.device_id > 0:
                return None, None
            gather = lambda x: x
        else:
            gather = all_gather
        
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                if expand_data:
                    data = expand_data(data)
                if ir_data is not None:
                    data = ir_data(data)
                if c_data is not None:
                    data = c_data(data)
                
                targets = gather(self._get_target(data)).detach().cpu()
                predict = self.model(data)
                predict = {key: gather(val).detach().cpu() for key, val in predict.items()}
                if self.device_id <= 0:
                    all_targets.append(targets)
                    for key, val in predict.items(): all_predict.setdefault(key, []).append(val)

        if self.device_id > 0:
            return None, None
        
        if all_targets[0] is not None:
            all_targets = torch.cat(all_targets)
        else:
            all_targets = None
        all_predict = {key: torch.cat(val) for key, val in all_predict.items()}

        dt = (datetime.now() - start_time).total_seconds()
        logger.info('Total evaluation time: {}s'.format(dt))

        return all_predict, all_targets

    def log_predict(self, predict, targets, dataset, epoch=-1, epoch_t=None, description='', repeat=None):

        datastrings = {'train': 'Training  ', 'test': 'Testing   ', 'valid': 'Validation'}

        if epoch >= 0:
            suffix = 'final'
        else:
            suffix = 'best'

        prefix = self.args.predictfile + '.' + suffix + '.' + dataset
        metrics, logstring = None, 'Metrics skipped because target is None!'

        predict = {key: val.cpu().double() for key, val in predict.items()}

        if targets is not None:
            targets = targets.cpu().double()

            if repeat is None:
                metrics, logstring = self.metrics_fn(predict['predict'], targets, self.loss_fn, prefix, logger)
            else:
                metrics, logstring = repeat[0], repeat[1]

            if epoch >= 0:
                logger.info(f'Epoch {epoch} {description} {datastrings[dataset]}'+logstring)
            else:
                logger.info(f'{description:<5} {datastrings[dataset]:<10}'+logstring)

            if epoch_t:
                logger.info(f'Total epoch time:      {(datetime.now() - epoch_t).total_seconds()}s')

            metricsfile = self.args.predictfile + '.metrics.' + dataset + '.csv'   
            testmetricsfile = os.path.join(self.args.workdir, self.args.logdir, self.args.prefix.split("-")[0]+'.'+description+'.metrics.csv')

            if epoch >= 0 and self.summarize_csv=='all':
                with open(metricsfile, mode='a' if (self.args.load or epoch>1) else 'w') as file_:
                    if epoch == 1:
                        file_.write(",".join(metrics.keys()))
                    file_.write(",".join(map(str, metrics.values())))
                    file_.write("\n")  # Next line.
            if epoch < 0 and self.summarize_csv in ['test','all']:
                if not os.path.exists(testmetricsfile):
                    with open(testmetricsfile, mode='a') as file_:
                        file_.write("prefix,timestamp,"+",".join(metrics.keys()))
                        file_.write("\n")  # Next line.
                with open(testmetricsfile, mode='a') as file_:
                    file_.write(self.args.prefix+','+str(datetime.now())+','+",".join(map(str, metrics.values())))
                    file_.write("\n")  # Next line.

            if self.summarize:
                if description == 'Best': dataset = 'Best_' + dataset
                if description == 'Final': dataset = 'Final_' + dataset
                for name, metric in metrics.items():
                    if not isinstance(metric, np.ndarray):
                        self.writer.add_scalar(dataset+'/'+name, metric, epoch)
                    else:
                        if metric.size==1:
                            self.writer.add_scalar(dataset+'/'+name, metric, epoch)
                        else:
                            for i, m in enumerate(metric.flatten()):
                                self.writer.add_scalar(dataset+'/'+name+'_'+str(i), m, epoch)


        if self.args.predict and (repeat is None):
            file = self.args.predictfile + '.' + suffix + '.' + dataset + '.pt'
            logger.info('Saving predictions to file: {}'.format(file))
            if targets is not None:
                predict.update({'targets': targets})
            torch.save(predict, file)
        
        if repeat is not None:
            logger.info('Predictions already saved above')

        return metrics, logstring
    
    
