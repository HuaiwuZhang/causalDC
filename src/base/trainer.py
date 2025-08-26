import logging
import os
import time
from typing import Optional, List, Union
from collections import defaultdict
import numpy as np
from src.utils.logging import get_logger
import torch
from torch import nn, Tensor
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
from tqdm import tqdm
from src.utils import metrics as mc

import pandas as pd
import wandb



class BaseTrainer():
    def __init__(
            self,
            model: nn.Module,
            data,
            args,
            device: Optional[Union[torch.device, str]] = None,


    ):
        super().__init__()

        if device is None:
            print("`device` is missing, try to train and evaluate the model on default device.")
            if torch.cuda.is_available():
                print("cuda device is available, place the model on the device.")
                self._device = torch.device("cuda")
            else:
                print("cuda device is not available, place the model on cpu.")
                self._device = torch.device("cpu")
        else:
            if isinstance(device, torch.device):
                self._device = device
            else:
                self._device = torch.device(device)
        self._logger = get_logger(
            args.log_dir, __name__, 'info_{}.log'.format(args.n_exp), level=logging.INFO)
        self._model = model
        self._wandb_flag = args.wandb
        self.num_param = self.model.param_num(self.model.name)
        self._logger.info("the number of parameters: {}".format(self.num_param))
        if args.wandb:
            wandb.run.summary["Params"] = self.num_param

        self.model.to(self._device)
        self._data = data
        self.args = args
        print(args.task)
        if args.task == 'regression':
            self._loss_criterion = nn.MSELoss()
        elif args.task == 'classification':
            self._loss_criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("task error")
        self._base_lr = args.base_lr
        self._optimizer = Adam(self.model.parameters(), args.base_lr)
        self._lr_decay_ratio = args.lr_decay_ratio
        self._steps = args.steps
        if args.lr_decay_ratio == 1:
            self._lr_scheduler = None
        else:
            self._lr_scheduler = MultiStepLR(self.optimizer,
                                             args.steps,
                                             gamma=args.lr_decay_ratio)
        self._clip_grad_value = args.max_grad_norm
        self._max_epochs = args.max_epochs
        self._patience = args.patience
        self._save_iter = args.save_iter
        self._save_path = args.log_dir
        self._n_exp = args.n_exp
        self._data = data

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def loss_criterion(self):
        return self._loss_criterion

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def data(self):
        return self._data

    @property
    def logger(self):
        return self._logger

    @property
    def save_path(self):
        return self._save_path

    def _check_device(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)

    def _to_numpy(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.cpu().detach().numpy() for tensor in tensors]
        else:
            return tensors.cpu().detach().numpy()

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [Tensor(array) for array in nparray]
        else:
            return Tensor(nparray)

    def save_model(self, epoch, save_path, n_exp, val_loss, temp_lr_scheduler, stage):
        save_path = save_path + '/' + stage
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_{}.pt'.format(n_exp)
        torch.save({'model_state_dict':self.model.state_dict(),
                    'epoch':epoch,
                    'val_loss':val_loss,
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler':temp_lr_scheduler.state_dict(),
                    "numpy_random": np.random.get_state(),
                    "torch_random": torch.get_rng_state(),
                    "torch_cuda_random": torch.cuda.get_rng_state_all()
                    }, os.path.join(save_path, filename))
        return True

    def load_model(self, save_path, n_exp, stage):
        save_path = save_path + '/' + stage
        filename = 'final_model_{}.pt'.format(n_exp)
        checkpoint = torch.load(os.path.join(save_path, filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        val_loss = checkpoint['val_loss']
        optimizer = checkpoint['optimizer']
        lr_scheduler = checkpoint['lr_scheduler']
        np.random.set_state(checkpoint["numpy_random"])
        torch.set_rng_state(checkpoint["torch_random"])
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random"])

        return epoch, val_loss, optimizer, lr_scheduler

    def early_stop(self, epoch, best_loss):
        self.logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch, best_loss))
        np.savetxt(os.path.join(self.save_path, 'val_loss_{}.txt'.format(self._n_exp)), [best_loss], fmt='%.4f', delimiter=',')

    def train_batch(self, X, label, iter):
        self.optimizer.zero_grad()
        pred = self.model(*X)
        if self.args.model == 'lasso':
            loss = self.loss_criterion(pred.view(-1), label.view(-1)) + self.model.l1_regularization()
        elif self.args.model == 'svm':
            loss = self.epsilon_insensitive_loss(pred.view(-1), label.view(-1))
        else:
            loss = self.loss_criterion(pred.view(-1), label.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=self._clip_grad_value)
        self.optimizer.step()
        return loss.item()
    
    def epsilon_insensitive_loss(self, y_pred, y_true, epsilon=0.1):
        residual = torch.abs(y_true - y_pred)
        loss = torch.where(residual < epsilon, torch.zeros_like(residual), residual - epsilon)
        return torch.mean(loss)


    def pretrain(self, stage):
        print("start training !!!!!")
        iter = 0
        val_losses = [np.inf]
        saved_epoch = -1
        existing_epoch = -1
        try:
            existing_epoch, val_loss, optimizer, lr_scheduler = self.load_model(self.save_path, self._n_exp, stage)
            self.optimizer.load_state_dict(optimizer)
            self.lr_scheduler.load_state_dict(lr_scheduler)
            saved_epoch = existing_epoch
            val_losses.append(val_loss)
            print('Existing pretrained model')
        except:
            print('No existing pretrained model')

        for epoch in range(max(existing_epoch, -1)+1,self._max_epochs):
            self.model.train()
            train_losses = []
            if epoch - saved_epoch > self._patience:
                if self._wandb_flag:
                    wandb.run.summary[stage + " final_valid_loss"] = min(val_losses)
                    wandb.run.summary[stage + " saved_epoch"] = saved_epoch
                self.early_stop(epoch, min(val_losses))
                break

            start_time = time.time()
            for id, (*X, label) in tqdm(enumerate(self.data['train_loader']), total=len(self.data['train_loader'])):

                X = self._check_device(X)
                label = self._check_device(label)
                train_losses.append(self.train_batch(X, label, iter))
                iter += 1
            end_time = time.time()

            val_loss = self.evaluate()

            if self.lr_scheduler is None:
                new_lr = self._base_lr
            else:
                new_lr = self.lr_scheduler.get_last_lr()[0]

            message = (
                "Epoch [{}/{}] ({}) train_mse: {:.4f}, val_mse: {:.4f}, lr: {:.6f}, "
                "{:.1f}s".format(
                    epoch,
                    self._max_epochs,
                    iter,
                    np.mean(train_losses),
                    val_loss,
                    new_lr,
                    (end_time - start_time),
                )
            )
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self._logger.info(message)

            if val_loss < np.min(val_losses):
                temp_lr_scheduler = self.lr_scheduler
                model_file_name = self.save_model(
                    epoch, self._save_path, self._n_exp, val_loss, temp_lr_scheduler, stage = 'pretrain')
                self._logger.info(
                    'Val loss decrease from {:.4f} to {:.4f}, '
                    'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                val_losses.append(val_loss)
                saved_epoch = epoch

            if self._wandb_flag:
                wandb_key1 = stage + '_val_loss'
                wandb_key2 = stage + '_train_loss'
                wandb.log({wandb_key1: val_loss, stage + " epoch": epoch}, step=epoch)
                wandb.log({wandb_key2: np.mean(train_losses), stage + " epoch": epoch}, step=epoch)
                wandb.run.summary[stage + " final_valid_loss"] = min(val_losses)

    def evaluate(self):
        labels = []
        preds = []
        with torch.no_grad():
            self.model.eval()
            for _, (*X, label) in tqdm(enumerate(self.data['val_loader']), total=len(self.data['val_loader'])):
                X = self._check_device(X)
                label = self._check_device(label)
                pred, label = self.test_batch(X, label)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        loss = self.loss_criterion(preds, labels).item()
        return loss

    def test_batch(self, X, label):
        pred = self.model(*X)
        return pred, label

    def test(self, stage, direct_finetune=False, mode='test'):
        if direct_finetune:
            stage = 'direct_finetune'
        _ = self.load_model(self.save_path, self._n_exp, stage=stage)

        labels = []
        preds = []

        with torch.no_grad():
            self.model.eval()
            for _, (*X, label) in tqdm(enumerate(self.data[mode + '_loader']), total=len(self.data[mode + '_loader'])):
                X = self._check_device(X)
                label = self._check_device(label)
                pred, label = self.test_batch(X, label)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        preds, labels = preds.numpy().flatten(), labels.numpy().flatten()
        metrics = mc.compute_reg_metrics(preds, labels)
        log = '====Using evaluate function for test data -- Test MAE: {:.4f}, Test RMSE: {:.4f}, Test R2: {:.4f}, Test PCC: {:.4f}, ===='
        print(log.format(metrics[0], metrics[1], metrics[2], metrics[3]))

        mae = metrics[0]
        rmse = metrics[1]
        r2 = metrics[2]
        pcc = metrics[3]

        if self._wandb_flag:
            wandb.run.summary[stage + " test MAE"] = mae
            wandb.run.summary[stage + " test RMSE"] = rmse
            wandb.run.summary[stage + " test R2"] = r2
            wandb.run.summary[stage + " test PCC"] = pcc

        results = np.stack([mae, rmse, r2, pcc], axis=0)
        np.savetxt(os.path.join(self.save_path, 'results_{}.csv'.format(self._n_exp)), results, fmt='%.4f',
                   delimiter=',')

        return mae, rmse, r2, pcc

    def finetune(self, stage, direct_finetune=False):
        print("start finetuning !!!!!")
        iter = 0
        val_losses = [np.inf]
        saved_epoch = -1
        existing_epoch = -1
        if direct_finetune:
            print('direct finetuning')
            stage = 'direct_finetune'
        else:
            try:
                _ = self.load_model(self.save_path, self._n_exp, stage = 'pretrain')
                print('Existing pretrained model')
            except:
                raise ValueError("No existing pretrained model")

        try:
            existing_epoch, val_loss, optimizer, lr_scheduler = self.load_model(self.save_path, self._n_exp,
                                                                                stage=stage)
            self.optimizer.load_state_dict(optimizer)
            self.lr_scheduler.load_state_dict(lr_scheduler)
            saved_epoch = existing_epoch
            val_losses.append(val_loss)
            print('Existing finetuned model')
        except:
            print("No existing finetuned model")

        for epoch in range(max(existing_epoch, -1)+1,self._max_epochs):
            self.model.train()
            train_losses = []
            if epoch - saved_epoch > self._patience:
                if self._wandb_flag:
                    wandb.run.summary[stage + " final_valid_loss"] = min(val_losses)
                    wandb.run.summary[stage + " saved_epoch"] = saved_epoch
                self.early_stop(epoch, min(val_losses))
                break
            start_time = time.time()
            for id, (*X, label) in tqdm(enumerate(self.data['test_loader']), total=len(self.data['test_loader'])):
                X = self._check_device(X)
                label = self._check_device(label)
                train_losses.append(self.train_batch(X, label, iter))
                iter += 1
            end_time = time.time()
            val_loss = self.evaluate()
            if self.lr_scheduler is None:
                new_lr = self._base_lr
            else:
                new_lr = self.lr_scheduler.get_last_lr()[0]

            message = (
                "Epoch [{}/{}] ({}) train_mse: {:.4f}, val_mse: {:.4f}, lr: {:.6f}, "
                "{:.1f}s".format(
                    epoch,
                    self._max_epochs,
                    iter,
                    np.mean(train_losses),
                    val_loss,
                    new_lr,
                    (end_time - start_time),
                )
            )
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self._logger.info(message)

            if val_loss < np.min(val_losses):
                temp_lr_scheduler = self.lr_scheduler
                model_file_name = self.save_model(
                    epoch, self._save_path, self._n_exp, val_loss, temp_lr_scheduler, stage = stage)
                self._logger.info(
                    'Val loss decrease from {:.4f} to {:.4f}, '
                    'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                val_losses.append(val_loss)
                saved_epoch = epoch

            if self._wandb_flag:
                wandb_key1 = stage + '_val_loss'
                wandb_key2 = stage + '_train_loss'
                wandb.log({wandb_key1: val_loss, stage + " epoch": epoch}, step=epoch)
                wandb.log({wandb_key2: np.mean(train_losses), stage + " epoch": epoch}, step=epoch)
                wandb.run.summary[stage + " final_valid_loss"] = min(val_losses)
    
    def train_batch_causal(self, X, label, iter):
        self.optimizer.zero_grad()
        pred_c, pred_t = self.model(*X)
        fake_label = torch.zeros_like(label)
        loss1 = self.loss_criterion(pred_c.view(-1), label.view(-1))
        loss2 = self.loss_criterion(pred_t.view(-1), fake_label.view(-1))
        loss = loss1 + loss2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=self._clip_grad_value)
        self.optimizer.step()
        return loss.item()

    def pretrain_causal(self, stage):
        print("start training causal !!!!!")
        iter = 0
        val_losses = [np.inf]
        saved_epoch = -1
        existing_epoch = -1
        try:
            existing_epoch, val_loss, optimizer, lr_scheduler = self.load_model(self.save_path, self._n_exp, stage)
            self.optimizer.load_state_dict(optimizer)
            self.lr_scheduler.load_state_dict(lr_scheduler)
            saved_epoch = existing_epoch
            val_losses.append(val_loss)
            print('Existing pretrained model')
        except:
            print('No existing pretrained model')

        for epoch in range(max(existing_epoch, -1)+1,self._max_epochs):
            self.model.train()
            train_losses = []
            if epoch - saved_epoch > self._patience:
                if self._wandb_flag:
                    wandb.run.summary[stage + " final_valid_loss"] = min(val_losses)
                    wandb.run.summary[stage + " saved_epoch"] = saved_epoch
                self.early_stop(epoch, min(val_losses))
                break

            start_time = time.time()
            for id, (*X, label) in tqdm(enumerate(self.data['train_loader']), total=len(self.data['train_loader'])):

                X = self._check_device(X)
                label = self._check_device(label)
                train_losses.append(self.train_batch_causal(X, label, iter))
                iter += 1
            end_time = time.time()

            val_loss = self.evaluate_causal()

            if self.lr_scheduler is None:
                new_lr = self._base_lr
            else:
                new_lr = self.lr_scheduler.get_last_lr()[0]

            message = (
                "Epoch [{}/{}] ({}) train_mse: {:.4f}, val_mse: {:.4f}, lr: {:.6f}, "
                "{:.1f}s".format(
                    epoch,
                    self._max_epochs,
                    iter,
                    np.mean(train_losses),
                    val_loss,
                    new_lr,
                    (end_time - start_time),
                )
            )
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self._logger.info(message)

            if val_loss < np.min(val_losses):
                temp_lr_scheduler = self.lr_scheduler
                model_file_name = self.save_model(
                    epoch, self._save_path, self._n_exp, val_loss, temp_lr_scheduler, stage = 'pretrain')
                self._logger.info(
                    'Val loss decrease from {:.4f} to {:.4f}, '
                    'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                val_losses.append(val_loss)
                saved_epoch = epoch

            if self._wandb_flag:
                wandb_key1 = stage + '_val_loss'
                wandb_key2 = stage + '_train_loss'
                wandb.log({wandb_key1: val_loss, stage + " epoch": epoch}, step=epoch)
                wandb.log({wandb_key2: np.mean(train_losses), stage + " epoch": epoch}, step=epoch)
                wandb.run.summary[stage + " final_valid_loss"] = min(val_losses)

    def evaluate_causal(self):
        labels = []
        preds = []
        with torch.no_grad():
            self.model.eval()
            for _, (*X, label) in tqdm(enumerate(self.data['val_loader']), total=len(self.data['val_loader'])):
                X = self._check_device(X)
                label = self._check_device(label)
                pred, label = self.test_batch_causal(X, label)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        loss = self.loss_criterion(preds, labels).item()
        return loss

    def test_batch_causal(self, X, label):
        pred, _ = self.model(*X)
        return pred, label

    def test_causal(self, stage, direct_finetune=False, mode='test'):
        if direct_finetune:
            stage = 'direct_finetune'
        _ = self.load_model(self.save_path, self._n_exp, stage=stage)

        labels = []
        preds = []

        with torch.no_grad():
            self.model.eval()
            for _, (*X, label) in tqdm(enumerate(self.data[mode + '_loader']), total=len(self.data[mode + '_loader'])):
                X = self._check_device(X)
                label = self._check_device(label)
                pred, label = self.test_batch_causal(X, label)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        preds, labels = preds.numpy().flatten(), labels.numpy().flatten()
        metrics = mc.compute_reg_metrics(preds, labels)
        log = '====Using evaluate function for test data -- Test MAE: {:.4f}, Test RMSE: {:.4f}, Test R2: {:.4f}, Test PCC: {:.4f}, ===='
        print(log.format(metrics[0], metrics[1], metrics[2], metrics[3]))

        mae = metrics[0]
        rmse = metrics[1]
        r2 = metrics[2]
        pcc = metrics[3]

        if self._wandb_flag:
            wandb.run.summary[stage + " test MAE"] = mae
            wandb.run.summary[stage + " test RMSE"] = rmse
            wandb.run.summary[stage + " test R2"] = r2
            wandb.run.summary[stage + " test PCC"] = pcc

        results = np.stack([mae, rmse, r2, pcc], axis=0)
        np.savetxt(os.path.join(self.save_path, 'results_{}.csv'.format(self._n_exp)), results, fmt='%.4f',
                   delimiter=',')

        return mae, rmse, r2, pcc
    
    def get_causal_genes(self, X, label):
        causal_score = self.model(*X, get_causal_genes=True)
        return causal_score

    def infer_causal(self, stage, cell_line_list, gene_list, direct_finetune=False, mode='infer'):
        if direct_finetune:
            stage = 'direct_finetune'
        _ = self.load_model(self.save_path, self._n_exp, stage=stage)

        labels = []
        preds = []

        with torch.no_grad():
            self.model.eval()
            for _, (*X, label) in tqdm(enumerate(self.data[mode + '_loader']), total=len(self.data[mode + '_loader'])):
                X = self._check_device(X)
                label = self._check_device(label)
                causal_score = self.get_causal_genes(X, label)
                causal_score_df = pd.DataFrame(causal_score.cpu().numpy(), index=cell_line_list, columns=gene_list)
                causal_score_df.to_csv(os.path.join(self.save_path, "causal_score.csv"), index_label="Row_Name")
        return 0

    def ig_analysis(self, stage, cline2id, gene_list, synergy_df_ig_celline, direct_finetune=False, mode='ig'):
        _ = self.load_model(self.save_path, self._n_exp, stage=stage)
        cell_line_list = self.map_values_to_keys(synergy_df_ig_celline, cline2id)
        print(cell_line_list)
        cell_line_list = self.add_duplicate_suffix_pd(cell_line_list)
        ig_results_attributions_list = []
        labels_items = []
        with torch.no_grad():
            self.model.eval()
            for _, (*X, label) in tqdm(enumerate(self.data[mode + '_loader']), total=len(self.data[mode + '_loader'])):
                X = self._check_device(X)
                label = self._check_device(label)
                labels_items.append(label.item())
                ig_results = self.model.integrated_gradients(*X, sample_index=0)
                ig_results_importance = ig_results['gene_importance']
                ig_results_attributions_list.append(ig_results_importance)


        ig_results_attributions_df = pd.DataFrame(np.array(ig_results_attributions_list), index=cell_line_list,
                                                  columns=gene_list)
        file_name = "{}-{}-{}-{}-{}-ig_results_attributions_df.csv".format(self.args.labels, self.args.torch_seed, self.args.n_exp, self.args.IG_drugA, self.args.IG_drugB)
        ig_results_attributions_df.to_csv(os.path.join(self.save_path, file_name), index_label="Row_Name")
        meta_data = pd.DataFrame({
            'celline': cell_line_list,
            'score': labels_items
                                  }
        )
        print(self.save_path)
        file_name = "{}-{}-{}-{}-{}-ig_results_attributions_meta_data.csv".format(self.args.labels, self.args.torch_seed, self.args.n_exp, self.args.IG_drugA, self.args.IG_drugB)
        meta_data.to_csv(os.path.join(self.save_path, file_name),
                                          index_label="Row_Name")

        print(1)

    def ig_ablation(self, stage, cline2id, gene_list, synergy_df_ig_celline, direct_finetune=False, mode='ig_test'):
        _ = self.load_model(self.save_path, self._n_exp, stage=stage)
        cell_line_list = self.map_values_to_keys(synergy_df_ig_celline, cline2id)
        print(cell_line_list)
        cell_line_list = self.add_duplicate_suffix_pd(cell_line_list)
        suffixes = ['e', 'm', 'c', 'd']
        ig_results_attributions_list = []
        gene_suffix_list =[f"{gene}_{suffix}" for suffix in suffixes for gene in gene_list]
        labels_items = []
        labels = []
        preds = []
        with torch.no_grad():
            self.model.eval()
            for _, (*X, label) in tqdm(enumerate(self.data[mode + '_loader']), total=len(self.data[mode + '_loader'])):
                X = self._check_device(X)
                label = self._check_device(label)
                ig_results = self.model.integrated_gradients(*X, sample_index=0)
                X = self.apply_attribution_mask(ig_results['attributions'], X, self.args.causal_ablation_ratio)
                pred, label = self.test_batch_causal(X, label)
                print(pred)
                labels.append(label.cpu())
                preds.append(pred.cpu())

            labels = torch.cat(labels, dim=0)
            preds = torch.cat(preds, dim=0)
            preds, labels = preds.numpy().flatten(), labels.numpy().flatten()
            metrics = mc.compute_reg_metrics(preds, labels)
            log = '====Using evaluate function for test data -- Test MAE: {:.4f}, Test RMSE: {:.4f}, Test R2: {:.4f}, Test PCC: {:.4f}, ===='
            print(log.format(metrics[0], metrics[1], metrics[2], metrics[3]))

            mae = metrics[0]
            rmse = metrics[1]
            r2 = metrics[2]
            pcc = metrics[3]

            if self._wandb_flag:
                wandb.run.summary[stage + " test MAE"] = mae
                wandb.run.summary[stage + " test RMSE"] = rmse
                wandb.run.summary[stage + " test R2"] = r2
                wandb.run.summary[stage + " test PCC"] = pcc
    def map_values_to_keys(self, series, value_to_key_dict):
        reverse_dict = defaultdict(list)
        for key, value in value_to_key_dict.items():
            reverse_dict[value].append(key)
        def get_keys(value):
            return reverse_dict.get(value, [])
        return series.apply(get_keys)

    def add_duplicate_suffix_pd(self, series):
        count_map = {}
        result_series = series.copy()
        def process(item_list):
            s = item_list[0] if len(item_list) > 0 else ""
            count_map[s] = count_map.get(s, 0) + 1
            if count_map[s] > 1:
                return f"{s}_{count_map[s] - 1}"
            else:
                return s

        return series.apply(process)

    def apply_attribution_mask(self, attributions, X_tensors, causal_ablation_ratio):
        n = attributions.shape[0]
        if abs(causal_ablation_ratio) > 0 and abs(causal_ablation_ratio) < 1:
            k = int(round(abs(causal_ablation_ratio) * n))
        for i in range(4):
            col_data = attributions[:, i]
            if causal_ablation_ratio > -1 and causal_ablation_ratio < 0:
                min_indices = np.argpartition(col_data, k)[:k]
                mask = np.zeros(n, dtype=bool)
                mask[min_indices] = True
            elif causal_ablation_ratio > 0 and causal_ablation_ratio < 1:
                max_indices = np.argpartition(col_data, -k)[-k:]
                mask = np.zeros(n, dtype=bool)
                mask[max_indices] = True
            else:
                mask = (col_data == 0)
            mask_tensor = torch.tensor(mask, device='cuda').reshape(1, -1)
            X_tensors[i + 2][mask_tensor] = 0
        return X_tensors

    def test_nc(self, stage, direct_finetune=False, mode='test'):
        if direct_finetune:
            stage = 'direct_finetune'
        _ = self.load_model(self.save_path, self._n_exp, stage=stage)

        labels = []
        preds = []

        with torch.no_grad():
            self.model.eval()
            for _, (*X, label) in tqdm(enumerate(self.data[mode + '_loader']), total=len(self.data[mode + '_loader'])):
                X = self._check_device(X)
                label = self._check_device(label)
                pred, label = self.test_batch(X, label)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        preds, labels = preds.numpy().flatten(), labels.numpy().flatten()
        metrics = mc.count_positive_products(preds, labels)
        acc = metrics/len(labels)

        if self._wandb_flag:
            wandb.run.summary[stage + " test ACC"] = acc

        return acc

    def test_nc_causal(self, stage, direct_finetune=False, mode='test'):
        if direct_finetune:
            stage = 'direct_finetune'
        _ = self.load_model(self.save_path, self._n_exp, stage=stage)

        labels = []
        preds = []

        with torch.no_grad():
            self.model.eval()
            for _, (*X, label) in tqdm(enumerate(self.data[mode + '_loader']), total=len(self.data[mode + '_loader'])):
                X = self._check_device(X)
                label = self._check_device(label)
                pred, label = self.test_batch_causal(X, label)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        preds, labels = preds.numpy().flatten(), labels.numpy().flatten()
        metrics = mc.count_positive_products(preds, labels)
        acc = metrics/len(labels)

        if self._wandb_flag:
            wandb.run.summary[stage + " test ACC"] = acc

        return acc

    def novel_causal(self, stage, synergy, cline2id, direct_finetune=False, mode='novel'):
        if direct_finetune:
            stage = 'direct_finetune'
        _ = self.load_model(self.save_path, self._n_exp, stage=stage)

        labels = []
        preds = []

        with torch.no_grad():
            self.model.eval()
            for _, (*X, label) in tqdm(enumerate(self.data[mode + '_loader']), total=len(self.data[mode + '_loader'])):
                X = self._check_device(X)
                label = self._check_device(label)
                pred, label = self.test_batch_causal(X, label)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        preds_np = np.array(preds)
        synergy['preds'] = preds_np.flatten()
        cell_line_list = self.map_values_to_keys(synergy['cell_line'], cline2id).tolist()
        flat_list = [item for sublist in cell_line_list for item in sublist]
        synergy['cell_line'] = flat_list
        file_name = "{}-{}-{}-novel.csv".format(self.args.labels, self.args.torch_seed,
                                                                            self.args.n_exp)
        synergy.to_csv(os.path.join(self.save_path, file_name), index_label="Row_Name")
        print('saved novel combos')

    def _get_cell_embedding(self, X, label):
        cell_embedding = self.model(*X)
        return cell_embedding

    def get_cell_embedding(self, stage, cell_line_list, gene_list, direct_finetune=False, mode='infer'):
        if direct_finetune:
            stage = 'direct_finetune'
        _ = self.load_model(self.save_path, self._n_exp, stage=stage)

        labels = []
        preds = []

        with torch.no_grad():
            self.model.eval()
            for _, (*X, label) in tqdm(enumerate(self.data[mode + '_loader']), total=len(self.data[mode + '_loader'])):
                X = self._check_device(X)
                label = self._check_device(label)
                cell_embedding = self._get_cell_embedding(X, label)
                cell_embedding_df = pd.DataFrame(cell_embedding.cpu().numpy(), index=cell_line_list)
                file_name = "{}-{}-{}-{}-cell_embedding.csv".format(self.args.model, self.args.labels, self.args.torch_seed,
                                                        self.args.n_exp)
                cell_embedding_df.to_csv(os.path.join(self.save_path, file_name), index_label="Row_Name")
            print('saved cell embedding')

    