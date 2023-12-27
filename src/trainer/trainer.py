#           "second_expr_mimic_50"
#           "second_expr_mimic_4_10"
#           "laat_expr_mimic_4_10"
#           "ICD_with_LayerNorm"
Drive_save_dir_name = "ICD_with_LayerNorm"
print(f"Model will be copied to drive: {Drive_save_dir_name}")

import gc
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
from typing import Optional

import pandas as pd
import torch
from omegaconf import OmegaConf
from rich.pretty import  pprint
from rich.progress import track
from torch.utils.data import DataLoader

from src.data.datatypes import Data, Lookups
from src.metrics import  MetricCollection
from src.models import BaseModel
from src.settings import ID_COLUMN, TARGET_COLUMN
from src.trainer.callbacks import BaseCallback
from src.utils.decision_boundary import f1_score_db_tuning, threshold_tuning_per_class, compute_micro_f1_wrt_class_thresholds, save_logits_and_targets, load_tensors_from_file
import subprocess
import json
from tqdm import tqdm
import datetime



class Trainer:
    def __init__(
        self,
        config: OmegaConf,
        data: Data,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        dataloaders: dict[str, DataLoader],
        metric_collections: dict[str, dict[str, MetricCollection]],
        callbacks: BaseCallback,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        lookups: Optional[Lookups] = None,
        accumulate_grad_batches: int = 1,
    ) -> None:
        self.config = config
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.callbacks = callbacks
        self.device = "cpu"
        self.metric_collections = metric_collections
        self.lr_scheduler = lr_scheduler
        self.lookups = lookups
        self.accumulate_grad_batches = accumulate_grad_batches
        pprint(f"Accumulating gradients over {self.accumulate_grad_batches} batch(es).")
        self.validate_on_training_data = config.trainer.validate_on_training_data
        self.print_metrics = config.trainer.print_metrics
        self.epochs = config.trainer.epochs
        self.epoch = 0
        self.use_amp = config.trainer.use_amp
        self.threshold_tuning = config.trainer.threshold_tuning
        self.gradient_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.experiment_path = Path(mkdtemp())
        self.current_val_results = None
        self.stop_training = False
        self.best_db = 0.5
        self.best_db_per_class = None
        self.on_initialisation_end()


        # Function to print all configs recursively
        def print_configs(config, prefix=""):
            for key, value in config.items():
                if OmegaConf.is_dict(value):
                    print_configs(value, prefix + key + ".")
                else:
                    print(f"{prefix}{key}: {value}")

        # Print all the configs
        print_configs(OmegaConf.to_container(config))
        
        
    def fit(self) -> None:
        """Train and validate the model."""
        try:
            try:
                path = "/content/best_model.pt"
                self.load_checkpoint(path)
            except:
                print(f"Can't Load {path}")
                
            self.on_fit_begin()

#            for _ in range(self.epoch, self.epochs):
#                if self.stop_training:
#                    break
            if self.epochs > 0:
              for i in range(self.epochs):
                self.on_epoch_begin()
                self.train_one_epoch(self.epoch)
                if self.validate_on_training_data:
                    self.train_val(self.epoch, "train_val")
                self.val(self.epoch, "val")
                self.val(self.epoch, "test", evaluating_best_model=True)
                self.on_epoch_end()
#                self.epoch += 1

            self.on_fit_end()
#            self.val(self.epoch, "val", evaluating_best_model=True)
            
            self.save_final_model()
            

            

        except KeyboardInterrupt:
            pprint("Training interrupted by user. Stopping training")
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.on_end()

    def save_model_and_copy_to_drive(self):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.gradient_scaler.state_dict(),
            "epoch": self.epoch,
            "db": self.best_db,
            "best_db_per_class": self.best_db_per_class
        }

        torch.save(checkpoint, "best_model.pt")
        pprint("Saved checkpoint to {}".format("best_model.pt"))
        # Execute the command
        try:               
            # Get the current time
            current_time = datetime.datetime.now()

            # Add 3 hours to the current time
            updated_time = current_time + datetime.timedelta(hours=3)

            # Format and print the updated time
            formatted_time = updated_time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"time now: {formatted_time}")

            source_path = "/content/best_model.pt"
            destination_path = f"/content/drive/MyDrive/{Drive_save_dir_name}/best_model_{formatted_time}.pt"


            # Construct the cp command
            command = ["cp", source_path, destination_path]

            subprocess.run(command, check=True)
            pprint(f"Model copied successfully to: /content/drive/MyDrive/{Drive_save_dir_name}/best_model_{formatted_time}.pt")
        except subprocess.CalledProcessError as e:
            pprint(f"Error copying the file: {e}")
        
#    def train_one_epoch(self, epoch: int) -> None:
#        """Train the model for one epoch.
#
#        Args:
#            epoch (int): The current epoch.
#        """
#        self.model.train()
#        self.on_train_begin()
#        num_batches = len(self.dataloaders["train"])
#        for batch_idx, batch in enumerate(
#            track(self.dataloaders["train"], description=f"Epoch: {epoch} | Training")
#        ):
#            batch = batch.to(self.device)
#            with torch.autocast(
#                device_type="cuda", dtype=torch.float16, enabled=self.use_amp
#            ):
#                output = self.model.training_step(batch)
#                loss = output["loss"] / self.accumulate_grad_batches
#            self.gradient_scaler.scale(loss).backward()
#            if ((batch_idx + 1) % self.accumulate_grad_batches == 0) or (
#                batch_idx + 1 == num_batches
#            ):
#                self.gradient_scaler.step(self.optimizer)
#                self.gradient_scaler.update()
#                if self.lr_scheduler is not None:
#                    if not isinstance(
#                        self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
#                    ):
#                        self.lr_scheduler.step()
#                self.optimizer.zero_grad()
#            self.update_metrics(output, "train")
#            # break
#        self.on_train_end(epoch)

    def train_one_epoch(self, epoch: int) -> None:
        """Train the model for one epoch.

        Args:
            epoch (int): The current epoch.
        """
        self.model.train()
        self.on_train_begin()
        num_batches = len(self.dataloaders["train"])
        
        split_name = "train"
        tensors_dict = load_tensors_from_file(f"{split_name}_tensor.pt")
        targets_tensor = tensors_dict["targets"].to(self.device)
        targets_batches = torch.chunk(targets_tensor, num_batches, dim=0)

        for batch_idx, batch in enumerate(
            track(self.dataloaders["train"], description=f"Epoch: {epoch} | Training")
        ):
            batch.targets = targets_batches[batch_idx]
            batch = batch.to(self.device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.use_amp
            ):
                output = self.model.training_step(batch)
                loss = output["loss"] / self.accumulate_grad_batches
            self.gradient_scaler.scale(loss).backward()
            if ((batch_idx + 1) % self.accumulate_grad_batches == 0) or (
                batch_idx + 1 == num_batches
            ):
                self.gradient_scaler.step(self.optimizer)
                self.gradient_scaler.update()
                if self.lr_scheduler is not None:
                    if not isinstance(
                        self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.lr_scheduler.step()
                self.optimizer.zero_grad()
            self.update_metrics(output, "train")
            # break
        self.on_train_end(epoch)
        
#    def train_val(self, epoch, split_name: str = "train_val") -> None:
#        """Validate on the training data. This is useful for testing for overfitting. Due to memory constraints, we donøt save the outputs.
#
#        Args:
#            epoch (_type_): _description_
#            split_name (str, optional): _description_. Defaults to "train_val".
#        """
#        self.model.eval()
#        self.on_val_begin()
#        with torch.no_grad():
#            for batch in track(
#                self.dataloaders[split_name],
#                description=f"Epoch: {epoch} | Validating on training data",
#            ):
#                with torch.autocast(
#                    device_type="cuda", dtype=torch.float16, enabled=self.use_amp
#                ):
#                    output = self.model.validation_step(batch.to(self.device))
#                self.update_metrics(output, split_name)
#            self.on_val_end(split_name, epoch)

    def train_val(self, epoch, split_name: str = "train_val") -> None:
        """Validate on the training data. This is useful for testing for overfitting. Due to memory constraints, we donøt save the outputs.

        Args:
            epoch (_type_): _description_
            split_name (str, optional): _description_. Defaults to "train_val".
        """
        self.model.eval()
        self.on_val_begin()
        num_batches = len(self.dataloaders["train"])
        split_name = "train"
        tensors_dict = load_tensors_from_file(f"{split_name}_tensor.pt")
        targets_tensor = tensors_dict["targets"].to(self.device)
        targets_batches = torch.chunk(targets_tensor, num_batches, dim=0)
        batch_idx = 0
        with torch.no_grad():
            for batch in track(
                self.dataloaders[split_name],
                description=f"Epoch: {epoch} | Validating on training data",
            ):
                batch.targets = targets_batches[batch_idx]
                batch = batch.to(self.device)
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=self.use_amp
                ):
                    output = self.model.validation_step(batch.to(self.device))
                self.update_metrics(output, split_name)
                batch_idx+=1
                # if batch_idx > 1 : break
            self.on_val_end(split_name, epoch)


#    def val(
#        self, epoch, split_name: str = "val", evaluating_best_model: bool = False
#    ) -> None:
#        self.model.eval()
#        self.on_val_begin()
#        logits = []
#        targets = []
#        logits_cpu = []
#        targets_cpu = []
#        ids = []
#        with torch.no_grad():
#            for idx, batch in enumerate(
#                track(
#                    self.dataloaders[split_name],
#                    description=f"Epoch: {epoch} | Validating on {split_name}",
#                )
#            ):
#                with torch.autocast(
#                    device_type="cuda", dtype=torch.float16, enabled=self.use_amp
#                ):
#                    output = self.model.validation_step(batch.to(self.device))
#                self.update_metrics(output, split_name)
#                logits.append(output["logits"])
#                targets.append(output["targets"])
#                ids.append(batch.ids)
#                if idx % 1000 == 0:
#                    # move to cpu to save gpu memory
#                    logits_cpu.append(torch.cat(logits, dim=0).cpu())
#                    targets_cpu.append(torch.cat(targets, dim=0).cpu())
#                    logits = []
#                    targets = []
#                # if idx > 1 : break
#            logits_cpu.append(torch.cat(logits, dim=0).cpu())
#            targets_cpu.append(torch.cat(targets, dim=0).cpu())
#
#            logits = torch.cat(logits_cpu, dim=0)
#            targets = torch.cat(targets_cpu, dim=0)
#            ids = torch.cat(ids, dim=0)
#        self.on_val_end(split_name, epoch, logits, targets, ids, evaluating_best_model)

    def val(
        self, epoch, split_name: str = "val", evaluating_best_model: bool = False
    ) -> None:
        self.model.eval()
        self.on_val_begin()
        logits = []
        targets = []
        logits_cpu = []
        targets_cpu = []
        ids = []
        tensors_dict = load_tensors_from_file(f"{split_name}_tensor.pt")
        targets_tensor = tensors_dict["targets"]
        num_batches = len(self.dataloaders[split_name])
        targets_batches = torch.chunk(targets_tensor, num_batches, dim=0)
        with torch.no_grad():
            for idx, batch in enumerate(
                track(
                    self.dataloaders[split_name],
                    description=f"Epoch: {epoch} | Validating on {split_name}",
                )
            ):
                batch.targets = targets_batches[idx]
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=self.use_amp
                ):
                    output = self.model.validation_step(batch.to(self.device))
                self.update_metrics(output, split_name)
                logits.append(output["logits"])
                targets.append(output["targets"])
                ids.append(batch.ids)
                if idx % 1000 == 0:
                    # move to cpu to save gpu memory
                    logits_cpu.append(torch.cat(logits, dim=0).cpu())
                    targets_cpu.append(torch.cat(targets, dim=0).cpu())
                    logits = []
                    targets = []
                # if idx > 1 : break
            logits_cpu.append(torch.cat(logits, dim=0).cpu())
            targets_cpu.append(torch.cat(targets, dim=0).cpu())

            logits = torch.cat(logits_cpu, dim=0)
            targets = torch.cat(targets_cpu, dim=0)
            ids = torch.cat(ids, dim=0)
        self.on_val_end(split_name, epoch, logits, targets, ids, evaluating_best_model)


    def update_metrics(self, outputs: dict[str, torch.Tensor], split_name: str) -> None:
#        for target_name in self.metric_collections[split_name].keys():
#            self.metric_collections[split_name][target_name].update(outputs)
        pass

    def calculate_metrics(
        self,
        split_name: str,
        logits: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        evaluating_best_model: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]]:
        results_dict = defaultdict(dict)
        if split_name == "val":
            for target_name in self.metric_collections[split_name].keys():
                results_dict[split_name][target_name] = self.metric_collections[
                    split_name
                ][target_name].compute()
        else:
            try:
                for target_name in self.metric_collections[split_name].keys():
                    results_dict[split_name][target_name] = self.metric_collections[
                        split_name
                    ][target_name].compute(logits, targets)
            except:
                pass
            try:
                micro_f1_score_thresh_per_class = compute_micro_f1_wrt_class_thresholds(logits, targets, self.best_db_per_class)
                print("Micro F1 Score (Based on the thresholds of the classes):", micro_f1_score_thresh_per_class.item())
            except:
                print("Can not get Micro F1 Score (Based on the thresholds of the classes)")
            try:  
                micro_f1_score_global_thresh = compute_micro_f1_wrt_class_thresholds(logits, targets, [self.best_db for i in range(targets.shape[1])])
                print("Micro F1 Score (Based on the Global threshold):", micro_f1_score_global_thresh.item())
            except:
                print("Can not get Micro F1 Score (Based on the Global threshold)")

        if self.threshold_tuning and split_name == "val":

            clipping_indeces = [i for i in range(logits.shape[1])]
            logits_clipped = logits[:, [i for i in range(logits.shape[1]) if i in clipping_indeces]]
            targets_clipped = targets[:, [i for i in range(targets.shape[1]) if i in clipping_indeces]]

            best_f1, best_db = f1_score_db_tuning(logits_clipped, targets_clipped)

            best_f1_per_class, best_db_per_class = threshold_tuning_per_class(logits_clipped, targets_clipped)

            dictinoary = {}
            changed_indeces = []
            micro_f1_scores = []
            for i in tqdm(range(0, 101)):  
                value = i / 100.0  # A float between 0.0 and 1.0
                f1_threshold = value
                for idx,db,f1 in zip(sorted(clipping_indeces),best_db_per_class,best_f1_per_class):
                  if f1 >= f1_threshold:
                    dictinoary[idx] = db
                  if f1 < f1_threshold:
                    dictinoary[idx] = best_db.item()
                    changed_indeces.append(idx)


                micro_f1_score = compute_micro_f1_wrt_class_thresholds(logits_clipped, targets_clipped, list(dictinoary.values()))

                micro_f1_scores.append((f1_threshold,micro_f1_score.item(),list(dictinoary.values())))

            # Sort the list based on the second element (index 1) of each tuple
            micro_f1_scores = sorted(micro_f1_scores, key=lambda x: x[1],reverse=True)
            best_db_per_class_list = micro_f1_scores[0][2]
            best_f1 = micro_f1_scores[0][1]
            print("Micro F1 Score (Based on the thresholds of the classes):", best_f1)

            best_result = best_f1
            self.best_db_per_class = best_db_per_class_list
            # try:
            #     micro_f1_score = compute_micro_f1_wrt_class_thresholds(logits, targets, self.best_db_per_class)
            #     print("Micro F1 Score (Based on the thresholds of the classes):", micro_f1_score.item())
            # except:
            #     print("Can not get Micro F1 Score (Based on the thresholds of the classes)")
            results_dict[split_name]["all"] |= {"f1_micro_tuned": best_result}
            if evaluating_best_model:
                pprint(f"Best threshold: {best_db}")
                pprint(f"Best result: {best_result}")
                for target_name in self.metric_collections["test"]:
                    self.metric_collections["test"][target_name].set_threshold(best_db)
            self.best_db = best_db.item()
        return results_dict

    def reset_metric(self, split_name: str) -> None:
        for target_name in self.metric_collections[split_name].keys():
            self.metric_collections[split_name][target_name].reset_metrics()

    def reset_metrics(self) -> None:
        for split_name in self.metric_collections.keys():
            for target_name in self.metric_collections[split_name].keys():
                self.metric_collections[split_name][target_name].reset_metrics()

    def on_initialisation_end(self) -> None:
        for callback in self.callbacks:
            callback.on_initialisation_end(self)

    def on_fit_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_begin(self)

    def on_fit_end(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_end(self)

    def on_train_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self, epoch: int) -> None:
        results_dict = self.calculate_metrics(split_name="train")
        results_dict["lr"] = self.optimizer.param_groups[0]["lr"]
        self.log_dict(results_dict, epoch)
        for callback in self.callbacks:
            callback.on_train_end()
        


    def on_val_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_val_begin()

    def on_val_end(
        self,
        split_name: str,
        epoch: int,
        logits: torch.Tensor = None,
        targets: torch.Tensor = None,
        ids: torch.Tensor = None,
        evaluating_best_model: bool = False,
    ) -> None:
        results_dict = self.calculate_metrics(
            split_name=split_name,
            logits=logits,
            targets=targets,
            evaluating_best_model=evaluating_best_model,
        )
        self.current_val_results = results_dict
        self.log_dict(results_dict, epoch)
        for callback in self.callbacks:
            callback.on_val_end()

        if evaluating_best_model:
#            self.save_predictions(
#                split_name=split_name, logits=logits, targets=targets, ids=ids
#            )
            pass

    def save_predictions(
        self,
        split_name: str = "test",
        logits: torch.Tensor = None,
        targets: torch.Tensor = None,
        ids: torch.Tensor = None,
    ):
        from time import time

        tic = time()
        pprint("Saving predictions")
        label_transform = self.dataloaders[split_name].dataset.label_transform
        code_names = label_transform.get_classes()
        logits = logits.numpy()
        pprint("Building dataframe")
        df = pd.DataFrame(logits, columns=code_names)
        pprint("Adding targets")
        df[TARGET_COLUMN] = list(map(label_transform.inverse_transform, targets))
        pprint("Adding ids")
        df[ID_COLUMN] = ids.numpy()
        pprint("Saving dataframe")
        df.to_feather(self.experiment_path / f"predictions_{split_name}.feather")
        pprint("Saved predictions in {:.2f} seconds".format(time() - tic))

    def on_epoch_begin(self) -> None:
        self.reset_metrics()
        for callback in self.callbacks:
            callback.on_epoch_begin(self)

    def on_epoch_end(self) -> None:
        if self.lr_scheduler is not None:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(
                    self.current_val_results["val"]["all"]["f1_micro"]
                )   

        for callback in self.callbacks:
            callback.on_epoch_end(self)
        self.save_model_and_copy_to_drive()

    def on_batch_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self) -> None:
        for callback in self.callbacks:
            callback.on_batch_end()

    def log_dict(
        self, nested_dict: dict[str, dict[str, torch.Tensor]], epoch: int
    ) -> None:
        if self.print_metrics:
            self.print(nested_dict)
        for callback in self.callbacks:
            callback.log_dict(nested_dict, epoch)

    def on_end(self) -> None:
        for callback in self.callbacks:
            callback.on_end()

    def print(self, nested_dict: dict[str, dict[str, torch.Tensor]]) -> None:
        for split_name in nested_dict.keys():
            pprint(nested_dict[split_name])

    def to(self, device: str) -> "Trainer":
        self.model.to(device)
        for split_name in self.metric_collections.keys():
            for target_name in self.metric_collections[split_name].keys():
                self.metric_collections[split_name][target_name].to(device)
        self.device = device
        return self

    def save_checkpoint(self, file_name: str) -> None:
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.gradient_scaler.state_dict(),
            "epoch": self.epoch,
            "db": self.best_db,
            "best_db_per_class": self.best_db_per_class
        }
        torch.save(checkpoint, self.experiment_path / file_name)
        pprint("Saved checkpoint to {}".format(self.experiment_path / file_name))
        
        torch.save(checkpoint, "best_model.pt")
        pprint("Saved checkpoint to {}".format("best_model.pt"))
        

        
    def load_checkpoint(self, file_path: str) -> None:
        file_path = "best_model.pt"
        checkpoint = torch.load(file_path)
#        self.model.load_state_dict(checkpoint["model"])

        self.model.load_state_dict(checkpoint["model"], strict=False)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.gradient_scaler.load_state_dict(checkpoint["scaler"])
        self.epoch = checkpoint["epoch"]
        self.best_db = checkpoint["db"]
        try:
            self.best_db_per_class = checkpoint["best_db_per_class"]
        except:
            print("best_db_per_class Not Found.")
        pprint("Loaded checkpoint from {}".format(file_path))


    def save_transforms(self) -> None:
        """Save text tokenizer and label encoder"""
        self.dataloaders["train"].dataset.text_transform.save(self.experiment_path)
        self.dataloaders["train"].dataset.label_transform.save(self.experiment_path)

    def save_final_model(self) -> None:
        self.save_checkpoint("final_model.pt")
        self.save_transforms()
        OmegaConf.save(self.config, self.experiment_path / "config.yaml")
