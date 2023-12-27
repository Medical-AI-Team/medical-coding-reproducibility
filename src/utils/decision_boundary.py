import torch
from tqdm import tqdm
import json

def load_tensors_from_file(file_path):
    # Load the dictionary of tensors from the file
    loaded_tensor_dict = torch.load(file_path)
    return loaded_tensor_dict

def save_logits_and_targets(logits=None, targets=None, file_name="tensors.pt"):
  # Create a dictionary of tensors
  tensor_dict = {}
  if logits != None:
    tensor_dict["logits"] = logits
  if targets != None:
    tensor_dict["targets"] = targets
  # Save the dictionary of tensors to the specified file
  torch.save(tensor_dict, file_name)
  print(f"save_logits_and_targets Completed! in {file_name}")


def f1_score_db_tuning(logits, targets, average="micro", type="single"):
    print("Start f1_score_db_tuning")

    if average not in ["micro", "macro"]:
        raise ValueError("Average must be either 'micro' or 'macro'")
    dbs = torch.linspace(0, 1, 100)
    tp = torch.zeros((len(dbs), targets.shape[1]))
    fp = torch.zeros((len(dbs), targets.shape[1]))
    fn = torch.zeros((len(dbs), targets.shape[1]))
    for idx, db in enumerate(dbs):
        predictions = (logits > db).long()
        tp[idx] = torch.sum((predictions) * (targets), dim=0)
        fp[idx] = torch.sum(predictions * (1 - targets), dim=0)
        fn[idx] = torch.sum((1 - predictions) * targets, dim=0)
    if average == "micro":

        f1_scores = tp.sum(1) / (tp.sum(1) + 0.5 * (fp.sum(1) + fn.sum(1)) + 1e-10)
        # print(f"-tp:{tp.sum(1)[f1_scores.argmax()]}")
        # print(f"-fp:{fp.sum(1)[f1_scores.argmax()]}")
        # print(f"-fn:{fn.sum(1)[f1_scores.argmax()]}")
    else:
        f1_scores = torch.mean(tp / (tp + 0.5 * (fp + fn) + 1e-10), dim=1)
    if type == "single":
        best_f1 = f1_scores.max()
        best_db = dbs[f1_scores.argmax()]
        print(f"Best F1: {best_f1:.4f} at DB: {best_db:.4f}")
        return best_f1, best_db
    if type == "per_class":
        best_f1 = f1_scores.max(1)
        best_db = dbs[f1_scores.argmax(0)]
        print(f"Best F1: {best_f1} at DB: {best_db}")
        return best_f1, best_db


def threshold_tuning_per_class(logits, targets, average = "micro"):
    print("Start threshold_tuning_per_class")
    # Check if GPU is available, and use it if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logits = logits.to(device)
    targets = targets.to(device)

    if average not in ["micro", "macro"]:
      raise ValueError("Average must be either 'micro' or 'macro'")
    dbs = torch.linspace(0, 1, 100).to(device)  # Move dbs tensor to the GPU

    # Initialize lists to store per-class results
    best_f1_per_class = []
    best_db_per_class = []

    for class_idx in tqdm(range(targets.shape[1]),total=targets.shape[1]):
      tp = torch.zeros(len(dbs)).to(device)  # Move tp tensor to the GPU
      fp = torch.zeros(len(dbs)).to(device)  # Move fp tensor to the GPU
      fn = torch.zeros(len(dbs)).to(device)  # Move fn tensor to the GPU

      for idx, db in enumerate(dbs):
          predictions = (logits[:, class_idx] > db).long()
          tp[idx] = torch.sum((predictions) * (targets[:, class_idx]), dim=0)
          fp[idx] = torch.sum(predictions * (1 - targets[:, class_idx]), dim=0)
          fn[idx] = torch.sum((1 - predictions) * targets[:, class_idx], dim=0)

      if average == "micro":
          f1_scores = tp / (tp + 0.5 * (fp + fn) + 1e-10)
      else:
          f1_scores = torch.mean(tp / (tp + 0.5 * (fp + fn) + 1e-10))

      f1s_dbs_dict = {str(db):f1 for f1,db in zip(f1_scores.tolist(),dbs.tolist())}
      # print(f1s_dbs_dict)

      best_f1 = f1_scores.max()
      best_db = dbs[f1_scores.argmax()]

      # print(f"tp:{tp[f1_scores.argmax()]}")
      # print(f"fp:{fp[f1_scores.argmax()]}")
      # print(f"fn:{fn[f1_scores.argmax()]}")

      best_f1_per_class.append(best_f1.item())
      best_db_per_class.append(best_db.item())

    return best_f1_per_class, best_db_per_class


def compute_micro_f1_wrt_class_thresholds(logits, targets, best_db_per_class):
    # Check if GPU is available, and use it if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logits = logits.to(device)
    targets = targets.to(device)
    class_thresholds = torch.tensor(best_db_per_class).to(device)

    # Apply class-specific thresholds to get binary predictions
    binary_predictions = (logits > class_thresholds).int()

    # Calculate true positives, false positives, and false negatives
    # TP = torch.sum(binary_predictions * targets, dim=0)
    # FP = torch.sum(binary_predictions * (1 - targets), dim=0)
    # FN = torch.sum((1 - binary_predictions) * targets, dim=0)

    TP = torch.zeros(targets.shape[1]).to(device)  # Move tp tensor to the GPU
    FP = torch.zeros(targets.shape[1]).to(device)  # Move fp tensor to the GPU
    FN = torch.zeros(targets.shape[1]).to(device)  # Move fn tensor to the GPU

    for class_idx,db in zip(range(targets.shape[1]),best_db_per_class):
      binary_predictions = (logits[:, class_idx] > db).int()
      TP[class_idx] = torch.sum((binary_predictions) * (targets[:, class_idx]), dim=0)
      FP[class_idx] = torch.sum(binary_predictions * (1 - targets[:, class_idx]), dim=0)
      FN[class_idx] = torch.sum((1 - binary_predictions) * targets[:, class_idx], dim=0)

    # print(TP)
    # print(FP)
    # print(FN)

    # # Calculate micro-averaged precision and recall
    micro_precision = torch.sum(TP) / (torch.sum(TP + FP) + 1e-10)
    micro_recall = torch.sum(TP) / (torch.sum(TP + FN) + 1e-10)
    # print(f"Micro Precision (Based on the thresholds of the classes):{micro_precision}")
    # print(f"Micro Recall (Based on the thresholds of the classes):{micro_recall}")
    # Calculate micro F1 score
    # micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-10)
    micro_f1 = 2 * sum(TP) / (2 * sum(TP) + sum(FP) + sum(FN))
    return micro_f1
