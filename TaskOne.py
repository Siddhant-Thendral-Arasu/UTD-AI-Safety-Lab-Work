from nuscenes import NuScenes
import torch
import matplotlib.pyplot as plt
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer
import numpy as np
from nuscenes.prediction import PredictHelper
from tqdm import tqdm
from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP
from nuscenes.prediction.models.covernet import CoverNet
from nuscenes.prediction.models.covernet import ConstantLatticeLoss
from nuscenes.prediction.models.covernet import mean_pointwise_l2_distance
import pickle
import torch.nn
import torch.optim as optim
import math
import os


DATAROOT = '/data/Datasets/nuscenes/mini'
nusc = NuScenes('v1.0-mini', dataroot=DATAROOT)


from nuscenes.eval.prediction.splits import get_prediction_challenge_split
mini_train = get_prediction_challenge_split("mini_train", dataroot=DATAROOT)


helper = PredictHelper(nusc)
static_layer_rasterizer = StaticLayerRasterizer(helper)
agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=1)
input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())


class NuScenesDataset(torch.utils.data.Dataset):
    def __init__(self, data_split, helper, rasterizer):
        self.data_split = data_split
        self.helper = helper
        self.rasterizer = rasterizer


    def __len__(self):
        return len(self.data_split)


    def __getitem__(self, idx):
        instance_token, sample_token = self.data_split[idx].split("_")
        img = self.rasterizer.make_input_representation(instance_token, sample_token)
        velocity = self.helper.get_velocity_for_agent(instance_token, sample_token)
        acceleration = self.helper.get_acceleration_for_agent(instance_token, sample_token)
        heading_change_rate = self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)
        if math.isnan(velocity):
            velocity = 0.0
        if math.isnan(acceleration):
            acceleration = 0.0
        if math.isnan(heading_change_rate):
            heading_change_rate = 0.0


        agent_state_vector = torch.Tensor([velocity, acceleration, heading_change_rate])


        ground_truth = self.helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)


        return torch.tensor(img).permute(2, 0, 1), agent_state_vector, torch.tensor(ground_truth)
   
def decode_predicted_trajectories(logits, lattice, top_k=64):
    """
    Decode predicted trajectories from logits and lattice.


    :param logits: Tensor of shape (batch_size, num_modes) - model output logits
    :param lattice: Tensor of shape (num_modes, num_timesteps, 2) - trajectory lattice
    :param top_k: Number of top trajectories to decode (default is 1)
    :return: Predicted trajectories of shape (batch_size, top_k, num_timesteps, 2)
    """
    top_indices = torch.topk(logits, k=top_k, dim=1).indices  # Shape: (batch_size, top_k)
    predicted_trajectories = lattice[top_indices]  # Shape: (batch_size, top_k, num_timesteps, 2)
    return predicted_trajectories
def compute_mink_ade(logits, truth, trajectory_modes, n):
    topN = torch.topk(logits, n, dim=1).indices
    topN_trajs = trajectory_modes[topN]
    truth_expanded = truth.unsqueeze(1)
    diff = topN_trajs - truth_expanded
    point_distances = torch.norm(diff, dim=-1)
    ade_per_mode = torch.mean(point_distances, dim=-1)
    min_ade = torch.min(ade_per_mode, dim=1).values

    return min_ade
def compute_fde(logits, ground_truth, lattice):
    """
    Compute Final Displacement Error (FDE).
   
    :param predicted_trajectories: Tensor of shape (batch_size, num_modes, num_timesteps, 2)
    :param ground_truth: Tensor of shape (batch_size, num_timesteps, 2)
    :return: Average FDE over the batch
    """
    fde_list = []

    top1_indices = torch.argmax(logits, dim=1)  # (B,)

    for i in range(logits.size(0)):
        pred = lattice[top1_indices[i]]  # (T, 2)
        gt = ground_truth[i]  # (T, 2)

        fde = torch.norm(pred[-1] - gt[-1]).item()
        fde_list.append(fde)

    return sum(fde_list)/len(fde_list)
def compute_HitRatek(logits, ground_truth, k, threshold, lattice):
    """
    Compute HitRatek.

    Args:
        predicted_trajectories: Tensor of shape (num_trajectories, T, 2).
        ground_truth: Tensor of shape (T, 2).
        k: Number of top trajectories to consider.
        threshold: Distance threshold for a "hit."

    Returns:
        HitRatek: Fraction of predictions that satisfy the threshold.
    """
    B = logits.size(0)
    hit_count = 0

    topk = torch.topk(logits, k=k, dim=1).indices  # (B, k)

    for i in range(B):
        gt_final = ground_truth[i][-1]  # (2,)
        hit = False
        for j in range(k):
            pred_idx = topk[i, j]
            pred_final = lattice[pred_idx][-1]  # (2,)
            dist = torch.norm(pred_final - gt_final)
            if dist <= threshold:
                hit = True
                break
        if hit:
            hit_count += 1

    return hit_count / B


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


backbone = ResNetBackbone('resnet50')
covernet = CoverNet(backbone, num_modes=64).to(device)
                   
lattice_path = 'epsilon_8.pkl'
with open(lattice_path, "rb") as file:
    lattice = pickle.load(file)
lattice = torch.Tensor(lattice).to(device)
print(lattice.shape)


from torch.utils.data import DataLoader
train_dataset = NuScenesDataset(mini_train, helper, input_representation)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)


epochs = 10
optimizer = optim.Adam(covernet.parameters(), lr=1e-4)
loss_function = ConstantLatticeLoss(lattice)
for epoch in range(epochs):
    covernet.train()


    total_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
    for batch_idx, (images, agent_states, ground_truth_trajectories) in progress_bar:
        images, agent_states = images.float().to(device), agent_states.float().to(device)
        ground_truth_trajectories = ground_truth_trajectories.float().to(device)
       
        optimizer.zero_grad()


        logits = covernet(images, agent_states)


        loss = loss_function(logits, ground_truth_trajectories)


        loss.backward()
        optimizer.step()


        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())


print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {total_loss / len(train_loader):.4f}")


mini_val = get_prediction_challenge_split("mini_val", dataroot=DATAROOT)
val_dataset = NuScenesDataset(mini_val, helper, input_representation)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)


covernet.eval()
t_ade1 = 0
t_ade5 = 0
t_ade10 = 0
t_ade15 = 0
t_fde = 0
t_hitrate = 0
run = 1
with torch.no_grad():  # No gradient computation needed
        for images, agent_states, ground_truths in val_loader:
            images = images.float().to(device)
            agent_states = agent_states.float().to(device)
            ground_truths = ground_truths.float().to(device)
            print(agent_states)
            logits = covernet(images, agent_states)

            print("LOGITS")
            print(logits)
            print("REALITY")
            print(ground_truths)

            print(logits.dim())
            print(ground_truths.dim())
            ade1 = compute_mink_ade(logits, ground_truths, lattice, 1)
            ade5 = compute_mink_ade(logits, ground_truths, lattice, 5)
            ade10 = compute_mink_ade(logits, ground_truths, lattice, 10)
            ade15 = compute_mink_ade(logits, ground_truths, lattice, 15)
            fde = compute_fde(logits, ground_truths, lattice)
            hitrate = compute_HitRatek(logits, ground_truths, 5, 2.0, lattice)
            print(f"Ade1: {torch.sum(ade1)/16}, Ade5: {torch.sum(ade5)/16}, Ade10: {torch.sum(ade10)/16}, Ade15: {torch.sum(ade15)/16}, Fde: {fde}, HitRate: {hitrate}")
            t_ade1 += torch.sum(ade1)/16
            t_ade5 += torch.sum(ade5)/16
            t_ade10 += torch.sum(ade10)/16
            t_ade15 += torch.sum(ade15)/16
            t_fde += fde
            t_hitrate += hitrate
            run += 1
print(f"Ade1: {t_ade1/run}, Ade5: {t_ade5/run}, Ade10: {t_ade10/run}, Ade15: {t_ade15/run}, Fde: {t_fde/run}, HitRate: {t_hitrate/run}")
print("MISSION COMPLETE, GET TO THE BASE")