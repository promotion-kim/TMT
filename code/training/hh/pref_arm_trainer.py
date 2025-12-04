from trl import DPOTrainer
import torch, random
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch.nn.functional as F
import torch.nn as nn
from transformers import PreTrainedModel
from scipy.stats import dirichlet

class PrefARMTrainer(DPOTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        training_args = kwargs["args"]
        self.pref_sample_p = training_args.pref_sample_p
        self.beta_obj = training_args.beta_obj
        self.obj_key = training_args.obj_key
        self.gamma = training_args.gamma # target_reward_margin
        self.length_normalization = training_args.length_normalization
        self.num_step = 0

        if self.length_normalization:
            print('\nUsing length normalization. This is not default for training Autoregressive RM and should only be used for testing purposes!\n')
        if self.gamma != 0 or self.length_normalization:
            print(f'\nARM Trainer: gamma = {self.gamma}, length_normalization = {self.length_normalization}\n')

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        batch = super().tokenize_row(feature, model)
        batch['labels'] = feature['labels']
        return batch

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        is_vision_model: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        concatenated_batch = DPOTrainer.concatenated_inputs(batch, is_encoder_decoder, is_vision_model, label_pad_token_id, padding_value, device)
        concatenated_batch['labels'] = batch['labels']
        return concatenated_batch

    def arm_loss(
        self,
        beta: float,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the arm loss for a batch of policy model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the arm loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        gamma_logratios = self.gamma / beta 
        pi_logratios = pi_logratios.to(self.accelerator.device)
        logits = pi_logratios - gamma_logratios

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )

        chosen_rewards = beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = beta * policy_rejected_logps.to(self.accelerator.device).detach()

        return losses, chosen_rewards, rejected_rewards
    
    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps, valid_length = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        if self.length_normalization:
            all_logps = all_logps / valid_length

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the arm loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        self.num_step += 1

        preference = torch.tensor(dirichlet.rvs([self.pref_sample_p]*len(self.obj_key)))[0]

        for n, p in model.named_parameters():
            if 'pref_vec' in n:
                p.data = preference.to(p.device)
                p.requires_grad = False

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        loss_list = []
        bs = policy_chosen_logps.size()[0]
        concat_logps = torch.cat([policy_chosen_logps.unsqueeze(1), policy_rejected_logps.unsqueeze(1)], dim=1) # [bs, 2]
        concat_logits = torch.cat([policy_chosen_logits.unsqueeze(1), policy_rejected_logits.unsqueeze(1)], dim=1) # [bs, 2]
        obj_len = bs // len(self.obj_key)

        for obj_idx, obj in enumerate(self.obj_key):
            chosen_idx = [batch['labels'][i][obj] for i in range(bs)]
            rejected_idx = [1 - batch['labels'][i][obj] for i in range(bs)]

            obj_policy_rejected_logps = concat_logps[range(bs), rejected_idx]
            obj_policy_chosen_logps = concat_logps[range(bs), chosen_idx]
            obj_policy_rejected_logits = concat_logits[range(bs), rejected_idx]
            obj_policy_chosen_logits = concat_logits[range(bs), chosen_idx]

            losses, chosen_rewards, rejected_rewards = self.arm_loss(
                self.beta_obj[obj_idx],
                obj_policy_chosen_logps,
                obj_policy_rejected_logps
            )
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            #!
            #rewards_collection.append(chosen_rewards)

            loss_list.append(losses.mean())

            prefix = "eval_" if train_eval == "eval" else ""

            metrics[f"{prefix}loss_{obj}"] = losses.mean().detach().cpu()

            # metrics[f"{prefix}rewards_{obj}/chosen"] = chosen_rewards.mean().cpu()
            # metrics[f"{prefix}rewards_{obj}/rejected"] = rejected_rewards.mean().cpu()
            metrics[f"{prefix}rewards_{obj}/accuracies"] = reward_accuracies.mean().cpu()
            # metrics[f"{prefix}rewards_{obj}/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
            # metrics[f"{prefix}logps_{obj}/rejected"] = obj_policy_rejected_logps.detach().mean().cpu()
            # metrics[f"{prefix}logps_{obj}/chosen"] = obj_policy_chosen_logps.detach().mean().cpu()
            # metrics[f"{prefix}logits_{obj}/rejected"] = obj_policy_rejected_logits.detach().mean().cpu()
            # metrics[f"{prefix}logits_{obj}/chosen"] = obj_policy_chosen_logits.detach().mean().cpu()

        #total_loss = sum([each_loss*each_pref for each_loss, each_pref in zip(loss_list, preference)])
        #!
        #linear
        task_loss = sum([each_loss*each_pref for each_loss, each_pref in zip(loss_list, preference)])
        #stch
        '''mu = getattr(self, "stch_mu", 0.1)
        utopian_point = -0.1
        stacked_losses = torch.stack(loss_list)
        preference = preference.to(dtype=stacked_losses.dtype, device=stacked_losses.device)
        weighted_gaps = preference * (stacked_losses - utopian_point)
        logsumexp_arg = weighted_gaps / mu
        stch_loss = mu * torch.logsumexp(logsumexp_arg, dim=0)
        task_loss = stch_loss'''
        #tch
        '''utopian_point = -0.1
        weighted_distances = []
        for loss, w in zip(loss_list, preference):
            dist = w * (loss-utopian_point)
            weighted_distances.append(dist)
        stacked_distances = torch.stack(weighted_distances)
        tchebycheff_loss = torch.max(stacked_distances)
        task_loss = tchebycheff_loss'''
        #rewards = []
        #safe_pref = torch.tensor([1.0, 0.0], device=self.accelerator.device)
        #self._inject_preference(model, safe_pref)
        #(safe_chosen_logps, _, _, _) = self.concatenated_forward(model, batch)
        #rewards.append(safe_chosen_logps)
        #help_pref = torch.tensor([0.0, 1.0], device=self.accelerator.device)
        #self._inject_preference(model, help_pref)
        #(help_chosen_logps, _, _, _) = self.concatenated_forward(model, batch)
        #rewards.append(help_chosen_logps)
        #corr = compute_correlation(rewards[0], rewards[1])
        #decorr_loss = corr ** 2
        #lambda_decorr = 0.1
        #total_loss = task_loss + lambda_decorr * decorr_loss
        #metrics["train_decorr_loss"] = decorr_loss.item()
        #metrics["train_total_loss"] = total_loss.item()
        return task_loss, metrics