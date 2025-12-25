from sdk.contracts import MetricAdapter


def dice_coef(pred, target, eps=1e-6):
    pred = pred.float()
    target = target.float()

    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


class MeanDice(MetricAdapter):
    name = "mean_dice"

    def reset(self):
        self.dices = []

    def update(self, predictions, targets):
        for pred, tgt in zip(predictions, targets):
            p_masks = pred.get("masks")
            t_masks = tgt.get("masks")

            if p_masks is None or t_masks is None:
                continue

            p_masks = p_masks > 0.5
            t_masks = t_masks > 0.5

            for pm, tm in zip(p_masks, t_masks):
                self.dices.append(dice_coef(pm[0], tm).item())

    def compute(self):
        return {self.name: sum(self.dices) / len(self.dices) if self.dices else 0.0}
