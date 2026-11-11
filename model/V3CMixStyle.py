"""V3 experiment adapter; no copy of the RGCN architecture or graph loader."""

from model.QuangV3 import Net
from model.v3_mixstyle import ClassConditionalGraphMixStyle


class CMixStyleNet(Net):
    """Same checkpoint parameter keys as V3, with a stateless augmentation.

    The trainer explicitly disables MixStyle on the clean supervision branch.
    At inference the module is identity and does not need labels or donors.
    """

    def __init__(self, *args, mixstyle_p=0.5, mixstyle_alpha=0.1,
                 mixstyle_strength=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixstyle = ClassConditionalGraphMixStyle(
            p=mixstyle_p, alpha=mixstyle_alpha, strength=mixstyle_strength,
        )

    def _mix_features(self, features, batch):
        return self.mixstyle(features, batch["bidx"], batch.get("y"), batch["B"])

    def forward(self, b, aug_mode=None, gen=None, *, mixstyle=True):
        transform = self._mix_features if mixstyle and self.training else None
        return super().forward(b, aug_mode=aug_mode, gen=gen, feature_transform=transform)
