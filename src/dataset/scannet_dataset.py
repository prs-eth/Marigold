# Author: Bingxin Ke
# Last modified: 2024-02-08

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode


class ScanNetDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # ScanNet data parameter
            min_depth=1e-3,
            max_depth=10,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode ScanNet depth
        depth_decoded = depth_in / 1000.0
        return depth_decoded
