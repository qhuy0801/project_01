from torch.utils.data import DataLoader

from entities.data.wound_roi import WoundROI
from models.embeddings.simple_class_embedder import ClassEmbedder
from models import Diffuser_v1

if __name__ == '__main__':
    wound_roi = WoundROI()
    diffuser = Diffuser_v1(
        train_dataset=wound_roi,
        embedder=ClassEmbedder(wound_roi.class_tuple.__len__()),
        batch_size=1,
    )
    diffuser.fit()
