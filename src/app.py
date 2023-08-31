from torch.utils.data import DataLoader

from entities.data.wound_roi import WoundROI
from models.embeddings.simple_class_embedder import ClassEmbedder
from models.trainer.diffuser import Diffuser

if __name__ == '__main__':
    wound_roi = WoundROI()

    # test = next(iter(wound_roi_loader))
    # print(test)
    diffuser = Diffuser(
        train_dataset=wound_roi,
        embedder=ClassEmbedder(wound_roi.class_tuple.__len__()),
        batch_size=1,
    )
    diffuser.fit()
