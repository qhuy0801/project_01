from torch.utils.data import DataLoader

from entities.data.wound_roi import WoundROI
from models.embeddings.simple_class_embedder import ClassEmbedder
from models import Diffuser_v1
from utils import revert_transform, save_pil_image, plot_chw
from utils.visualise_utils import plot_hwc

if __name__ == '__main__':
    wound_roi = WoundROI()
    # diffuser = Diffuser_v1(
    #     train_dataset=wound_roi,
    #     embedder=ClassEmbedder(wound_roi.__class_tuple.__len__()),
    #     batch_size=1,
    # )
    # diffuser.fit()


    test = wound_roi[1][0]
    # plot_chw(test)
    test2 = revert_transform(test)
    #
    # plot_hwc(test2)
    save_pil_image(test2, "test2", "")