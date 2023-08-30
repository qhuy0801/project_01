from torch.utils.data import DataLoader

from entities.data.wound_roi import WoundROI
from models.diffuser import Diffuser_v1

if __name__ == '__main__':
    wound_roi_data = WoundROI()
    # wound_roi_loader = DataLoader(wound_roi_data, batch_size=10)

    # test = next(iter(wound_roi_loader))
    # print(test)
    # diffuser = Diffuser(
    #     train_data=wound_roi_loader,
    #     embedder=ClassEmbedder(wound_roi_data.class_tuple.__len__())
    # )
    # diffuser.fit()

    diffusers = Diffuser_v1(
        train_data=wound_roi_data
    )
    diffusers.fit()

