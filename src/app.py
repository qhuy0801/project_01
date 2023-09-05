from entities.data.wound_roi import WoundROI
from models.embeddings.simple_class_embedder import ClassEmbedder
from models import Diffuser_v1

if __name__ == '__main__':
    wound_roi = WoundROI()
    diffuser = Diffuser_v1(
        train_dataset=wound_roi,
        embedder=ClassEmbedder(len(wound_roi.class_dict)),
        batch_size=1
    )
    diffuser.fit()

    # test = wound_roi[1]
    # plot_chw(test)
    # test2 = revert_transform(test)
    #
    # plot_hwc(test2)
    # save_pil_image(test2, "test2", "")

    # labels = torch.Tensor([1., 3., 0., 2.]).long()
    # s = diffuser.sample(labels)

    # a = 1
