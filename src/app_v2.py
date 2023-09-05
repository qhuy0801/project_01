from entities.data.wound_roi import WoundROI
from models.trainer.diffuser_v2 import Diffuser_v2

if __name__ == '__main__':
    wound_roi = WoundROI()
    diffuser = Diffuser_v2(
        train_dataset=wound_roi,
        batch_size=1,
    )

    diffuser.sample(epoch=4, num_samples=2)
    # diffuser.fit()
