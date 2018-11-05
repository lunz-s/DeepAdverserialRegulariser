from ClassFiles.Framework import AdversarialRegulariser
from ClassFiles.networks import ConvNetClassifier
from ClassFiles.data_pips import LUNA
from ClassFiles.forward_models import CT


DATA_PATH = '/local/scratch/public/sl767/LUNA/'
SAVES_PATH = '/local/scratch/public/sl767/DeepAdversarialRegulariser/'


class Experiment1(AdversarialRegulariser):
    experiment_name = 'Low_Noise'
    noise_level = 0.02

    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = .5

    learning_rate = 0.0001
    step_size = .5
    total_steps_default = 30
    starting_point = 'Mini'

    def get_network(self, size, colors):
        return ConvNetClassifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(7, 1, y, fbp, 0)

    def get_Data_pip(self, data_path):
        return LUNA(data_path)

    def get_model(self, size):
        return CT(size=size)


experiment = Experiment1(DATA_PATH, SAVES_PATH)
experiment.find_good_lambda(32)
for k in range(10):
    experiment.train(200)
experiment.log_optimization(32, 50, 0.5, .7)