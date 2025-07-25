import copy
import json

class Individuo:
    def __init__(self, params: dict):
        """
        params: diccionario de hiperparámetros detectron2
        {
            'modalidad': 'FLAIR',
            'modelo': 'mask_rcnn_R_50_FPN_3x',
            'flip': True,
            'batch_size': 4,
            'gamma': 0.1,
            'base_lr': 0.00025,
            'weight_decay': 0.0001,
            'maxiter_steps': 3000
        }
        """
        self.params = params
        self.fitness = None 

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value

    def as_dict(self):
        return self.params

    def as_tuple(self):
        return tuple(self.params[k] for k in sorted(self.params))

    def copy(self):
        return Individuo(copy.deepcopy(self.params))

    def to_json(self):
        return json.dumps(self.params, indent=2)

    def __repr__(self):
        return f"Individuo({self.params}, fitness={self.fitness})"
