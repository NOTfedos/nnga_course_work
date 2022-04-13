import torch
import torch.nn.functional as F  # вроде как для функций, но есть и в nn
from random import choice, randint, uniform


LAYER_TYPES = ("Linear", )  # "Dropout",)
ACTIVATION_TYPES = ("ReLU", "Tanh", "Softmax", "Sigmoid")
CRITERION_TYPES = ("L1Loss", "MSELoss", "CrossEntropyLoss",)
OPTIMIZER_TYPES = ("SGD", "Adam",)


class Entity:
    gens = None

    CRITERIONS = {
        "L1Loss": torch.nn.L1Loss,
        "MSELoss": torch.nn.MSELoss,
        "CrossEntropyLoss": torch.nn.CrossEntropyLoss
    }

    def __init__(self, gens: dict):
        self.gens = gens

        # Creating the NN model
        self.layers = []
        for layer in self.gens["layers"]:
            if layer["type"] == "Linear":
                self.layers.append(torch.nn.Linear(layer["in"], layer["out"]))
            if layer["type"] == "ReLU":
                self.layers.append(torch.nn.ReLU())
            if layer["type"] == "Sigmoid":
                self.layers.append(torch.nn.Sigmoid())
            if layer["type"] == "Softmax":
                self.layers.append(torch.nn.Softmax())
            if layer["type"] == "Tanh":
                self.layers.append(torch.nn.Tanh())
            if layer["type"] == "Dropout":
                self.layers.append(torch.nn.Dropout(p=layer["p"], inplace=layer["inplace"]))
        self.model = torch.nn.Sequential(*self.layers)

        self.loss = self.CRITERIONS[gens["criterion"]]()

        if gens["optimizer"] is not None:
            opt = gens["optimizer"]
            if opt["name"] == "SGD":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt["lr"], momentum=opt["momentum"])
            if opt["name"] == "Adam":
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt["lr"])

    # Обучение одной эпохи
    def _fit(self, train_loader):

        res_loss = 0
        for data in train_loader:
            inputs, labels = data

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            ans_loss = self.loss(outputs, labels)
            ans_loss.backward()
            self.optimizer.step()

            res_loss += ans_loss.item()

        return res_loss / len(train_loader)

    # Метод обучения на всём датасете в течение эпох
    def train(self, train_loader, epochs):
        loss_history = []
        for i in range(epochs):
            loss_history.append(self._fit(train_loader))
        return loss_history

    # Метод получения предсказания по одному примеру
    def predict(self, data):
        with torch.no_grad():
            return self.model(data)


class Environment:
    entities = None

    def __init__(self, entity_count, train_loader, train_epochs=50, validation_loader=None, test_loader=None):
        self.entity_count = entity_count
        self.create_entities(entity_count)

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.train_epochs = train_epochs

        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')

    # Создание существ
    def create_entities(self, entity_count):
        self.entities = []
        for i in range(entity_count):
            self.entities.append(Entity(self.generate_random_gen()))

    # Обучение особей на 1 эволюционной эпохе
    def train_epoch(self):
        for ent in self.entities:
            h = ent.train(self.train_loader, self.train_epochs)
            print(h[-3:])

    # Валидация результатов обучения особей
    def validate_entities(self):
        pass

    # Генерация рандомного набора генов для одной особи
    def generate_random_gen(self):

        # Set layer count
        layers_count = randint(1, 10)

        # Creating first layer (with 1 input)
        layers = []
        l_type = choice(LAYER_TYPES)
        layer = {
            "type": l_type,
            "in": 1,
            "out": randint(1, 40)
        }
        layers.append(layer)

        # Generating other layers
        for i in range(1, layers_count):
            if randint(1, 3) == 1:  # Обычные слои
                l_type = choice(LAYER_TYPES)
                layer = {
                    "type": l_type,
                    "in": layers[i-1]["out"],
                    "out": randint(1, 40)
                }
            else:  # Слои активации
                l_type = choice(ACTIVATION_TYPES)
                layer = {
                    "type": l_type,
                    "in": layers[i-1]["out"],
                    "out": layers[i-1]["out"]
                }
            layers.append(layer)
        layers.append({
            "type": "Linear",
            "in": layers[-1]["out"],
            "out": 1
        })

        # Generating optimizer
        opt = dict()
        opt_name = choice(OPTIMIZER_TYPES)
        if opt_name == "SGD":
            opt["momentum"] = uniform(0.0001, 0.5)
        opt["name"] = opt_name
        opt["lr"] = uniform(0.01, 0.7)

        # Compiling gen
        gen = {
            "layers": layers,
            "criterion": choice(CRITERION_TYPES),
            "optimizer": opt
        }
        print(gen)
        return gen
