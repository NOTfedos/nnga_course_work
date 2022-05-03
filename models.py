import torch
# import torch.nn.functional as F  # вроде как для функций, но есть и в nn
from random import choice, randint, uniform
from copy import deepcopy


LAYER_TYPES = ("Linear", )  # "Dropout",)
ACTIVATION_TYPES = ("ReLU", "Tanh", "Sigmoid")
CRITERION_TYPES = ("MSELoss", )  # ("L1Loss", "MSELoss")
OPTIMIZER_TYPES = ("SGD", "Adam",)


class Entity:
    gens = None
    model = None
    loss = None
    optimizer = None
    
    parent = None
    child = None

    CRITERIONS = {
        "L1Loss": torch.nn.L1Loss,
        "MSELoss": torch.nn.MSELoss,
        "BCELoss": torch.nn.BCELoss
    }

    def __init__(self, gens: dict, color, parent=None, entity_history=None):
        self.gens = gens
        self.init_gen()
        self.layers = []
        self.parent = parent
        self.color = color

        if entity_history is not None:
            self.entity_history = entity_history
        else:
            self.entity_history = []

    def init_gen(self):
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

        self.loss = self.CRITERIONS[self.gens["criterion"]]()

        if self.gens["optimizer"] is not None:
            opt = self.gens["optimizer"]
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
    # entities = None
    # val_loss = []
    #
    # evo_epochs = 0

    def __init__(self, entity_count, train_loader, train_epochs=50, validation_loader=None, test_loader=None,
                 in_shape=1, out_shape=1):
        # Set loaders
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.train_epochs = train_epochs

        self.evo_epochs = 0

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.history = []

        # Create entities
        self.entity_count = entity_count if entity_count > 7 else 7
        self.create_entities(entity_count)

        self.val_loss = []

        # Set device for training
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')

        print(device)

    # Entities create with random gens
    def create_entities(self, entity_count):
        self.entities = []
        for i in range(entity_count):
            self.entities.append(Entity(self.generate_random_gen(self.in_shape, self.out_shape), color=i))

    # Train all the entities
    def train_epoch(self):
        for ent in self.entities:
            ent.train(self.train_loader, self.train_epochs)

    # Validate entities (append last loss to each ones)
    def validate_entities(self):
        losses = []
        
        for i, ent in enumerate(self.entities):
            
            ent_loss = 0

            # Loop through all validation dataset and accumulate loss
            for batch in self.validation_loader:
                inputs, labels = batch
                outputs = ent.predict(inputs)
                ans_loss = ent.loss(outputs, labels)
                
                ent_loss += ans_loss.item()

            # Save loss to entity history
            ent_loss = ent_loss / len(self.validation_loader) / self.validation_loader.batch_size
            ent.entity_history.append(ent_loss)
            losses.append(ent_loss)

        # Set the count of losses in loss history lists
        self.evo_epochs += 1
        self.val_loss = losses[:]
        return losses

    # Step the evolution
    def evo_step(self):
        if len(self.val_loss) == 0:
            print("Validate entities")
            return

        # Create dict of val_loss: entity
        ent_losses = dict(zip(self.val_loss, self.entities))

        # Sort the list of val losses
        sorted_val_loss = sorted(self.val_loss, key=lambda x: 1000 if x is None else x)

        # Save first three models with less loss
        top1, top2, top3 = sorted_val_loss[:3]

        # Create new list of entities, first to save the best 3 entities
        new_list = [ent_losses[top1], ent_losses[top2], ent_losses[top3]]

        # Append mutated entities
        for val_loss in sorted_val_loss[:-3]:
            print(val_loss, ent_losses[val_loss].color)
            new_list.append(Entity(deepcopy(ent_losses[val_loss].gens), color=ent_losses[val_loss].color,
                                   entity_history=ent_losses[val_loss].entity_history[:]))
            self.mutate(new_list[-1])
        
        self.entities = new_list
                
        self.reset_models()
        self.entity_count = len(self.entities)
        self.val_loss = []

    # Функция мутации особи
    def mutate(self, entity):

        i = randint(1, len(entity.gens["layers"])-1)
        
        layer = entity.gens["layers"][i]

        # Удаление слоя
        if randint(0, 20) == 1:

            # Проходимся по дальнейшим слоям активации и присваиваем им предыдущие значения входных параметров
            k = i+1
            while (k < len(entity.gens["layers"])) and entity.gens["layers"][k]["type"] in ACTIVATION_TYPES:
                entity.gens["layers"][k]["out"] = entity.gens["layers"][i-1]["out"]
                entity.gens["layers"][k]["in"] = entity.gens["layers"][i-1]["out"]
                k += 1
            # Если все оставшиеся слои - активационные
            if k == len(entity.gens["layers"]):
                # Последущие слои - по 1 параметры
                for j in range(i+1, len(entity.gens["layers"]), 1):
                    entity.gens["layers"][k]["out"] = self.out_shape
                    entity.gens["layers"][k]["in"] = self.out_shape

                k = i - 1
                # Все слои до - тоже 1
                while entity.gens["layers"][k]["type"] in ACTIVATION_TYPES:
                    entity.gens["layers"][k]["out"] = self.out_shape
                    entity.gens["layers"][k]["in"] = self.out_shape
                    k -= 1
                entity.gens["layers"][k]["out"] = self.out_shape
            else:
                entity.gens["layers"][k]["in"] = entity.gens["layers"][i-1]["out"]

            entity.gens["layers"].pop(i)

            return

        # Изменение количества параметров
        if layer["type"] in LAYER_TYPES:
            layer["in"] = int(layer["in"] * (1 + uniform(-0.2, 0.2)))
            k = i-1
            while entity.gens["layers"][k]["type"] in ACTIVATION_TYPES:
                entity.gens["layers"][k]["out"] = layer["in"]
                entity.gens["layers"][k]["in"] = layer["in"]
                k -= 1
            entity.gens["layers"][k]["out"] = layer["in"]
            return
        
        entity.gens["layers"][i] = layer
        
        # Добавить или модифицировать тип слоя
        if randint(0, 10) <= 2:
            p = randint(0, 5)
            if (p > 3) and (layer["type"] in ACTIVATION_TYPES):  # изменяем слой в теле гена
                if randint(1, 2) == 1:
                    entity.gens["layers"][i] = {
                        "type": choice(LAYER_TYPES),
                        "in": entity.gens["layers"][i]["in"],
                        "out": entity.gens["layers"][i]["out"]
                    }
                    return
            elif p > 1:  # Добавляем слой в конец
                if randint(1, 3) == 1:  # Обычные слои
                    layer = {
                        "type": choice(LAYER_TYPES),
                        "in": randint(0, 10),
                        "out": self.out_shape
                    }
                    k = len(entity.gens["layers"]) - 1
                    while entity.gens["layers"][k]["type"] in ACTIVATION_TYPES:
                        entity.gens["layers"][k]["out"] = layer["in"]
                        entity.gens["layers"][k]["in"] = layer["in"]
                        k -= 1
                    entity.gens["layers"][k]["out"] = layer["in"]
                    entity.gens["layers"].append(layer)
                else:  # Слой активации
                    l_type = choice(ACTIVATION_TYPES)
                    layer = {
                        "type": l_type,
                        "in": self.out_shape,
                        "out": self.out_shape
                    }
                    entity.gens["layers"].append(layer)
            else:  # Добавляем слой активации на место i
                layer = {
                    "type": choice(ACTIVATION_TYPES),
                    "in": entity.gens["layers"][i-1]["out"],
                    "out": entity.gens["layers"][i-1]["out"]
                }
                entity.gens["layers"].insert(i, layer)

    # Перегенерируем параметры всех нейронок
    def reset_models(self):
        for ent in self.entities:
            ent.init_gen()
            # for layer in ent.model.children():
            #     if hasattr(layer, 'reset_parameters'):
            #         layer.reset_parameters()

    # Генерация рандомного набора генов для одной особи
    def generate_random_gen(self, in_shape, out_shape):

        # Set layer count
        layers_count = randint(1, 10)

        # Creating first layer (with 1 input)
        layers = []
        l_type = choice(LAYER_TYPES)
        layer = {
            "type": l_type,
            "in": in_shape,
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
            "out": out_shape
        })

        # Generating optimizer
        opt = dict()
        opt_name = choice(OPTIMIZER_TYPES)
        if opt_name == "SGD":
            opt["momentum"] = uniform(0.0001, 0.1)
        opt["name"] = opt_name
        opt["lr"] = uniform(0.01, 0.1)

        # Compiling gen
        gen = {
            "layers": layers,
            "criterion": choice(CRITERION_TYPES),
            "optimizer": opt
        }
        return gen
