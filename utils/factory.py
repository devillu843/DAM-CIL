from models.DAM import DAM


def get_model(model_name, args):

    if model_name == "dam":
        return DAM(args)
    else:
        assert 0
