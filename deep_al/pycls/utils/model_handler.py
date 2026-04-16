import torch
import deep_al.pycls.core.builders as model_builder
import deep_al.pycls.utils.checkpoint as cu
from deep_al.pycls.utils.meters import TestMeter

def get_model(cfg, ckpt_file=None, model_init_state=None):
    model = model_builder.build_model(cfg)
    if ckpt_file is not None:
        model = cu.load_checkpoint(ckpt_file, model, weights_only=False)
    if model_init_state is not None and not cfg.MODEL.USE_1NN and not getattr(cfg.MODEL, 'USE_LOGISTIC_REGRESSION', False):
        model.load_state_dict(model_init_state)
    return model


def get_label_from_model(images_loader, checkpoint_file, cfg, model=None):
    """
    returns the labels of the images according to the checkpoint file model
    """
    get_label_meter = TestMeter(len(images_loader))
    if model is None:
        model = get_model(cfg, checkpoint_file)

    pred = get_label_epoch(images_loader, model, get_label_meter)
    return pred


def get_label_epoch(images_loader, model, get_label_meter):
    """get labels according to the model."""
    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    get_label_meter.iter_tic()

    all_preds = []
    for cur_iter, (inputs, _) in enumerate(images_loader):
        with torch.no_grad():
            inputs = inputs.cuda().type(torch.cuda.FloatTensor)
            preds = model(inputs)
            all_preds += preds

    final_preds = [torch.argmax(p).item() for p in all_preds]
    model.train()

    return final_preds
