# based on cuda11.3.1 cudnn8 nvidia/cuda container

from utils import *
from args import *
from trainer import *

init_file_path()
train_loader, val_loader, actual_val_loader = get_dataset()
model = get_model()
torch.cuda.set_device(parser_args.gpu)
model.cuda(parser_args.gpu)

criterion = get_criterion()
optimizer = get_optimizer(model)

train(model, train_loader, val_loader, optimizer, criterion)

if parser_args.sparse:
    get_sparse_model_acc(model, val_loader, criterion)
    get_sparse_model_acc_train(model, train_loader, criterion)