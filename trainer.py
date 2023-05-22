import tqdm, sys
from args import *
from utils import *

def train(model, train_loader, val_loader, optimizer, criterion):
    parser_args.mode = 'train'
    scheduler = get_scheduler(optimizer)
    iter_per_epoch = len(train_loader)
    if parser_args.warm > 0:
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * parser_args.warm)
    acc1_list = []
    for epoch in range(0, parser_args.epoch):
        print("training epoch {}".format(epoch))
        for i, (images, target) in tqdm.tqdm(
            enumerate(train_loader), ascii=True, total=len(train_loader)
        ):

            images = images.cuda(parser_args.gpu, non_blocking=True)
            target = target.cuda(parser_args.gpu, non_blocking=True)

            if parser_args.model == 'AlexNet':
                images = F.interpolate(images, scale_factor=7)

            output = model(images)
            if parser_args.criterion in ['mse']:
                label = torch.nn.functional.one_hot(target, num_classes=np.shape(output)[1]).to(torch.float)
            else:
                label = target
            loss = criterion(output, label)

            if parser_args.l1:
                regularization_loss = torch.tensor(0)
                regularization_loss = get_regularization_loss_my(model, regularizer=parser_args.regularizer, lmbda=parser_args.lmbda)
                loss += regularization_loss

            optimizer.zero_grad()

            loss.backward()
            
            optimizer.step()
            if epoch < parser_args.warm:
                warmup_scheduler.step()
            if parser_args.schedu:
                scheduler.step()
        
        acc1, loss = validate(model, val_loader, criterion)
        
        acc1_list.append(acc1.cpu().numpy())
        model.train()

        write_to_csv(acc1_list)

        if parser_args.save_model:
            save_model(model, epoch)

def validate(model, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        all_loss = 0
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):

            images = images.cuda(parser_args.gpu, non_blocking=True)
            target = target.cuda(parser_args.gpu, non_blocking=True)
            if parser_args.model == 'AlexNet':
                images = F.interpolate(images, scale_factor=7)

            output = model(images)
            if parser_args.criterion in ['mse']:
                label = torch.nn.functional.one_hot(target, num_classes=np.shape(output)[1]).to(torch.float)
            else:
                label = target
            loss = criterion(output, label)

            regularization_loss = torch.tensor(0)
            if parser_args.l1:
                regularization_loss = torch.tensor(0)
                regularization_loss = get_regularization_loss_my(model, regularizer=parser_args.regularizer,lmbda=parser_args.lmbda)


            _, predicted = torch.max(output.data, 1)

            total += target.size(0)
            correct +=(predicted == target).sum()

            all_loss += loss.detach().cpu()*target.size(0)
        
        acc1 = 100*correct/total
        print(acc1)
        if parser_args.l1:
            loss_result = all_loss/total+regularization_loss
        else:
            loss_result = all_loss/total
    
    return acc1, loss_result

def get_sparse_model_acc(model, val_loader, criterion):
    parser_args.mode = 'sparse'
    acc1_list  = []
    loss_list = []
    for i in range(0,parser_args.sparse_epoch):
        print("sparsing epoch {}".format(i))
        sparsity_rate = i/parser_args.sparse_epoch
        sparse_model = get_sparse_model(model, sparsity_rate)

        acc1, loss = validate(sparse_model, val_loader, criterion)
        acc1_list.append(acc1.cpu().numpy())
        loss_list.append(loss.detach().cpu().numpy())

        write_to_csv(acc1_list)
        write_AL_to_csv(acc1_list, loss_list)

        if parser_args.save_model:
            save_model(sparse_model, i)

def get_sparse_model_acc_train(model, val_loader, criterion):
    parser_args.mode = 'sparse_train'
    acc1_list  = []
    loss_list = []
    for i in range(0,parser_args.sparse_epoch):
        print("sparsing epoch {}".format(i))
        sparsity_rate = i/parser_args.sparse_epoch
        sparse_model = get_sparse_model(model, sparsity_rate)

        acc1, loss = validate(sparse_model, val_loader, criterion)
        acc1_list.append(acc1.cpu().numpy())
        loss_list.append(loss.detach().cpu().numpy())

        write_to_csv(acc1_list)
        write_AL_to_csv(acc1_list, loss_list)

        if parser_args.save_model:
            save_model(sparse_model, i)