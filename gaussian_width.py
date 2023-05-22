from cal_rate import *
from yaohessian import *
from utils import *

init_file_path()
if 'FC' in parser_args.model:
    num = 0
else:
    gradient_dict = np.load(parser_args.file_path + 'gradient_dict.npy', allow_pickle=True).item()
    add_threshold_list = gradient_dict["add_threshold_list"]

    num = 0
    for item in add_threshold_list:
        if item < 1e-7:
            num+=1
        print(item)
    print(num)

get_density(iter_num=parser_args.lanczos_iter, n_v_num=parser_args.lanczos_num, index=0)
trace = None
index_list = []
index = 0
index_list.append(index)
sparse_index = cal_sparse_rate(index=index, trace=trace, important_ev_num=num*10)
print('important ev rate: ', num/100, ' predicted maxium pruning ratio: ', sparse_index/parser_args.sparse_epoch)

csv_data = pd.read_csv(parser_args.file_path + 'sparse/acc_loss.csv')
orign_acc = csv_data['acc'][0]
for i in range(parser_args.sparse_epoch):
    sparse_acc = csv_data['acc'][parser_args.sparse_epoch-1-i]
    if sparse_acc > orign_acc:
        sparse_threshold = (parser_args.sparse_epoch-1-i)/parser_args.sparse_epoch
        print('actual maximum pruning ratio: ', sparse_threshold)
        break

with open(parser_args.file_path + 'result.txt','w') as f:
    f.write('important ev rate: {}\n'.format(num/100))
    f.write('predicted maximum pruning ratio: {}\n'.format(sparse_index/parser_args.sparse_epoch))
    f.write('actual maximum pruning ratio: {}\n'.format(sparse_threshold))