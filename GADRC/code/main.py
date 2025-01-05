import argparse
import time
from GADRC.code.test import model_test
from GADRC.code.train import model_train
from utils import *
from model import Model
from self_define_loss import *
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=3000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--negative_rate', type=float, default=1, help='negative_rate')
    parser.add_argument('--dataset', default='C-dataset', help='dataset')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout')

    parser.add_argument('--dis_hidden_1', default=128, type=int, help='disease hidden number(GCN1)')
    parser.add_argument('--dis_hidden_2', default=128, type=int, help='disease hidden number(GCN2)')
    parser.add_argument('--dis_DRSNet_output', default=128, type=int, help='DRSNet_output_channels')
    # dis_hidden_1得与dis_DRSNet_output数值一样
    parser.add_argument('--dis_input_channels', default=128, type=int, help='output_channels(GAT)')
    parser.add_argument('--dis_output', default=64, type=int, help='disease output channels')

    parser.add_argument('--drug_hidden_1', default=128, type=int, help='drug hidden number(GCN1)')
    parser.add_argument('--drug_hidden_2', default=128, type=int, help='drug hidden number(GCN2)')
    parser.add_argument('--drug_DRSNet_output', default=128, type=int, help='DRSNet_output_channels')
    # drug_hidden_1得与drug_DRSNet_output数值一样
    parser.add_argument('--drug_input_channels', default=128, type=int, help='output_channels(GAT)')
    parser.add_argument('--drug_output', default=64, type=int, help='drug output channels')  # 256->64

    parser.add_argument('--att_head_num', default=1, type=int, help='attention head number(GAT)')

    args = parser.parse_args()
    args.data_dir = '../data/' + args.dataset + '/'
    data = get_data(args)

    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']

    data = data_processing(data, args)

    data = k_fold(data, args)

    data = move_to_device(data, Device)

    # loss_function = torch.nn.functional.binary_cross_entropy_with_logits  # 使用二元交叉熵损失函数来作为损失函数
    loss_function = weighted_cross_entropy_loss

    header = '{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}'.format(
        'Epoch', 'Time', 'AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Mcc', 'loss'
    )

    AUCs, AUPRs, ACCs, Precisions, Recalls, F1s, Mccs = [], [], [], [], [], [], []
    # 开始计时
    start_time = time.time()
    print('Dataset:{}'.format(args.dataset))
    for i in range(args.k_fold):

        # 每一折弄出一个文件
        print('fold:', i)
        print(header)

        model = Model(args)  # 实例化模型
        model = model.to(Device)  # 将模型移动到GPU上

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_auc = 0
        best_aupr = 0
        # 获取第i折的训练集和测试集以及它们对应的标签
        X_train = torch.LongTensor(data['X_train'][i]).to(Device)  # 训练集索引
        Y_train = torch.LongTensor(data['Y_train'][i]).to(Device).flatten()  # 训练集标签
        X_test = torch.LongTensor(data['X_test'][i]).to(Device)  # 测试集索引
        Y_test = torch.LongTensor(data['Y_test'][i]).to(Device).flatten()  # 测试集标签

        for epoch in range(args.epochs):
            # output是模型前馈网络最后生成的663*409的矩阵，它是一个分数矩阵
            loss, output = model_train(model, optimizer, loss_function, data, X_train.T, Y_train)

            auc, acc, aupr, precision, recall, f1, mcc = model_test(model, X_test.T, Y_test, output)

            # 计算并打印总共训练所花费的时间
            time1 = time.time() - start_time

            metrics = [epoch, time1, auc, aupr, acc, precision, recall, f1, mcc, loss]
            row = '{:<10}{:<10.2f}{:<10.5f}{:<10.5f}{:<10.5f}{:<10.5f}{:<10.4f}{:<10.5f}{:<10.4f}{:<10.5f}'.format(*metrics)
            print(row)

            if auc > best_auc:
                best_epoch = epoch + 1
                best_auc = auc
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = aupr, acc, precision, recall, f1, mcc
                print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc, ';\tbest_aupr:', best_aupr)
        AUCs.append(best_auc)
        AUPRs.append(best_aupr)
        ACCs.append(best_accuracy)
        Precisions.append(best_precision)
        Recalls.append(best_recall)
        F1s.append(best_f1)
        Mccs.append(best_mcc)

    print_metrics('AUC', AUCs)
    print_metrics('AUPR', AUPRs)
    print_metrics('ACC', ACCs)
    print_metrics('Precision', Precisions)
    print_metrics('Recall', Recalls)
    print_metrics('F1', F1s)
    print_metrics('Mcc', Mccs)




















