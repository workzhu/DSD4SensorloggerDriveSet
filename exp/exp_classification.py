from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.nn.utils import clip_grad_norm_

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

import numpy as np

import seaborn as sns

import csv

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):

    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    # 初始化模型
    def _build_model(self):

        # 获取序列最长长度
        self.args.seq_len = self.args.window_size

        self.args.pred_len = 0

        # 获取特征维度
        self.args.enc_in = self.dataset.enc_in

        # 获取标签数
        self.args.num_class = self.dataset.num_class

        self.total_samples = sum(self.dataset.label_counts.values())

        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self):
        dataset, train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = \
            data_provider(self.args)
        return dataset, train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader

    def _select_optimizer(self):
        # model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0.0025)
        # model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=0.001)
        return model_optim

    def _select_criterion(self):

        # 打印标签顺序
        sorted_labels = sorted(self.train_dataset.label_to_index.items(), key=lambda item: item[1])
        for label, index in sorted_labels:
            print(f"Index: {index}, Label: {label}")

        # 根据标签顺序构建权重
        weights = {class_label: self.train_dataset.total_samples / count for class_label, count in
                       self.train_dataset.label_counts.items()}

        weights_tensor = torch.tensor([weights[label] for label, _ in sorted_labels], dtype=torch.float32)
        print(weights_tensor)
        weights_tensor = weights_tensor.to(self.device)

        # 创建带有惩罚权重的损失函数
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        # criterion = nn.CrossEntropyLoss()
        # criterion = torch.nn.BCELoss()
        # criterion = criterion.to(self.device)
        return criterion

    def validate(self, vali_loader, criterion):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_true_labels = []
        with torch.no_grad():
            for i, (batch_x, window_features, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                window_features = window_features.float().to(self.device)

                # 将标签转换为类别索引
                #  true_labels = torch.argmax(batch_y, dim=1)
                true_labels = batch_y.long()
                true_labels = true_labels.to(self.device)

                outputs = self.model(batch_x, window_features)

                loss = criterion(outputs, true_labels)
                # outputs = outputs.squeeze()
                # loss = criterion(outputs, true_labels.float())

                total_loss += loss.item()

                preds = torch.nn.functional.softmax(outputs.detach(), dim=1)

                predictions = torch.argmax(preds, dim=1)

                # predictions = (outputs > 0.5).long()

                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(true_labels.cpu().numpy())

        avg_loss = total_loss / len(vali_loader)
        avg_accuracy = accuracy_score(all_true_labels, all_predictions)
        avg_precision = precision_score(all_true_labels, all_predictions, average='macro')
        avg_recall = recall_score(all_true_labels, all_predictions, average='macro')
        avg_f1 = f1_score(all_true_labels, all_predictions, average='macro')

        return avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1

    # 训练
    def train(self, setting):

        # 根据设置的参数创建用于存储模型检查点的路径
        path = os.path.join(self.args.checkpoints, setting)

        if not os.path.exists(path):
            # 如果路径不存在，则创建该路径
            os.makedirs(path)

        # 记录当前时间，用于计算训练耗时
        time_now = time.time()

        saved_metrics_list = []

        # Rebuild the model for each fold to reset weights
        self.model = self._build_model().to(self.device)

        # Reset the optimizer
        model_optim = self._select_optimizer()

        # 选择损失函数
        criterion = self._select_criterion()

        # 初始化早停机制，根据设置的耐心值来决定何时停止训练
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        folder_path = './results/' + setting + '/'

        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 打开一个文件用于追加写入指标
        with open(folder_path + "metrics.csv", "w", newline="") as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(
                ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1'])

        for epoch in range(self.args.train_epochs):

            # 将模型设置为训练模式
            self.model.train()

            # 记录当前周期的开始时间
            epoch_time = time.time()

            total_loss = 0
            total_acc = 0

            # 遍历训练数据加载器中的每个批次
            for i, (batch_x, window_features, batch_y) in enumerate(self.train_loader):
                # 清除之前的梯度
                model_optim.zero_grad()
                # 将数据和标签转移到设备（例如GPU）
                batch_x = batch_x.float().to(self.device)
                window_features = window_features.float().to(self.device)

                # 将标签转换为类别索引
                # true_labels = torch.argmax(batch_y, dim=1)

                true_labels = batch_y.long()
                true_labels = true_labels.to(self.device)

                outputs = self.model(batch_x, window_features)

                loss = criterion(outputs, true_labels)
                # outputs = outputs.squeeze()
                # loss = criterion(outputs, true_labels.float())

                loss.backward()

                clip_grad_norm_(self.model.parameters(), max_norm=4.0)

                model_optim.step()

                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                # predictions = (outputs > 0.5).long()

                acc = (predictions == true_labels).float().mean()

                total_acc += acc.item()

            avg_train_loss = total_loss / len(self.train_loader)
            avg_train_acc = total_acc / len(self.train_loader)

            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate(self.valid_loader, criterion)

            # test_loss, test_acc, test_precision, test_recall, test_f1 = self.validate(self.test_loader, criterion)

            print(f"Epoch {epoch + 1}/{self.args.train_epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
                  # f"Test Loss: {test_loss:.4f},Test F1: {test_f1:.4f}")

            # 在每轮训练后立即将指标追加到文件
            with open(folder_path + "metrics.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [epoch + 1, avg_train_loss, avg_train_acc, val_loss, val_acc, val_precision, val_recall,
                     val_f1])

            early_stopping(val_f1, self.model, path)

            # adjust_learning_rate(model_optim, epoch + 1, self.args)

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Training complete in {time.time() - time_now:.2f} seconds")

        return self.model

    def visualize_tsne(self, features, labels, title, folder_path):
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features)

        plt.figure(figsize=(8, 6))

        ## Code Availability

        # Part of the code used in this project will be made publicly available after the related research paper is
        # accepted for publication.We will update this repository with the code at that time.
        # For any inquiries, please contact workzhu@outlook.com.

        plt.savefig(os.path.join(folder_path, f"{title}.png"))  # 保存图像
        plt.close()  # 关闭图像

    # 在一个单独的测试函数中调用
    def test(self, setting, test=0):

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        all_predictions = []
        all_true_labels = []

        all_features_before = []
        all_features_after = []
        with torch.no_grad():
            for i, (batch_x, window_features, batch_y) in enumerate(self.test_loader):
                batch_x = batch_x.float().to(self.device)
                window_features = window_features.float().to(self.device)

                # 提取输入数据作为特征提取前的特征
                features_before = batch_x.view(batch_x.size(0), -1).detach().cpu().numpy()  # 展平为二维
                all_features_before.extend(features_before)

                # 将标签转换为类别索引
                # true_labels = torch.argmax(batch_y, dim=1)

                true_labels = batch_y.long()
                true_labels = true_labels.to(self.device)

                outputs = self.model(batch_x, window_features)

                all_features_after.extend(outputs.detach().cpu().numpy())

                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(true_labels.cpu().numpy())

        avg_accuracy = accuracy_score(all_true_labels, all_predictions)
        avg_precision = precision_score(all_true_labels, all_predictions, average='macro')
        avg_recall = recall_score(all_true_labels, all_predictions, average='macro')
        avg_f1 = f1_score(all_true_labels, all_predictions, average='macro')

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 调用 t-SNE 可视化函数
        self.visualize_tsne(np.array(all_features_before), all_true_labels, 't-SNE Before Feature Extraction',
                            folder_path)
        self.visualize_tsne(np.array(all_features_after), all_true_labels, 't-SNE After Feature Extraction',
                            folder_path)

        print(f"Val Acc: {avg_accuracy:.4f}, "
              f"Val Precision: {avg_precision:.4f}, Val Recall: {avg_recall:.4f}, Val F1: {avg_f1:.4f}")

        file_name = 'result_classification.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write(setting + "  \n")
        f.write(f"Val Acc: {avg_accuracy:.4f}, "
                f"Val Precision: {avg_precision:.4f}, Val Recall: {avg_recall:.4f}, Val F1: {avg_f1:.4f}")
        f.write('\n')
        f.write('\n')
        f.close()

        # 标签对应关系
        labels = ['aggressive', 'normal']

        # 计算混淆矩阵
        conf_mat = confusion_matrix(all_true_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels,
                    cbar=False, fmt="d")  # 显示数量

        # 计算每个类别的F1分数
        f1_scores = f1_score(all_true_labels, all_predictions, average=None)

        # 设置阈值
        thresh = conf_mat.max() / 2  # 使用矩阵最大值的一半作为阈值

        # 在每个单元格中显示数量和F1分数
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                if i == j:
                    info = f"{conf_mat[i, j]} (F1: {f1_scores[i]:.2f})"
                else:
                    info = f"{conf_mat[i, j]}"

                # 根据阈值设置颜色
                color = "white" if conf_mat[i, j] > thresh else "black"

                plt.text(j + 0.5, i + 0.5, info, ha="center", va="center", color=color, fontsize=10, weight="bold")

        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (with Counts and F1 Score)')
        plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45)
        plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
        plt.tight_layout()  # 保证图像不重叠
        plt.savefig(folder_path + 'confusion_matrix.png')  # 保存到安全的路径
        plt.show()

    def kmeans_clustering(self, train_x, valid_x, test_x, test_y):
        best_k = 2
        best_score = -1
        for k in range(2, 20):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(train_x)
            valid_labels = kmeans.predict(valid_x)
            score = silhouette_score(valid_x, valid_labels)
            if score > best_score:
                best_score = score
                best_k = k

        kmeans = KMeans(n_clusters=best_k, random_state=42)
        kmeans.fit(train_x)
        cluster_labels = kmeans.predict(test_x)

        kmeans_accuracy = accuracy_score(test_y, cluster_labels)
        kmeans_precision = precision_score(test_y, cluster_labels, average='weighted')
        kmeans_recall = recall_score(test_y, cluster_labels, average='weighted')
        kmeans_f1 = f1_score(test_y, cluster_labels, average='weighted')

        return cluster_labels, kmeans_accuracy, kmeans_precision, kmeans_recall, kmeans_f1, kmeans.cluster_centers_

    def random_forest(self, train_x, train_y, valid_x, valid_y, test_x, test_y):
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1,
                                   scoring='accuracy')
        grid_search.fit(train_x, train_y)
        best_params = grid_search.best_params_

        clf = RandomForestClassifier(**best_params, random_state=42)
        clf.fit(train_x, train_y)
        y_pred = clf.predict(test_x)

        accuracy = accuracy_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred, average='weighted')
        recall = recall_score(test_y, y_pred, average='weighted')
        f1 = f1_score(test_y, y_pred, average='weighted')

        return y_pred, accuracy, precision, recall, f1

    def svm_classifier(self, train_x, train_y, valid_x, valid_y, test_x, test_y):
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf']
        }
        grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=3, n_jobs=-1, scoring='accuracy')
        grid_search.fit(train_x, train_y)
        best_params = grid_search.best_params_

        clf = SVC(**best_params, random_state=42)
        clf.fit(train_x, train_y)
        y_pred = clf.predict(test_x)

        accuracy = accuracy_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred, average='weighted')
        recall = recall_score(test_y, y_pred, average='weighted')
        f1 = f1_score(test_y, y_pred, average='weighted')

        return y_pred, accuracy, precision, recall, f1

    def decision_tree(self, train_x, train_y, valid_x, valid_y, test_x, test_y):
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=3, n_jobs=-1,
                                   scoring='accuracy')
        grid_search.fit(train_x, train_y)
        best_params = grid_search.best_params_

        clf = DecisionTreeClassifier(**best_params, random_state=42)
        clf.fit(train_x, train_y)
        y_pred = clf.predict(test_x)

        accuracy = accuracy_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred, average='weighted')
        recall = recall_score(test_y, y_pred, average='weighted')
        f1 = f1_score(test_y, y_pred, average='weighted')

        return y_pred, accuracy, precision, recall, f1

    def adaboost(self, train_x, train_y, valid_x, valid_y, test_x, test_y):
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 1]
        }
        grid_search = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, scoring='accuracy')
        grid_search.fit(train_x, train_y)
        best_params = grid_search.best_params_

        clf = AdaBoostClassifier(**best_params, random_state=42)
        clf.fit(train_x, train_y)
        y_pred = clf.predict(test_x)

        accuracy = accuracy_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred, average='weighted')
        recall = recall_score(test_y, y_pred, average='weighted')
        f1 = f1_score(test_y, y_pred, average='weighted')

        return y_pred, accuracy, precision, recall, f1

    def classify(self):
        train_x = []
        train_y = []

        for i, (batch_x, window_features, batch_y) in enumerate(self.train_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            train_x.append(batch_x.cpu().numpy())
            train_y.append(batch_y.cpu().numpy())
        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)

        valid_x = []
        valid_y = []

        for i, (batch_x, window_features, batch_y) in enumerate(self.valid_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            valid_x.append(batch_x.cpu().numpy())
            valid_y.append(batch_y.cpu().numpy())
        valid_x = np.concatenate(valid_x, axis=0)
        valid_y = np.concatenate(valid_y, axis=0)

        test_x = []
        test_y = []

        for i, (batch_x, window_features, batch_y) in enumerate(self.test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            test_x.append(batch_x.cpu().numpy())
            test_y.append(batch_y.cpu().numpy())
        test_x = np.concatenate(test_x, axis=0)
        test_y = np.concatenate(test_y, axis=0)

        # 将输入数据重新塑形为2D
        train_x = train_x.reshape(train_x.shape[0], -1)
        valid_x = valid_x.reshape(valid_x.shape[0], -1)
        test_x = test_x.reshape(test_x.shape[0], -1)

        print("train_x shape:", train_x.shape)
        print("valid_x shape:", valid_x.shape)
        print("test_x shape:", test_x.shape)
        '''
            # K-means 聚类
        cluster_labels, kmeans_accuracy, kmeans_precision, kmeans_recall, kmeans_f1, kmeans_centers = self.kmeans_clustering(train_x, valid_x, test_x, test_y)
        print("K-means聚类标签:", cluster_labels)
        print("K-means质心:", kmeans_centers)
        print("K-means Accuracy:", kmeans_accuracy)
        print("K-means Precision:", kmeans_precision)
        print("K-means Recall:", kmeans_recall)
        print("K-means F1 Score:", kmeans_f1)
        '''

        # 随机森林分类
        y_pred_rf, rf_accuracy, rf_precision, rf_recall, rf_f1 = self.random_forest(train_x, train_y, valid_x, valid_y,
                                                                                    test_x, test_y)
        print("Random Forest Accuracy:", rf_accuracy)
        print("Random Forest Precision:", rf_precision)
        print("Random Forest Recall:", rf_recall)
        print("Random Forest F1 Score:", rf_f1)

        # SVM分类
        y_pred_svm, svm_accuracy, svm_precision, svm_recall, svm_f1 = self.svm_classifier(train_x, train_y, valid_x,
                                                                                          valid_y, test_x, test_y)
        print("SVM Accuracy:", svm_accuracy)
        print("SVM Precision:", svm_precision)
        print("SVM Recall:", svm_recall)
        print("SVM F1 Score:", svm_f1)

        # 决策树分类
        y_pred_dt, dt_accuracy, dt_precision, dt_recall, dt_f1 = self.decision_tree(train_x, train_y, valid_x, valid_y,
                                                                                    test_x, test_y)
        print("Decision Tree Accuracy:", dt_accuracy)
        print("Decision Tree Precision:", dt_precision)
        print("Decision Tree Recall:", dt_recall)
        print("Decision Tree F1 Score:", dt_f1)

        # Adaboost分类
        y_pred_ab, ab_accuracy, ab_precision, ab_recall, ab_f1 = self.adaboost(train_x, train_y, valid_x, valid_y,
                                                                               test_x, test_y)
        print("Adaboost Accuracy:", ab_accuracy)
        print("Adaboost Precision:", ab_precision)
        print("Adaboost Recall:", ab_recall)
        print("Adaboost F1 Score:", ab_f1)