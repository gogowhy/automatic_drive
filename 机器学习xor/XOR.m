clear; close all; clc;

% 初始化
% 样本
input = [0, 0; 0, 1; 1, 0; 1, 1];
output = [0; 1; 1; 0];
m = size(input, 1);
% 初始权值
w_1 = [1, 1; -1, -1; 0, 0];
w_2 = [1; -1; 0];
% 学习率
lr = 0.6;
% 初始误差
err = 0.5;
% 迭代次数
j = 0;
% 统计误差，进行绘图
E = [];

% 迭代过程
while err >= 0.008
    err = 0;
    for i = 1 : m
        % 正向信号传递
        % 输入层激励信号
        x_1(i, :) = [input(i, :), 1];
        % 隐层激励信号，sigmoid为激活函数
        x_2(i, :) = [sigmoid(x_1(i, :) * w_1), 1];
        % 输出层结果
        y(i, :) = sigmoid(x_2(i, :) * w_2);
        % 计算误差
        err = err + 0.5 * (output(i) - y(i))^2;
        % 反向误差传递
        delta_3 = (y(i) - output(i)) * y(i) * (1 - y(i));
        delta_21 = delta_3 * w_2(1) * x_2(i, 1) * (1 - x_2(i, 1));
        delta_22 = delta_3 * w_2(2) * x_2(i, 2) * (1 - x_2(i, 2));
        w_2(1) = w_2(1) - lr * delta_3 * x_2(i, 1);
        w_2(2) = w_2(2) - lr * delta_3 * x_2(i, 2);
        w_2(3) = w_2(3) - lr * delta_3 * x_2(i, 3);
        w_1(1, 1) = w_1(1, 1) - lr * delta_21 * x_1(i, 1);
        w_1(2, 1) = w_1(2, 1) - lr * delta_21 * x_1(i, 2);
        w_1(3, 1) = w_1(3, 1) - lr * delta_21 * x_1(i, 3);
        w_1(1, 2) = w_1(1, 2) - lr * delta_22 * x_1(i, 1);
        w_1(2, 2) = w_1(2, 2) - lr * delta_22 * x_1(i, 2);
        w_1(3, 2) = w_1(3, 2) - lr * delta_22 * x_1(i, 3);
    end
    % 迭代次数+1
    j = j + 1;
    % 统计误差
    E = [E err];
end

fprintf('输入组合  期望输出  实际输出\n');
for i = 1 : 4
    fprintf('  %d  %d       %d      %4f\n',input(i, 1), input(i, 2), output(i), y(i));
end

fprintf('\n迭代 %d 轮结束\n此时误差为 %f \n\n', j, err);
i = 1 : j;
plot(i, E(i));
xlabel('迭代次数');
ylabel('误差');