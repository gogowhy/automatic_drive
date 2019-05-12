clear; close all; clc;

% ��ʼ��
% ����
input = [0, 0; 0, 1; 1, 0; 1, 1];
output = [0; 1; 1; 0];
m = size(input, 1);
% ��ʼȨֵ
w_1 = [1, 1; -1, -1; 0, 0];
w_2 = [1; -1; 0];
% ѧϰ��
lr = 0.6;
% ��ʼ���
err = 0.5;
% ��������
j = 0;
% ͳ�������л�ͼ
E = [];

% ��������
while err >= 0.008
    err = 0;
    for i = 1 : m
        % �����źŴ���
        % ����㼤���ź�
        x_1(i, :) = [input(i, :), 1];
        % ���㼤���źţ�sigmoidΪ�����
        x_2(i, :) = [sigmoid(x_1(i, :) * w_1), 1];
        % �������
        y(i, :) = sigmoid(x_2(i, :) * w_2);
        % �������
        err = err + 0.5 * (output(i) - y(i))^2;
        % ��������
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
    % ��������+1
    j = j + 1;
    % ͳ�����
    E = [E err];
end

fprintf('�������  �������  ʵ�����\n');
for i = 1 : 4
    fprintf('  %d  %d       %d      %4f\n',input(i, 1), input(i, 2), output(i), y(i));
end

fprintf('\n���� %d �ֽ���\n��ʱ���Ϊ %f \n\n', j, err);
i = 1 : j;
plot(i, E(i));
xlabel('��������');
ylabel('���');