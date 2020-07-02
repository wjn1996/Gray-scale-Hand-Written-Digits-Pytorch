import matplotlib.pyplot as plt
import numpy as np
# 记录训练和测试过程中的loss和acc变化
save_path = './model_save/'
train_loss = np.load(save_path + 'train_loss.npy')
train_acc = np.load(save_path + 'train_acc.npy')
test_loss = np.load(save_path + 'test_loss.npy')
test_acc = np.load(save_path + 'test_acc.npy')
# print(train_loss, '\n', train_acc, '\n', test_loss, '\n', test_acc)
train_loss_ = [train_loss[i] for i in range(0, len(train_loss), 100)]
train_acc_ = [train_acc[i] for i in range(0, len(train_acc), 100)]


fig1 = plt.figure()
ax1 = fig1.add_subplot(221)
ax2 = fig1.add_subplot(222)
ax1.set(xlim=[0,len(train_loss_)], ylim=[0,1], title='train loss',ylabel='loss', xlabel='train times')
ax1.plot([i for i in range(len(train_loss_))], train_loss_)
ax2.set(xlim=[0,len(test_loss)], ylim=[0,0.01], title='test loss',ylabel='loss', xlabel='test times')
ax2.plot([i for i in range(len(test_loss))], test_loss)

# fig2 = plt.figure()
ax3 = fig1.add_subplot(223)
ax4 = fig1.add_subplot(224)
ax3.set(xlim=[0,len(train_acc_)], ylim=[0.7,1], title='train acc',ylabel='acc(%)', xlabel='train times')
ax3.plot([i for i in range(len(train_acc_))], train_acc_)
ax4.set(xlim=[0,len(test_acc)], ylim=[0.95,1], title='test acc',ylabel='acc(%)', xlabel='test times')
ax4.plot([i for i in range(len(test_acc))], test_acc)

plt.show()