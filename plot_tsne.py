import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
def scatter(x, colors, n_class):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", n_class))

    # We create a scatter plot.
    f = plt.figure(figsize=(10, 10))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int32)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    # plt.grid(c='r')
    ax.axis('on')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(n_class):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts
def t_sne(x, y, n_class):
    # import warnings filter
    from warnings import simplefilter
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    sns.set_style('whitegrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    from sklearn.manifold import TSNE
    digits_proj = TSNE(random_state=128).fit_transform(x)

    scatter(digits_proj, y, n_class)

    # plt.savefig(savename)
def LoadData_pickle(path,name,type='rb'):
  with open(path+name+'.pkl', type) as f:
          data=pickle.load(f)

  return data

faults=['C3S3L0_all','C4S3L0_all','C6S3L0_all','C7S3L0_all','C8S3L0_all']
# real load
dataset_real=[]
for fault in faults:
    x=LoadData_pickle(path='../dataset/original/',
                            name=fault)[5]

    dataset_real.extend(x)
dataset_real=np.array(dataset_real)

#generated load
faults=['gen_data_C_1','gen_data_C_2','gen_data_C_3','gen_data_C_4','gen_data_C_5']
dataset_gen=[]
for fault in faults:
    x=LoadData_pickle(path='../dataset/generated/',
                            name=fault)

    dataset_gen.extend(x)
dataset_gen=np.array(dataset_gen)

# x=dataset_gen
x=np.vstack((dataset_real,dataset_gen))
num=1000
y=np.array([0]*num+[1]*num+[2]*num+[3]*num+[4]*num+
           [5]*num+[6]*num+[7]*num+[8]*num+[9]*num)
print('start')
t_sne(x,y,n_class=10)
plt.show()
plt.savefig('./1.png')

