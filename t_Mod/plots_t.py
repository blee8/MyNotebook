import matplotlib as mpl
import matplotlib.pyplot as plt

from t_Mod import compile       # relative path
from t_Mod import plots_t



def plot_curve(trg, valg, teg):
    plt.plot(trg['IR'], trg['FlowHt'], 'go', label='train', markersize=5, zorder=1)
    plt.plot(valg['IR'], valg['FlowHt'], 'mo', label='val',  markersize=5, zorder=1)
    plt.plot(teg['IR'], teg['FlowHt'], 'ro', label='test',  markersize=5, zorder=1)

#    plt.plot(trg['IR'], trg['FlowHt'], 'g-',   linewidth=2.0, zorder=2)
#    plt.plot(valg['IR'], valg['FlowHt'], 'm-', linewidth=2.0,  zorder=2)
#    plt.plot(teg['IR'], teg['FlowHt'], 'r-', linewidth=2.0,  zorder=2)
 #   plt.plot(curve_0['IR'], curve_0['FlowHt'], 'k-', label='curve')

    plt.xlabel('norm IR')
    plt.ylabel('norm Flow Height [m]')
    plt.xlim([plt.xlim()[1], plt.xlim()[0]])
    plt.ylim([plt.ylim()[0], plt.ylim()[1]])
    plt.legend(loc='best')



def eval(hist) :
    fig, loss_ax = plt.subplots(figsize =(7,3))

    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    if 'val_loss' in hist.history:
        loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(hist.history['mean_absolute_error'], 'b', label='train MAE')
    if 'val_mean_absolute_error' in hist.history:
        acc_ax.plot(hist.history['val_mean_absolute_error'], 'g', label='val MAE')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('mean_absolute_error')

    loss_ax.legend(loc='lower left')
    acc_ax.legend(loc='upper right')
    #loss_ax.legend(loc='center')
    #acc_ax.legend(loc='lower center')

    plt.show()

MAX_EPOCHS = 20

def add_curve(curve=None, window = None, window_t=None,
              model=None, npat= 2, num_epoch = 20, add_num=0,  df_name=None ) :

    train_loss = []
    val_loss = []
    train_mae = []
    val_mae = []

    #for i in range(0,(3-add_num)) :
    for i in range(0, 3):
        plots_t.MAX_EPOCHS = num_epoch + 10*i
#        MAX_EPOCHS = 20 + 10*i
        train_df = curve.iloc[i]['tr']
        val_df = curve.iloc[i]['va']
        #val_df = curve.iloc[i]['tr']
        test_df = curve.iloc[i]['te']

        #print(f'****** i =   {i}')

        window.train_df = train_df
        window.val_df = val_df
        window.test_df = test_df

        if i >= add_num :
            hist = compile.fit_2(model, window, patience= npat, EPOCHS = 'True' )

            train_loss += hist.history['loss'][-1:]
            if 'val_loss' in hist.history:
                print("val_loss not in the list")
                val_loss += hist.history['val_loss'][-1:]

            train_mae += hist.history['mean_absolute_error'][-1:]
            if 'val_mean_absolute_error' in hist.history:
                print("val_mean_absolute_error not in the list")
                val_mae += hist.history['val_mean_absolute_error'][-1:]

            eval(hist)

            window_t.train_df = train_df
            window_t.val_df = val_df
            window_t.test_df = test_df

            #window_t.plot_batch(model, dset_name='train', n_batch=10 )
            window_t.plot_xy(model, n_batch=30)

            subplot_title = (df_name+str(i))
            plt.gca().set_title(subplot_title)

            #plt.text(0.9, 0.1, "subtitle".format(subplot_title))
            #plt.gca().text(0.9, 0.1,  subplot_title)


    #    window_t.plot_batch(model, dset_name='val', n_batch=10, figures=0)

#    window_t.plot_batch(model, dset_name='test', n_batch=10, figures=0)

    train_loss = [x/4 for x in train_loss]
    val_loss = [x/4 for x in val_loss]
    train_mae = [x/4 for x in train_mae]
    val_mae = [x/4 for x in val_mae]

    return train_loss, val_loss, train_mae, val_mae

#%%

def plot_all(tr, model=None, window_t=None, set_name='default',
             marker2='x', edgecolor2 ='b', ax=None ) :
    ax = plt.gca()
    train_df = tr
    window_t.train_df = train_df

    dataset = window_t.train
    for i, batch in enumerate(dataset) :
      inputs, targets, lab = batch
      predictions = model(inputs)
      ax.scatter(lab[:,:,0], predictions[:,:,0], marker=marker2
              , edgecolor =edgecolor2 )

    ax.text(lab[0,0,0], predictions[0,0,0], "Curve{}".format(set_name), fontsize=14)
