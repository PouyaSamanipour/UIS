import cProfile
import pstats
import numpy as np
import random
import sys
from UIS_Module.plot_res_Lyap import plot_invariant_set,plot_level_set,plot_polytope_2D,plot_polytope
from UIS_Module import Finding_Barrier
import time
import matplotlib.pyplot as plt
import time
mode="Rapid_mode" 
parallel=False
# mode="Low_Ram"
# from memory_profiler import profile
if __name__=='__main__':
    with cProfile.Profile() as pr:
        NN_file="NN_files/model_2d_IP_8.pt"
        eps1=0.01
        eps2=0.01
        name="IP_Lyap"
        TH=3.14
        eps1=0.01
        eps2=0.01
        name="IP_BF"
        def random_color():
            return (random.random(), random.random(), random.random())
        colors=["brown","black","green"]
        TH=3.14
        alpha=[0.025,0.05,0.06]
        # alpha=[0.005]
        X=[]
        Y=[]
        Z=[]
        fig, ax = plt.subplots()
        time_start=time.time()
        index=0
        for alph in alpha:

            NN,h,all_hyperplanes,all_bias,W,c,enumerate_poly,D,border_hype,border_bias,zeros=Finding_Barrier(NN_file,name,eps1,eps2,TH,mode,parallel,alph)
            x,y,z=plot_invariant_set(h,zeros,TH,alph,color=[random_color()])
            X.append(x)
            Y.append(y)
            Z.append(z)
            CS=ax.contour(X[0],Y[0],Z[alpha.index(alph)],levels=[0],colors=colors[index],linestyles='--',linewidths=2)
            plt.clabel(CS, inline=True, fmt={0: fr'$\alpha={alph:.3f}$'}, fontsize=8) 
            index=index+1

        # plt.legend([f"$\alpha={a}" for a in alpha])
        time_end=time.time()

        lines=[]
        labels=[]
        Z_new=np.max(np.array(Z),axis=0)
        ax.contour(X[0],Y[0],Z_new,levels=[0],colors='red',linestyles='solid')
        plt.legend([plt.Rectangle((0,0),1,2,color='r',fill=False,linewidth = 2,linestyle='solid'),plt.Rectangle((0,0),1,2,color='brown',fill=False,linewidth = 2,linestyle='--'),plt.Rectangle((0,0),1,2,color='k',fill=False,linewidth = 2),plt.Rectangle((0,0),1,2,color='g',fill=False,linewidth = 2,linestyle='-')]\
           ,[fr'UIS',r'$S(\mathcal{P}_1,\alpha_1$)',r'$S(\mathcal{P}_2,\alpha_2$)',r'$S(\mathcal{P}_3,\alpha_3$)'],loc='upper right',fontsize=14)
        plot_polytope_2D(NN_file,TH)
        plt.title(fr'$\alpha={alpha}$')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.show()
        print("Time for four different alphas:",time_end-time_start)






