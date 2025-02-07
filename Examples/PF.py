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
        NN_file="NN_files/model_Dai.pt"
        # NN_file="NN_files/model_2d_Pedram3.pt"
        eps1=0.01
        eps2=1e-04
        name="Path_following"
        TH=0.8
        eps1=1e-2
        eps2=1e-4
        name="IP_BF"
        TH=3.0
        def random_color():
            return (random.random(), random.random(), random.random())
        alpha=[0.003,0.004]
        fig, ax = plt.subplots()
        X=[]
        Y=[]
        Z=[]
        time_start=time.time()
        for alph in alpha:
            NN,h,all_hyperplanes,all_bias,W,c,enumerate_poly,D,border_hype,border_bias,zeros=Finding_Barrier(NN_file,name,eps1,eps2,TH,mode,parallel,alph)
            # zeros=[]
            x,y,z=plot_invariant_set(h,zeros,TH,alph,color=[random_color()])
            X.append(x)
            Y.append(y)
            Z.append(z)
            CS=ax.contour(X[0],Y[0],Z[alpha.index(alph)],levels=[0],colors=[random_color()],linestyles=':')
            plt.clabel(CS, inline=True, fmt={0: fr'$\alpha={alph:.3f}$'}, fontsize=8) 

        # plt.legend([f"$\alpha={a}" for a in alpha])
        time_end=time.time()

        lines=[]
        labels=[]
        Z_new=np.max(np.array(Z),axis=0)
        ax.contour(X[0],Y[0],Z_new,levels=[0],colors='red',linestyles='solid')
        plt.legend([plt.Rectangle((0,0),1,2,color='r',fill=False,linewidth = 2,linestyle='solid')]\
           ,[fr'UIS Invariant Set'],loc='upper right',fontsize=14)
        plot_polytope_2D(NN_file,TH)
        plt.title(fr'$\alpha={alpha}$')
        plt.xlabel('Distance Error')
        plt.ylabel('Angle Error')
        plt.show()
        print("Time for four different alphas:",time_end-time_start)
        



