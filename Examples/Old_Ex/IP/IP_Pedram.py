import cProfile
import pstats
import numpy as np
import random
import sys
# from Updating_Zero_LevelSets import updating_BF_LV
# from Enum_module_Lyap import Finding_Lyapunov_function 
# from Enum_module_BF import Finding_Barrier, Finding_Lyap_Invariant
# from preprocessing_LF import preprocessing_Lyap
# from utils_n_old import checking_sloution
# sys.path.insert(0, 'C:/Users/psa254/OneDrive - University of Kentucky/Desktop/Codes/UIS/UIS_Module')  # Add the package directory to sys.path
from Enum_module_BF import Finding_Barrier
import time
from plot_res_Lyap import plotting_results_lyap,plot_invariant_set,plot_level_set,plot_polytope_2D,plot_polytope
import matplotlib.pyplot as plt
# from invarint_PWA import finding_PWA_Invariat_set
# from Finding_Lyapunov import finding_Lyapunov
# import Lyap_PostProcess
from Lyap_plot_sol import plot2d
import time
mode="Rapid_mode" 
parallel=False
# mode="Low_Ram"
# from memory_profiler import profile
if __name__=='__main__':
    with cProfile.Profile() as pr:
        NN_file="NN_files/model_2d_IP_8.pt"
        # NN_file="NN_files/Inverted_Penduluem20.xlsx"
        # NN_file="NN_files/model_2d_simple_5.pt"
        # NN_file="NN_files/Path_following_20.xlsx"
        # NN_file="NN_files/model_2d_Pedram3.pt"
        eps1=0.01
        eps2=0.01
        name="IP_Lyap"
        TH=3.14
        # V=Finding_Lyapunov_function(NN_file,name,eps1,eps2,TH,mode,parallel)
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
        # ax.contour(X[0],Y[0],Z[0],levels=[0],colors='red',linestyles='dashed')
        # ax.contour(X[1],Y[1],Z[1],levels=[0],colors='green',linestyles='dashed')
        # ax.contour(X[2],Y[2],Z[2],levels=[0],colors='black',linestyles='dashed')
            # ax.clabel(CS1, inline=1, fontsize=10)
            # lines.append(CS1.collections[0])
            # labels.append(f"alpha={alpha[i]}")
            # contour.collections[0].set_label(f'a = {alph}')  # Set the label on the contour line
        # plt.legend([plt.Rectangle((0,0),1,2,color='brown',fill=False,linewidth = 2,linestyle='--'),plt.Rectangle((0,0),1,2,color='k',fill=False,linewidth = 2),plt.Rectangle((0,0),1,2,color='k',fill=False,linewidth = 2,linestyle='-')]\
        #    ,[fr'$\alpha={alph}$' for alph in alpha],loc='upper right',fontsize=14)
        # plt.legend(lines, labels)
        # plot_level_set(V,TH,'cyan',[18])
        plot_polytope_2D(NN_file,TH)
        plt.title(fr'$\alpha={alpha}$')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.show()
        print("Time for four different alphas:",time_end-time_start)






