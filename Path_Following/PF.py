import cProfile
import pstats
import random
import numpy as np
from Updating_Zero_LevelSets import updating_BF_LV
from Enum_module_Lyap import Finding_Lyapunov_function 
from Enum_module_BF import Finding_Barrier, Finding_Lyap_Invariant
from preprocessing_LF import preprocessing_Lyap
from utils_n_old import checking_sloution
import time
from plot_res_Lyap import plotting_results_lyap,plot_invariant_set,plot_level_set,plot_polytope_2D,plot_polytope
import matplotlib.pyplot as plt
from invarint_PWA import finding_PWA_Invariat_set
from Finding_Lyapunov import finding_Lyapunov
import Lyap_PostProcess
from Lyap_plot_sol import plot2d
mode="Rapid_mode" 
parallel=False
# mode="Low_Ram"
# from memory_profiler import profile
if __name__=='__main__':
    with cProfile.Profile() as pr:
        # NN_file="NN_files/model_2d_IP_8.pt"
        # NN_file="NN_files/Inverted_Penduluem20.xlsx"
        # NN_file="NN_files/model_2d_simple_5.pt"
        NN_file="NN_files/model_Dai.pt"
        # NN_file="NN_files/model_2d_Pedram3.pt"
        eps1=0.01
        eps2=1e-04
        name="Path_following"
        TH=0.8
        # V=Finding_Lyapunov_function(NN_file,name,eps1,eps2,TH,mode,parallel)
        # plot_level_set(V,TH,'green',[21])
        eps1=1e-2
        eps2=1e-4
        name="IP_BF"
        TH=3.0
        def random_color():
            return (random.random(), random.random(), random.random())
        # TH=3.14
        # alpha=[0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.18,0.19,0.2,0.3,0.4,0.5,0.6,0.7]
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
        # ax.contour(X[0],Y[0],Z[0],levels=[0],colors='red',linestyles='dashed')
        # ax.contour(X[1],Y[1],Z[1],levels=[0],colors='green',linestyles='dashed')
        # ax.contour(X[2],Y[2],Z[2],levels=[0],colors='black',linestyles='dashed')
            # ax.clabel(CS1, inline=1, fontsize=10)
            # lines.append(CS1.collections[0])
            # labels.append(f"alpha={alpha[i]}")
            # contour.collections[0].set_label(f'a = {alph}')  # Set the label on the contour line
        # plt.legend([plt.Rectangle((0,0),1,2,color='r',fill=False,linewidth = 2,linestyle='--'),plt.Rectangle((0,0),1,2,color='g',fill=False,linewidth = 2),plt.Rectangle((0,0),1,2,color='k',fill=False,linewidth = 2,linestyle='-')]\
        #    ,[fr'$\alpha={alph}$' for alph in alpha],loc='upper right',fontsize=14)
        # plt.legend(lines, labels)
        # plot_level_set(V,TH,'cyan',[18])
        plot_polytope_2D(NN_file,TH)
        plt.title(fr'$\alpha={alpha}$')
        plt.xlabel('Distance Error')
        plt.ylabel('Angle Error')
        plt.show()
        print("Time for four different alphas:",time_end-time_start)
        # Refined_polytope,new_hype,new_bias,A_dyn,B_dyn,all_hype,all_b=updating_BF_LV(NN,h,enumerate_poly,D)
        # h_sol=np.hstack((all_hype,all_b.reshape((len(all_b),1))))
        # h_sol=h_sol.reshape(-1)
        # Refined_polytope,A_new,B_new=finding_PWA_Invariat_set(h_sol,Refined_polytope,A_dyn,B_dyn,2)
        # # plot_invariant_set(h,TH,'cyan')
        # # plot_polytope(Refined_polytope, name)
        # eps1=0.01
        # eps2=0.01
        # name="IP_BF_Lyap"
        # TH=3.14
        # n=2
        # # V_lyp1,A_lyap1,H_lyap1,sol_Lyap1,A_PD2,id_var2=finding_Lyapunov(Refined_polytope,A_new,n,B_new,eps1,eps2,Threshold=0.099)

        # V_final,_,_,_,_,_,_,_,_=Finding_Lyap_Invariant(NN_file,h,Refined_polytope,all_hyperplanes,all_bias,border_hype,border_bias,new_hype,new_bias,W,c,eps1,eps2,TH,parallel)

        # # plot_level_set(V,TH,'green',[20])
        # # min_val,max_val,ls,sol_n2,list_points,levset_pts=Lyap_PostProcess.sol_Process(sol_Lyap1,A_PD2,id_var2,n,V_lyp1,len(V_lyp1))
        # # plot2d(A_lyap1,sol_n2,len(V_lyp1),H_lyap1,1,list_points,levset_pts,V_lyp1,'red',"level set after finding the safe set")
        # plot_level_set(V_final,TH,'red',[1,2,2.5,3])
        # plot_invariant_set(h,TH,'cyan')
        # plot_polytope_2D(NN_file,TH)
        # plt.show()




