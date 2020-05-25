##Importation des modules
import matplotlib.pyplot as plt
import numpy as np
from math import cos,sin

##fonction ouverture de fichier

def f_ouverture_fichier(nom_du_fichier):
    fichier = open(nom_du_fichier,"r")
    L_ligne = fichier.readlines()
    fichier.close()
    return L_ligne

##Reponse fonctionnelle
'''
on recupere les fichier textes qui contienne le nombre de proies tuer en fonction du nombre de proies et on trace les resultats sur un graphique
'''
banc_poisson_1 = "C:\\Users\\SpacetacuS\\Desktop\\TIPE\\TIPE\\modele individu centré\\Modele Mark 6 et 6.5\\banc_poisson_1.txt"

pas_banc_poisson_3_1 = "C:\\Users\\SpacetacuS\\Desktop\\TIPE\\TIPE\\modele individu centré\\Modele sans banc de poisson\\pas_banc_poisson_3.1_fonctionnelle.txt"

banc_poisson_7_1 = "C:\\Users\\SpacetacuS\\Desktop\\TIPE\\TIPE\\\modele individu centré\\Modele Mark 7 et 7.5\\banc_poisson_fonctionnelle_7-1.txt"

def nettoyage(L_ligne):
    L_propre = []
    L_ligne.pop(0)
    for i in L_ligne:
        k = i
        k = k.strip('\n')
        k = k.split(",")
        k_N = float(k[0])
        k_f = float(k[1])
        L_propre.append([k_N,k_f])
    return np.array(L_propre)

def Trace_fonctionnelle(nom_du_fichier):
    L = f_ouverture_fichier(nom_du_fichier)
    temps = L.pop(-1)
    L = nettoyage(L)
    L_x = L[:,0]
    L_y = L[:,1]
    plt.plot(L_x,L_y,label = 'reponse fonctionnelle')
    plt.title(temps)
    plt.show()
    return


##Trajectoire
'''
Ici on recupere les positions des proies et predateur dans le temps et on les traces de differentes facon
'''
pas_banc_poisson_traj_2_1 = "C:\\Users\\SpacetacuS\\Desktop\\TIPE\\TIPE\\modele individu centré\\Modele sans banc de poisson\\pas_banc_poisson_traj_2-1.txt"

pas_banc_poisson_traj_3_1 = "C:\\Users\\SpacetacuS\\Desktop\\TIPE\\TIPE\\modele individu centré\\Modele sans banc de poisson\\pas_banc_poisson_traj_3-1.txt"

banc_poisson_traj_1 = "C:\\Users\\SpacetacuS\\Desktop\\TIPE\\TIPE\\modele individu centré\\Modele Mark 6 et 6.5\\banc_poisson_traj_1.txt"

banc_poisson_traj_2 = "C:\\Users\\SpacetacuS\\Desktop\\TIPE\\TIPE\\modele individu centré\\Modele Mark 6 et 6.5\\banc_poisson_traj_2.txt"

banc_poisson_traj_1_exemple_de_surround= "C:\\Users\\SpacetacuS\\Desktop\\TIPE\\TIPE\\modele individu centré\\Modele Mark 6 et 6.5\\banc_poisson_traj - exemple de surround.txt"

banc_poisson_traj_7_1 = "C:\\Users\\SpacetacuS\\Desktop\\TIPE\\TIPE\\\modele individu centré\\Modele Mark 7 et 7.5\\banc_poisson_traj_7-1.txt"

banc_poisson_traj_7_2 = "C:\\Users\\SpacetacuS\\Desktop\\TIPE\\TIPE\\\modele individu centré\\Modele Mark 7 et 7.5\\banc_poisson_traj_7-2.txt"


def recup_traj(L_ligne):
    L_trajx = []
    L_trajy = []
    L_dir = []
    L_trajpredx = []
    L_trajpredy = []
    L_L_last_coord = []
    L_ligne.pop(0)
    i = 0
    while str(L_ligne[i]) != 'L_trajy:\n':
        k = L_ligne[i]
        k = k.strip('\n')
        k = k.split(" , ")
        k.pop(-1)
        l = []
        for j in k:
            l.append(float(j))
        L_trajx.append(l)
        i += 1
    while str(L_ligne[i]) != 'L_dir:\n':
        k = L_ligne[i]
        k = k.strip('\n')
        k = k.split(" , ")
        k.pop(-1)
        l = []
        for j in k:
            l.append(float(j))
        L_trajy.append(l)
        i += 1
    while str(L_ligne[i]) != 'L_trajpredx:\n':
        k = L_ligne[i]
        k = k.strip('\n')
        k = k.split(" , ")
        k.pop(-1)
        l = []
        for j in k:
            l.append(float(j))
        L_dir.append(l)
        i +=1
    while str(L_ligne[i]) != 'L_trajpredy:\n':
        k = L_ligne[i]
        k = k.strip('\n')
        k = k.split(" , ")
        k.pop(-1)
        l = []
        for j in k:
            l.append(float(j))
        L_trajpredx.append(l)
        i += 1
    while str(L_ligne[i]) != 'L_L_last_coord:\n':
        k = L_ligne[i]
        k = k.strip('\n')
        k = k.split(" , ")
        k.pop(-1)
        l = []
        for j in k:
            l.append(float(j))
        L_trajpredy.append(l)
        i += 1
    while not 'tps' in L_ligne[i]:
        k = L_ligne[i]        
        k = k.strip('\n')
        k = k.split(" - ")
        for j in k:
            if j == '[]':
                L_L_last_coord.append([])
            if '[[' in j:
                j = j.strip('[array(')
                j = j.strip(')]')
                j = j.split(' ')
                while '' in j:
                    j.remove('')
                L_L_nbr = []
                for a in j:
                    a = a.split(", ")
                    L_nbr = [float(c) for c in a]
                    L_L_nbr.append(L_nbr)
                L_L_last_coord.append(np.array(L_L_nbr))
        i += 1
    return L_trajx,L_trajy,L_dir,L_trajpredx,L_trajpredy,L_L_last_coord

'''
Cette fonction trace les positions des proies et predateurs avec:
points bleu: position initiales des proies
fleche rouge: positions finales des proies (pointe dans leur direction finale
pointillés jaune: trajectoire des proies
cercle violet: rayon de repulsion autour des positions finales des proies
points rouges: positions initiale des predateurs
points violets: positions finale des predateurs
pointillés violet: trajectoire des predateurs
croix noir: lieu ou une proie c'est faite tuer par un predateur 
'''
def Trace_trajectoire(nom_du_fichier):
    L = f_ouverture_fichier(nom_du_fichier)
    L_trajx,L_trajy,L_dir,L_trajpredx,L_trajpredy,L_L_last_coord = recup_traj(L)
    temps = L.pop(-1)
    L_trajy.pop(0)
    L_dir.pop(0)
    L_trajpredx.pop(0)
    L_trajpredy.pop(0)
    
    L_px_ini = [k[0] for k in L_trajx]
    L_py_ini = [k[0] for k in L_trajy]
    plt.plot(L_px_ini,L_py_ini,'ob')
    
    for k in range(len(L_trajx)):
        plt.plot(L_trajx[k],L_trajy[k],'y,')
    
    L_px_fin = [k[-1] for k in L_trajx]
    L_py_fin = [k[-1] for k in L_trajy]
    
    L_len = [len(k) for k in L_trajx]
    max_len = max(L_len) 
    nbr_surv = L_len.count(max_len)
    
    for k in range(nbr_surv):
        tetak = L_dir[k][-1]
        x = L_px_fin[k]
        y = L_py_fin[k]
        plt.axes().arrow(x, y, cos(tetak)*0.01,sin(tetak)*0.01, head_width=0.5, head_length=0.5,fc='r')
    
    L_mx_ini = [k[0] for k in L_trajpredx]
    L_my_ini = [k[0] for k in L_trajpredy]
    plt.plot(L_mx_ini,L_my_ini,'or')
    
    for k in range(len(L_trajpredx)):
        plt.plot(L_trajpredx[k],L_trajpredy[k],'m,')
    
    L_mx_fin = [k[-1] for k in L_trajpredx]
    L_my_fin = [k[-1] for k in L_trajpredy]
    plt.plot(L_mx_fin,L_my_fin,'om')
    
    for k in L_L_last_coord:
        if not k == []:
            L_last_coordx = k[0,0]
            L_last_coordy = k[1,0]
            plt.plot(L_last_coordx,L_last_coordy,'+k')
    
    plt.title(temps)
    plt.axis("equal")
    plt.show()
    return

'''
Cette fonction trace les positions des proies et predateurs  sur plusieurs graphique afin d'ameliorer la visibilté avec:
points bleu: position initiales des proies
fleche rouge: positions finales des proies (pointe dans leur direction finale
pointillés jaune: trajectoire des proies
cercle violet: rayon de repulsion autour des positions finales des proies
points rouges: positions initiale des predateurs
points violets: positions finale des predateurs
pointillés violet: trajectoire des predateurs
croix noir: lieu ou une proie c'est faite tuer par un predateur 
'''

def Trace_trajectoire_fragmente(nom_du_fichier):
    L = f_ouverture_fichier(nom_du_fichier)
    L_trajx,L_trajy,L_dir,L_trajpredx,L_trajpredy,L_L_last_coord = recup_traj(L)
    temps = L.pop(-1)
    L_trajy.pop(0)
    L_dir.pop(0)
    L_trajpredx.pop(0)
    L_trajpredy.pop(0)
    
    L_len_p = [len(k) for k in L_trajx]
    max_len_p = max(L_len_p) 
    nbr_surv = L_len_p.count(max_len_p)
    len_pred = len(L_trajpredx[0])
    tm = max_len_p - len_pred
    
    plt.subplot(231)
    L_px_ini = [k[0] for k in L_trajx]
    L_py_ini = [k[0] for k in L_trajy]
    plt.plot(L_px_ini,L_py_ini,'ob')
    
    for k in range(len(L_trajx)):
        plt.plot(L_trajx[k][0:tm],L_trajy[k][0:tm],'y,')
    
    L_px_fin = [k[tm] for k in L_trajx]
    L_py_fin = [k[tm] for k in L_trajy]
    
    for k in range(len(L_trajx)):
        tetak = L_dir[k][tm-1]
        x = L_px_fin[k]
        y = L_py_fin[k]
        plt.arrow(x, y, cos(tetak)*0.01,sin(tetak)*0.01, head_width=0.5, head_length=0.5,fc='r')
    plt.axis("equal")
    for i in range(4):   #[tm + int(i*(len_pred/4)):tm + int((i+1)*(len_pred/4))]
        plt.subplot(230+i+2)
  
        nbr_surv_i = 0
        for k in L_len_p:
            if k >= tm + int((i+1)*(len_pred/4)):
                nbr_surv_i += 1
        
        L_trajx_loc = []
        L_trajy_loc = []
        L_dir_loc = []
        
        for k in range(nbr_surv_i):
            L_trajx_loc.append(L_trajx[k][tm + int(i*(len_pred/4)):tm + int((i+1)*(len_pred/4))])
            L_trajy_loc.append(L_trajy[k][tm + int(i*(len_pred/4)):tm + int((i+1)*(len_pred/4))])
            L_dir_loc.append(L_dir[k][tm + int(i*(len_pred/4))-1:tm + int((i+1)*(len_pred/4))-1])
        
        for j in range(len(L_len_p)):
            if  tm + int(i*(len_pred/4)) < L_len_p[j] < tm + int((i+1)*(len_pred/4)):
                L_trajx_loc.append(L_trajx[j][tm + int(i*(len_pred/4)):L_len_p[j]])
                L_trajy_loc.append(L_trajy[j][tm + int(i*(len_pred/4)):L_len_p[j]])
                L_dir_loc.append(L_dir[j][tm + int(i*(len_pred/4))-1:L_len_p[j]-1])
            

        L_px_ini = [k[0] for k in L_trajx_loc]
        L_py_ini = [k[0] for k in L_trajy_loc]
        plt.plot(L_px_ini,L_py_ini,'ob')
        
        for j in range(len(L_trajx_loc)):
            plt.plot(L_trajx_loc[j],L_trajy_loc[j],'y,')
        
        L_px_fin = [k[-1] for k in L_trajx_loc[:nbr_surv_i]]
        L_py_fin = [k[-1] for k in L_trajy_loc[:nbr_surv_i]]
        for k in range(nbr_surv_i):
            tetak = L_dir[k][int((i+1)*(len_pred/4))-1]
            x = L_px_fin[k]
            y = L_py_fin[k]
            plt.arrow(x, y, cos(tetak)*0.01,sin(tetak)*0.01, head_width=0.5, head_length=0.5,fc='r')
        
        L_trajpredx_loc = [k[int(i*(len_pred/4)):int((i+1)*(len_pred/4))] for k in L_trajpredx]
        L_trajpredy_loc = [k[int(i*(len_pred/4)):int((i+1)*(len_pred/4))] for k in L_trajpredy]    
        
        
        L_mx_ini = [k[0] for k in L_trajpredx_loc]
        L_my_ini = [k[0] for k in L_trajpredy_loc]
        plt.plot(L_mx_ini,L_my_ini,'or')
        
        for k in range(len(L_trajpredx_loc)):
            plt.plot(L_trajpredx_loc[k],L_trajpredy_loc[k],'m,')
        
        L_mx_fin = [k[-1] for k in L_trajpredx_loc]
        L_my_fin = [k[-1] for k in L_trajpredy_loc]
        plt.plot(L_mx_fin,L_my_fin,'om')
        
        for k in L_L_last_coord[tm + int(i*(len_pred/4)):tm + int((i+1)*(len_pred/4))]:
            if not k == []:
                L_last_coordx = k[0,0]
                L_last_coordy = k[1,0]
                plt.plot(L_last_coordx,L_last_coordy,'+k')
        plt.axis("equal")
    plt.title(temps)
    plt.axis("equal")
    plt.show()
    return

'''
Cette fonction affiche le tracé dans le temps de facon dynamique on voit donc les proies et predateur bouger
les croix noir represente les endroits ou une proie c'est fait tuer
les flèches bleu represente les proies et pointe dans la direction qu'on les proies
les points rouge sont les predateurs
'''
def Trace_dynamique(nom_du_fichier):
    L = f_ouverture_fichier(nom_du_fichier)
    L_trajx,L_trajy,L_dir,L_trajpredx,L_trajpredy,L_L_last_coord = recup_traj(L)
    temps = L.pop(-1)
    L_trajy.pop(0)
    L_dir.pop(0)
    L_trajpredx.pop(0)
    L_trajpredy.pop(0)
    
    L_len_p = [len(k) for k in L_trajx]
    max_len_p = max(L_len_p) 
    nbr_surv = L_len_p.count(max_len_p)
    len_pred = len(L_trajpredx[0])
    tm = max_len_p - len_pred              #Attention ici tm=n tel que tm = t0 + n*dt

    for i in range(max_len_p):
        graph = plt.figure(1)
        
        nbr_surv_i = 0
        for j in L_len_p:
            if j>i:
                nbr_surv_i +=1
            
        L_px_i = [k[i] for k in L_trajx[:nbr_surv_i]]
        L_py_i = [k[i] for k in L_trajy[:nbr_surv_i]]
        
        if i == 0:
            plt.plot(L_px_i,L_py_i,'ob')
        else :
            L_pdir_i = [k[i-1] for k in L_dir[:nbr_surv_i]]
            
            for k in range(nbr_surv_i):
                tetak = L_pdir_i[k]
                x = L_px_i[k]
                y = L_py_i[k]
                plt.axes().arrow(x, y, cos(tetak)*0.01,sin(tetak)*0.01, head_width=0.5, head_length=0.5,fc='b')
        
        if i >= tm:
            L_mx_i = [k[i-tm] for k in L_trajpredx]
            L_my_i = [k[i-tm] for k in L_trajpredy]
            plt.plot(L_mx_i,L_my_i,'or')
        if i >= tm:
            for k in L_L_last_coord[:i-tm]:
                if not k == []:
                    L_last_coordx = k[0,0]
                    L_last_coordy = k[1,0]
                    plt.plot(L_last_coordx,L_last_coordy,'+k')
            
        plt.xlim(0,50)
        plt.ylim(0,50)
        plt.pause(0.001)
        graph.clear()
    return
    






